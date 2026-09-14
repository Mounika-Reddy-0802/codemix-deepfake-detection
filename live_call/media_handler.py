"""Twilio Media Streams -> 16 kHz PCM frames (owner SK).

A Twilio ``<Stream>`` opens a WebSocket to this server and sends JSON messages:
``connected``, then ``start`` (stream and call SIDs, custom parameters), then one
``media`` message per 20 ms carrying base64 **8 kHz mono G.711 mu-law**, then
``stop``. This module turns that into the same ``PcmFrameSource`` the WebRTC harness
produces, so the scorer, verdict engine and dashboard never know it was a phone.

That 8 kHz mu-law narrowband audio is the exact condition the channel-matched
adapter was trained and evaluated on (G.711 at 20 dB): the live demo is the
evaluation protocol, deployed.

Decoding is numpy only. mu-law uses the G.711 formula from ``channel_sim``;
upsampling to 16 kHz is a windowed-sinc interpolator, so CI needs no scipy.
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import numpy as np

from live_call.audio_source import PcmFrame

TWILIO_RATE = 8_000
TARGET_RATE = 16_000


def mulaw_bytes_to_float(payload: bytes) -> np.ndarray:
    """G.711 mu-law bytes (as sent on the wire, bit-inverted) to float32 in [-1, 1]."""
    codes = ~np.frombuffer(payload, dtype=np.uint8)
    sign = np.where(codes & 0x80, -1.0, 1.0)
    exponent = (codes >> 4) & 0x07
    mantissa = codes & 0x0F
    magnitude = ((mantissa.astype(np.int32) << 3) + 0x84) << exponent
    return (sign * (magnitude - 0x84) / 32768.0).astype(np.float32)


def float_to_mulaw_bytes(signal: np.ndarray) -> bytes:
    """Inverse of :func:`mulaw_bytes_to_float` (used to simulate Twilio in tests)."""
    pcm = np.clip(np.round(np.asarray(signal, dtype=np.float64) * 32768.0), -32768, 32767).astype(
        np.int32
    )
    sign = np.where(pcm < 0, 0x80, 0).astype(np.uint8)
    magnitude = np.minimum(np.abs(pcm), 32635) + 0x84
    exponent = np.clip(np.floor(np.log2(magnitude)).astype(np.int32) - 7, 0, 7)
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    codes = sign | (exponent.astype(np.uint8) << 4) | mantissa.astype(np.uint8)
    return (~codes).astype(np.uint8).tobytes()


def _halfband_kernel(taps: int = 31) -> np.ndarray:
    n = np.arange(taps) - (taps - 1) / 2
    kernel = np.sinc(n / 2) * np.hamming(taps)
    return (kernel / kernel.sum() * 2).astype(np.float32)


_KERNEL = _halfband_kernel()


class Upsampler2x:
    """Streaming 8 -> 16 kHz: zero-stuff then half-band low-pass, state kept across chunks."""

    def __init__(self) -> None:
        self._tail = np.zeros(len(_KERNEL) - 1, dtype=np.float32)

    def process(self, samples: np.ndarray) -> np.ndarray:
        if samples.size == 0:
            return samples.astype(np.float32)
        stuffed = np.zeros(samples.size * 2, dtype=np.float32)
        stuffed[::2] = samples
        joined = np.concatenate([self._tail, stuffed])
        out = np.convolve(joined, _KERNEL, mode="valid")
        self._tail = joined[-(len(_KERNEL) - 1) :]
        return out.astype(np.float32)


@dataclass
class StreamStart:
    stream_sid: str
    call_sid: str
    account_sid: str
    tracks: list[str]
    parameters: dict = field(default_factory=dict)


def parse_start(message: dict) -> StreamStart:
    start = message.get("start", {})
    return StreamStart(
        stream_sid=message.get("streamSid") or start.get("streamSid", ""),
        call_sid=start.get("callSid", ""),
        account_sid=start.get("accountSid", ""),
        tracks=list(start.get("tracks", [])),
        parameters=dict(start.get("customParameters", {})),
    )


class TwilioStreamSource:
    """A ``PcmFrameSource`` fed by a Twilio Media Streams WebSocket.

    The WebSocket handler calls :meth:`feed` with each decoded JSON message and
    :meth:`close` when the socket ends; the session consumes :meth:`frames`.
    """

    def __init__(self, max_queue: int = 500) -> None:
        self._queue: asyncio.Queue[PcmFrame | None] = asyncio.Queue(maxsize=max_queue)
        self._upsampler = Upsampler2x()
        self.start: StreamStart | None = None
        self.media_messages = 0

    def feed(self, message: dict) -> str:
        """Handle one Twilio message; returns its event name."""
        event = message.get("event", "")
        if event == "start":
            self.start = parse_start(message)
        elif event == "media":
            media = message.get("media", {})
            if media.get("track", "inbound") not in ("inbound", "inbound_track"):
                return event
            pcm8 = mulaw_bytes_to_float(base64.b64decode(media.get("payload", "")))
            pcm16 = self._upsampler.process(pcm8)
            frame = PcmFrame(
                pcm=(np.clip(pcm16, -1.0, 32767 / 32768) * 32768).astype("<i2").tobytes()
            )
            if self._queue.full():  # keep the stream live if scoring falls behind
                self._queue.get_nowait()
            self._queue.put_nowait(frame)
            self.media_messages += 1
        elif event == "stop":
            self.close()
        return event

    def feed_text(self, text: str) -> str:
        return self.feed(json.loads(text))

    def close(self) -> None:
        if self._queue.full():
            self._queue.get_nowait()
        self._queue.put_nowait(None)

    async def frames(self) -> AsyncIterator[PcmFrame]:
        while True:
            frame = await self._queue.get()
            if frame is None:
                return
            yield frame


def media_message(payload: bytes, stream_sid: str = "MZtest", chunk: int = 0) -> dict:
    """A Twilio ``media`` message, for tests and the local stream simulator."""
    return {
        "event": "media",
        "sequenceNumber": str(chunk + 2),
        "streamSid": stream_sid,
        "media": {
            "track": "inbound",
            "chunk": str(chunk + 1),
            "timestamp": str(20 * chunk),
            "payload": base64.b64encode(payload).decode(),
        },
    }
