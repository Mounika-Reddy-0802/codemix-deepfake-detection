"""Twilio Media Streams decoding (owner SK): mu-law, 8 -> 16 kHz, the stream protocol."""

from __future__ import annotations

import asyncio
import base64

import numpy as np

from live_call import media_handler as mh
from live_call.audio_source import PcmFrameSource

# G.711 reference points: code 0xFF is zero, 0x80 the largest positive, 0x00 the largest negative.
REFERENCE = {0xFF: 0, 0x7F: 0, 0x80: 32124, 0x00: -32124, 0xF0: 120, 0x70: -120}


def test_mulaw_decode_matches_g711_reference_points():
    codes = bytes(REFERENCE)
    decoded = mh.mulaw_bytes_to_float(codes) * 32768.0
    np.testing.assert_allclose(decoded, list(REFERENCE.values()), atol=0.5)


def test_mulaw_round_trip_is_close():
    t = np.arange(8000) / 8000
    signal = 0.5 * np.sin(2 * np.pi * 440 * t)
    back = mh.mulaw_bytes_to_float(mh.float_to_mulaw_bytes(signal))
    snr = 10 * np.log10(np.sum(signal**2) / np.sum((signal - back) ** 2))
    assert snr > 30  # G.711 is about 38 dB on a full sine


def test_upsampler_doubles_length_and_is_chunk_invariant():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1600).astype(np.float32) * 0.1
    whole = mh.Upsampler2x().process(x)
    chunked_up = mh.Upsampler2x()
    chunked = np.concatenate([chunked_up.process(x[i : i + 160]) for i in range(0, 1600, 160)])
    assert whole.size == chunked.size == 3200
    np.testing.assert_allclose(whole, chunked, atol=1e-5)


def test_upsampler_keeps_a_tone_and_suppresses_its_image():
    t = np.arange(4000) / 8000
    y = mh.Upsampler2x().process(np.sin(2 * np.pi * 1000 * t).astype(np.float32))
    spectrum = np.abs(np.fft.rfft(y[200:7800] * np.hanning(7600)))
    freqs = np.fft.rfftfreq(7600, 1 / 16000)
    assert abs(freqs[np.argmax(spectrum)] - 1000) < 5
    image = spectrum[np.argmin(np.abs(freqs - 7000))]
    assert 20 * np.log10(image / spectrum.max()) < -40


def _twilio_session(seconds: float = 1.0):
    t = np.arange(int(8000 * seconds)) / 8000
    payload = mh.float_to_mulaw_bytes(0.3 * np.sin(2 * np.pi * 300 * t))
    messages = [
        {"event": "connected", "protocol": "Call", "version": "1.0.0"},
        {
            "event": "start",
            "streamSid": "MZ1",
            "start": {
                "streamSid": "MZ1",
                "callSid": "CA1",
                "accountSid": "AC1",
                "tracks": ["inbound"],
                "customParameters": {"conference": "call-CA1", "from": "+911234"},
            },
        },
    ]
    messages += [
        mh.media_message(payload[i : i + 160], "MZ1", n)
        for n, i in enumerate(range(0, len(payload), 160))
    ]
    messages.append({"event": "stop", "streamSid": "MZ1", "stop": {"callSid": "CA1"}})
    return messages


def test_stream_source_parses_start_and_yields_16k_frames():
    source = mh.TwilioStreamSource()
    assert isinstance(source, PcmFrameSource)
    for message in _twilio_session(1.0):
        source.feed(message)
    assert source.start.call_sid == "CA1" and source.start.parameters["conference"] == "call-CA1"

    async def collect():
        return [f async for f in source.frames()]

    frames = asyncio.run(collect())
    assert len(frames) == 50  # 20 ms messages for 1 s
    assert sum(f.num_samples() for f in frames) == 16000
    assert all(f.sample_rate == 16000 for f in frames)


def test_outbound_track_media_is_ignored():
    source = mh.TwilioStreamSource()
    message = mh.media_message(b"\xff" * 160)
    message["media"]["track"] = "outbound"
    source.feed(message)
    assert source.media_messages == 0


def test_media_message_payload_is_base64():
    message = mh.media_message(b"\x01\x02")
    assert base64.b64decode(message["media"]["payload"]) == b"\x01\x02"
