"""Free WebRTC dev harness (Weeks 1-5): browser<->browser call, server taps audio.

Two browsers open ``call.html``, grant mic access, and join the same room. Each
browser holds one WebRTC peer connection to this FastAPI + aiortc server. The
server:

1. receives every participant's audio track,
2. resamples it to **16 kHz mono PCM**, and
3. exposes it as an async generator of :class:`PcmFrame` -- the same
   :class:`PcmFrameSource` contract that ``live_call/media_handler.py`` will
   implement for Twilio Media Streams in Week 6.

So all streaming-inference and alert code can be built and tested here for free,
then ported to Twilio unchanged. As a convenience the server also relays an
already-present participant's audio to a new joiner (best-effort SFU) so you can
hear the call; see the renegotiation caveat in ``webrtc_harness`` README.

Run::

    uvicorn live_call.webrtc_harness.rtc_server:app --host 0.0.0.0 --port 8000

then open http://localhost:8000/ in two tabs, join the same room, and Connect.

This module is imported only in the live-call runtime (needs aiortc/av/fastapi),
never in CI.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import suppress
from pathlib import Path

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay
from aiortc.mediastreams import MediaStreamTrack
from av.audio.resampler import AudioResampler
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from live_call.audio_source import (
    TARGET_CHANNELS,
    TARGET_SAMPLE_RATE,
    PcmFrame,
    PcmFrameSource,
)

logger = logging.getLogger("rtc_harness")

HERE = Path(__file__).resolve().parent
CALL_HTML = HERE / "call.html"

# How much tapped audio to buffer per room before dropping the oldest frame.
# (A slow/absent consumer must never grow memory without bound.)
_ROOM_QUEUE_MAXSIZE = 200


class Room:
    """One call room: its peer connections, an audio relay, and a tap queue."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.relay = MediaRelay()
        self.pcs: set[RTCPeerConnection] = set()
        # Most-recent relayed audio track per participant, for best-effort SFU.
        self.audio_tracks: list[MediaStreamTrack] = []
        # Fan-in of every participant's resampled 16 kHz mono PCM.
        self.queue: asyncio.Queue[PcmFrame] = asyncio.Queue(maxsize=_ROOM_QUEUE_MAXSIZE)

    def source(self) -> PcmFrameSource:
        """Return a :class:`PcmFrameSource` view over this room's tapped audio."""
        return _RoomSource(self)


class _RoomSource:
    """Adapts a :class:`Room`'s queue to the shared :class:`PcmFrameSource`."""

    def __init__(self, room: Room) -> None:
        self._room = room

    async def frames(self) -> AsyncIterator[PcmFrame]:
        while True:
            yield await self._room.queue.get()


_rooms: dict[str, Room] = {}


def get_room(name: str) -> Room:
    """Get or create the room with ``name``."""
    room = _rooms.get(name)
    if room is None:
        room = Room(name)
        _rooms[name] = room
        logger.info("created room %s", name)
    return room


async def _tap_track(track: MediaStreamTrack, room: Room) -> None:
    """Read a participant's audio track, resample to 16 kHz mono, enqueue PCM."""
    resampler = AudioResampler(
        format="s16", layout="mono" if TARGET_CHANNELS == 1 else "stereo", rate=TARGET_SAMPLE_RATE
    )
    try:
        while True:
            frame = await track.recv()
            for resampled in resampler.resample(frame):
                pcm = bytes(resampled.planes[0])
                item = PcmFrame(pcm=pcm)
                if room.queue.full():
                    with suppress(asyncio.QueueEmpty):
                        room.queue.get_nowait()  # drop oldest, keep the stream live
                await room.queue.put(item)
    except Exception as exc:  # track ended / peer gone
        logger.info("tap for room %s ended: %s", room.name, exc)


app = FastAPI(title="WebRTC dev harness")


@app.get("/")
async def index() -> HTMLResponse:
    """Serve the minimal two-party call page."""
    return HTMLResponse(CALL_HTML.read_text(encoding="utf-8"))


@app.post("/offer")
async def offer(request: Request) -> JSONResponse:
    """WebRTC signaling: accept an SDP offer, tap audio, return the SDP answer."""
    params = await request.json()
    room = get_room(params.get("room", "default"))

    pc = RTCPeerConnection()
    room.pcs.add(pc)

    @pc.on("connectionstatechange")
    async def _on_state() -> None:
        logger.info("room %s pc state -> %s", room.name, pc.connectionState)
        if pc.connectionState in {"failed", "closed", "disconnected"}:
            await _close_pc(pc, room)

    @pc.on("track")
    def _on_track(track: MediaStreamTrack) -> None:
        if track.kind != "audio":
            return
        logger.info("room %s received audio track", room.name)
        relayed = room.relay.subscribe(track)
        room.audio_tracks.append(relayed)
        asyncio.ensure_future(_tap_track(room.relay.subscribe(track), room))

    # Best-effort: let this joiner hear a participant who is already in the room.
    for existing in room.audio_tracks:
        pc.addTrack(existing)

    await pc.setRemoteDescription(RTCSessionDescription(sdp=params["sdp"], type=params["type"]))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return JSONResponse({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})


async def _close_pc(pc: RTCPeerConnection, room: Room) -> None:
    with suppress(Exception):
        await pc.close()
    room.pcs.discard(pc)


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    """Close every peer connection on server shutdown."""
    for room in _rooms.values():
        await asyncio.gather(*(pc.close() for pc in room.pcs), return_exceptions=True)
        room.pcs.clear()


def _rms_s16(pcm: bytes) -> float:
    """Root-mean-square level of signed-16-bit little-endian PCM (stdlib only)."""
    import array
    import math

    if not pcm:
        return 0.0
    samples = array.array("h")
    samples.frombytes(pcm)
    return math.sqrt(sum(s * s for s in samples) / len(samples))


async def log_levels(room_name: str = "default", every: float = 1.0) -> None:
    """Demo consumer: print the RMS level of tapped audio ~once per second.

    Shows that audio is flowing browser -> server and that the PcmFrameSource
    works, without any model. Run it against a room from an async context.
    """
    room = get_room(room_name)
    source = room.source()
    acc, samples, last = 0.0, 0, 0.0
    async for frame in source.frames():
        n = frame.num_samples()
        acc += _rms_s16(frame.pcm) * n
        samples += n
        last += frame.duration_seconds()
        if last >= every and samples:
            logger.info("room %s mean RMS=%.1f", room_name, acc / samples)
            acc, samples, last = 0.0, 0, 0.0
