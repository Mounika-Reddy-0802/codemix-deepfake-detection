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
import itertools
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from pathlib import Path

from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
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
        #: peer connection id -> connection. Ids let a client renegotiate its own
        #: connection instead of opening a second one (see :func:`handle_offer`).
        self.pcs: dict[str, RTCPeerConnection] = {}
        #: peer connection id -> the audio that connection is currently sending.
        #: Keyed by peer, not appended to a list: a room that accumulated one track
        #: per past call would hand a new joiner more tracks than its offer has
        #: m-lines, and aiortc fails the whole negotiation.
        self.tracks: dict[str, MediaStreamTrack] = {}
        #: peer connection id -> ids of the source tracks already given to it.
        self.sent: dict[str, set[int]] = {}
        # Fan-in of every participant's resampled 16 kHz mono PCM.
        self.queue: asyncio.Queue[PcmFrame] = asyncio.Queue(maxsize=_ROOM_QUEUE_MAXSIZE)

    def live_tracks(self, except_pc: str) -> list[MediaStreamTrack]:
        """Audio currently being sent by the other connected peers."""
        return [t for pc_id, t in self.tracks.items() if pc_id != except_pc and pc_id in self.pcs]

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


class _TrackSource:
    """One participant's own audio as a :class:`PcmFrameSource`, ending when they leave.

    The room queue mixes every participant, which is fine for a level meter but
    wrong for detection: two voices interleaved frame by frame are neither speaker.
    The detector gets one of these per audio track instead.
    """

    def __init__(self) -> None:
        self.queue: asyncio.Queue[PcmFrame | None] = asyncio.Queue(maxsize=_ROOM_QUEUE_MAXSIZE)

    def put(self, item: PcmFrame | None) -> None:
        if self.queue.full():
            with suppress(asyncio.QueueEmpty):
                self.queue.get_nowait()
        self.queue.put_nowait(item)

    async def frames(self) -> AsyncIterator[PcmFrame]:
        while True:
            item = await self.queue.get()
            if item is None:
                return
            yield item


async def _tap_track(track: MediaStreamTrack, room: Room, own: _TrackSource | None = None) -> None:
    """Read a participant's audio track, resample to 16 kHz mono, enqueue PCM."""
    resampler = AudioResampler(
        format="s16", layout="mono" if TARGET_CHANNELS == 1 else "stereo", rate=TARGET_SAMPLE_RATE
    )
    try:
        while True:
            frame = await track.recv()
            for resampled in resampler.resample(frame):
                pcm = bytes(resampled.planes[0])[: resampled.samples * 2]
                item = PcmFrame(pcm=pcm)
                if room.queue.full():
                    with suppress(asyncio.QueueEmpty):
                        room.queue.get_nowait()  # drop oldest, keep the stream live
                await room.queue.put(item)
                if own is not None:
                    own.put(item)
    except Exception as exc:  # track ended / peer gone
        logger.info("tap for room %s ended: %s", room.name, exc)
    finally:
        if own is not None:
            own.put(None)


#: Called as ``on_track(room_name, participant_id, source)`` for every audio track.
TrackHook = Callable[[str, str, PcmFrameSource], None]
#: Called as ``on_renegotiate(room_name, pc_id)`` when a peer has to re-offer to hear
#: audio that arrived after it connected. Without it a two-party call is one-way:
#: whoever connects first never receives the track the second party adds later.
RenegotiateHook = Callable[[str, str], None]

_pc_ids = itertools.count(1)


def _forward(room: Room, pc_id: str, pc: RTCPeerConnection, track: MediaStreamTrack) -> bool:
    """Give ``track`` to ``pc`` unless it already has it. True if newly added."""
    seen = room.sent.setdefault(pc_id, set())
    if id(track) in seen:
        return False
    pc.addTrack(track)
    seen.add(id(track))
    return True


async def handle_offer(
    params: dict,
    on_track: TrackHook | None = None,
    on_renegotiate: RenegotiateHook | None = None,
) -> JSONResponse:
    """WebRTC signalling shared by this harness and the detection server.

    Pass the ``pc_id`` from an earlier answer to renegotiate that same connection
    after new audio has been added to it; omit it to create a new one. Set
    ``monitor`` false to carry a participant's audio without handing it to
    ``on_track``, which is how the receiver's own voice is kept out of the detector.
    """
    room = get_room(params.get("room", "default"))
    pc_id = str(params.get("pc_id") or "")
    monitor = bool(params.get("monitor", True))

    pc = room.pcs.get(pc_id)
    if pc is None:  # a new connection, not a renegotiation
        # A public STUN server lets a phone on mobile data reach a laptop behind home NAT.
        pc = RTCPeerConnection(
            RTCConfiguration(iceServers=[RTCIceServer(urls="stun:stun.l.google.com:19302")])
        )
        pc_id = f"pc{next(_pc_ids)}"
        room.pcs[pc_id] = pc
        this_id = pc_id

        @pc.on("connectionstatechange")
        async def _on_state() -> None:
            logger.info("room %s pc %s state -> %s", room.name, this_id, pc.connectionState)
            if pc.connectionState in {"failed", "closed", "disconnected"}:
                await _close_pc(pc, room)

        @pc.on("track")
        def _on_track(track: MediaStreamTrack) -> None:
            if track.kind != "audio":
                return
            logger.info("room %s pc %s received audio track", room.name, this_id)
            room.tracks[this_id] = room.relay.subscribe(track)
            own = _TrackSource() if (on_track is not None and monitor) else None
            asyncio.ensure_future(_tap_track(room.relay.subscribe(track), room, own))
            if own is not None and on_track is not None:
                on_track(room.name, this_id, own)
            # Give the new audio to everyone already in the room, then ask them to
            # re-offer, because a track added after connecting needs negotiating.
            for other_id, other in list(room.pcs.items()):
                if other_id == this_id:
                    continue
                if _forward(room, other_id, other, room.relay.subscribe(track)):
                    logger.info("room %s: pc %s must renegotiate", room.name, other_id)
                    if on_renegotiate is not None:
                        on_renegotiate(room.name, other_id)

        # Let this joiner hear whoever is already on the call. Only live peers, so a
        # fresh call never inherits tracks from calls that have ended.
        for existing in room.live_tracks(except_pc=pc_id):
            _forward(room, pc_id, pc, existing)

    await pc.setRemoteDescription(RTCSessionDescription(sdp=params["sdp"], type=params["type"]))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return JSONResponse(
        {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "pc_id": pc_id}
    )


app = FastAPI(title="WebRTC dev harness")


@app.get("/")
async def index() -> HTMLResponse:
    """Serve the minimal two-party call page."""
    return HTMLResponse(CALL_HTML.read_text(encoding="utf-8"))


@app.post("/offer")
async def offer(request: Request) -> JSONResponse:
    """WebRTC signaling: accept an SDP offer, tap audio, return the SDP answer."""
    return await handle_offer(await request.json())


async def _close_pc(pc: RTCPeerConnection, room: Room) -> None:
    with suppress(Exception):
        await pc.close()
    for pc_id, known in list(room.pcs.items()):
        if known is pc:
            del room.pcs[pc_id]
            room.sent.pop(pc_id, None)
            room.tracks.pop(pc_id, None)  # its audio leaves the room with it


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    """Close every peer connection on server shutdown."""
    for room in _rooms.values():
        await asyncio.gather(*(pc.close() for pc in room.pcs.values()), return_exceptions=True)
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
