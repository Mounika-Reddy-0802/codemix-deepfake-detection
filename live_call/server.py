"""The live-call deepfake detection server: web demo, WebRTC calls and Twilio (owner SK).

One FastAPI app serves everything the evaluation demo needs:

| Route | What it is |
|---|---|
| ``GET /`` | evaluator home: what the system does, live status, links |
| ``GET /dashboard`` | receiver dashboard: every live call, verdict light, score timeline, alerts |
| ``GET /call`` | WebRTC call page; a participant can speak or play a clip into the call |
| ``POST /offer`` | WebRTC signalling; every participant's audio becomes a monitored session |
| ``POST /api/predict`` | upload or record a clip, get a verdict and its window timeline |
| ``GET /api/demo-clips``, ``POST /api/replay/{id}`` | play a prepared real or cloned clip as a live call |
| ``GET /api/results`` | the measured S1/S2/S3 tables and the calibration, from committed JSONs |
| ``WS /ws/dashboard`` | push channel for the dashboard |
| ``POST /twilio/voice`` | Twilio webhook for an incoming call: stream it, bridge the receiver |
| ``POST /twilio/join``, ``/twilio/announce``, ``/twilio/status`` | receiver leg, receiver-only warning, call end |
| ``WS /twilio/media`` | Twilio Media Streams: 8 kHz mu-law caller audio |

Run::

    set DFD_CHECKPOINT=<data>/checkpoints/lora_norm_rvc_channel_best.pt
    uvicorn live_call.server:app --host 0.0.0.0 --port 8000

Twilio needs ``TWILIO_*``, ``RECEIVER_NUMBER`` and ``PUBLIC_BASE_URL`` in ``.env``
(see ``live_call/README.md``); without them every Twilio alert runs dry and is shown
on the dashboard instead.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import io
import json
import logging
import os
from base64 import b64encode
from pathlib import Path
from typing import Annotated
from xml.sax.saxutils import escape, quoteattr

import numpy as np
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from live_call.alerts import AlertDispatcher, announcement_twiml
from live_call.detector import DetectorService, DetectorSettings
from live_call.media_handler import TwilioStreamSource
from live_call.replay import FileSource
from live_call.session import CallSession, Hub, SessionInfo, SessionRegistry, new_session_id

log = logging.getLogger("live_call")

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
DEMO_CLIPS_DIR = Path(os.environ.get("DEMO_CLIPS_DIR", "demo_assets"))
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

RESULT_FILES = {
    "systems": "experiments/results/systems_s1_s2_s3.json",
    "ablation": "experiments/results/ablations.json",
    "reverse_degradation": "experiments/results/s3_reverse_degradation.json",
    "calibration": "experiments/results/threshold_calibration.json",
}


def _load_env_file(path: Path) -> None:
    """Minimal ``.env`` loader: KEY=VALUE lines, existing environment wins."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split(" #", 1)[0].strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class App:
    """Everything the routes share, created once per process."""

    def __init__(
        self, detector: DetectorService | None = None, dispatcher: AlertDispatcher | None = None
    ):
        _load_env_file(ROOT / ".env")
        self.detector = detector or DetectorService(DetectorSettings.from_env())
        self.dispatcher = dispatcher or AlertDispatcher()
        self.hub = Hub()
        self.sessions = SessionRegistry()
        self.tasks: set[asyncio.Task] = set()
        # Twilio call SID of the caller -> details needed to reach the receiver.
        self.twilio_calls: dict[str, dict] = {}

    def start_session(self, info: SessionInfo, source) -> CallSession:
        session = CallSession(
            info,
            self.detector.score_fn(),
            self.detector.config(),
            self.hub,
            self.dispatcher,
            self.detector.executor,
        )
        self.sessions.add(session)
        task = asyncio.create_task(session.run(source))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return session


# --------------------------------------------------------------------------- #
# Twilio request signature (https://www.twilio.com/docs/usage/security)
# --------------------------------------------------------------------------- #
def twilio_signature(auth_token: str, url: str, params: dict) -> str:
    payload = url + "".join(f"{k}{params[k]}" for k in sorted(params))
    digest = hmac.new(auth_token.encode(), payload.encode(), hashlib.sha1).digest()
    return b64encode(digest).decode()


def voice_twiml(stream_url: str, conference: str, caller: str, status_url: str) -> str:
    """Incoming call: stream the caller's own audio, then hold them in a conference."""
    return (
        '<?xml version="1.0" encoding="UTF-8"?><Response>'
        f'<Start><Stream url={quoteattr(stream_url)} track="inbound_track">'
        f'<Parameter name="conference" value={quoteattr(conference)}/>'
        f'<Parameter name="from" value={quoteattr(caller)}/>'
        "</Stream></Start>"
        '<Say voice="alice">Connecting your call.</Say>'
        f'<Dial><Conference startConferenceOnEnter="true" endConferenceOnExit="true" '
        f'statusCallback={quoteattr(status_url)} statusCallbackEvent="end">{escape(conference)}</Conference></Dial>'
        "</Response>"
    )


def join_twiml(conference: str) -> str:
    """Receiver leg: join the caller's conference."""
    return (
        '<?xml version="1.0" encoding="UTF-8"?><Response><Dial>'
        f'<Conference startConferenceOnEnter="true" endConferenceOnExit="true">{escape(conference)}</Conference>'
        "</Dial></Response>"
    )


def decode_audio(data: bytes, filename: str = "") -> np.ndarray:
    """Any browser or file format to 16 kHz mono float32 (wav/flac/ogg/webm/mp3/m4a)."""
    try:
        import soundfile as sf

        audio, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
    except Exception:
        import av

        container = av.open(io.BytesIO(data))
        resampler = av.AudioResampler(format="flt", layout="mono", rate=16000)
        chunks = []
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
        container.close()
        if not chunks:
            raise ValueError(f"no audio decoded from {filename or 'upload'}") from None
        return np.concatenate(chunks).astype(np.float32)
    if rate != 16000:
        from src.utils.audio_utils import resample

        audio = resample(audio, rate, 16000)
    return audio.astype(np.float32)


def warning_beep_wav(rate: int = 8000) -> bytes:
    """Two short tones as an 8 kHz WAV, built in memory so no audio file is shipped."""
    import wave

    t = np.arange(int(0.22 * rate)) / rate
    envelope = np.minimum(1.0, np.minimum(t, t[::-1]) / 0.02)
    tone = lambda f: 0.5 * np.sin(2 * np.pi * f * t) * envelope  # noqa: E731
    gap = np.zeros(int(0.08 * rate))
    signal = np.concatenate([tone(880), gap, tone(660), gap])
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes((signal * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


def list_demo_clips(directory: Path = DEMO_CLIPS_DIR) -> list[dict]:
    index = directory / "demo_clips.json"
    if not index.is_file():
        return []
    clips = json.loads(index.read_text(encoding="utf-8"))
    return [c for c in clips if (directory / c["file"]).is_file()]


def create_app(state: App | None = None) -> FastAPI:
    api = FastAPI(title="Code-mixed deepfake call detection")
    api.state.app = state or App()
    shared: App = api.state.app

    if STATIC.is_dir():
        api.mount("/static", StaticFiles(directory=STATIC), name="static")
    figures = ROOT / "experiments" / "figures"
    if figures.is_dir():
        api.mount("/figures", StaticFiles(directory=figures), name="figures")

    def page(name: str) -> HTMLResponse:
        return HTMLResponse((STATIC / name).read_text(encoding="utf-8"))

    @api.get("/")
    async def home() -> HTMLResponse:
        return page("index.html")

    @api.get("/dashboard")
    async def dashboard() -> HTMLResponse:
        return page("dashboard.html")

    @api.get("/call")
    async def call_page() -> HTMLResponse:
        return page("call.html")

    @api.get("/api/health")
    async def health() -> JSONResponse:
        return JSONResponse(
            {
                "detector": shared.detector.status(),
                "twilio": {
                    "configured": shared.dispatcher.live,
                    "public_base_url": shared.dispatcher.settings.public_base_url or None,
                },
                "live_sessions": sum(not s.ended for s in shared.sessions.all()),
                "dashboards": shared.hub.subscribers,
                "demo_clips": len(list_demo_clips()),
            }
        )

    @api.get("/api/results")
    async def results() -> JSONResponse:
        out = {}
        for key, rel in RESULT_FILES.items():
            path = ROOT / rel
            out[key] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
        return JSONResponse(out)

    @api.get("/api/sessions")
    async def sessions() -> JSONResponse:
        return JSONResponse([s.snapshot() for s in shared.sessions.all()])

    @api.get("/api/alerts")
    async def alerts() -> JSONResponse:
        return JSONResponse([a.__dict__ for a in shared.dispatcher.sent[-50:]])

    # ----------------------------------------------------------------- upload
    @api.post("/api/predict")
    async def predict(file: Annotated[UploadFile, File()]) -> JSONResponse:
        from src.inference.predict import Threshold, predict_audio

        data = await file.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "clip larger than 20 MB")
        try:
            audio = decode_audio(data, file.filename or "")
        except Exception as exc:
            raise HTTPException(400, f"could not decode audio: {exc}") from exc
        config = shared.detector.config()
        score_fn = shared.detector.score_fn()
        loop = asyncio.get_running_loop()
        threshold = Threshold(config.threshold, "configs/threshold.yaml", provisional=False)

        def run():
            from src.inference.streaming import StreamingScorer

            scorer = StreamingScorer(score_fn)
            samples = audio
            if 0 < samples.size < scorer.windower.window:
                samples = np.pad(samples, (0, scorer.windower.window - samples.size))
            windows = [
                {
                    "t": r.end_seconds,
                    "score": r.score,
                    "smoothed": r.smoothed,
                    "level": round(r.level_dbfs, 1),
                }
                for r in scorer.push(samples)
            ]
            return predict_audio(
                audio, score_fn, threshold, path=file.filename or ""
            ).as_dict(), windows

        prediction, windows = await loop.run_in_executor(shared.detector.executor, run)
        return JSONResponse({**prediction, "window_scores": windows})

    # ----------------------------------------------------------------- replay
    @api.get("/api/demo-clips")
    async def demo_clips() -> JSONResponse:
        return JSONResponse(
            [{k: v for k, v in c.items() if k != "source"} for c in list_demo_clips()]
        )

    @api.get("/api/demo-clips/{clip_id}/audio")
    async def demo_clip_audio(clip_id: str) -> FileResponse:
        clip = next((c for c in list_demo_clips() if c["id"] == clip_id), None)
        if clip is None:
            raise HTTPException(404, "unknown clip")
        return FileResponse(DEMO_CLIPS_DIR / clip["file"])

    @api.post("/api/replay/{clip_id}")
    async def replay_clip(clip_id: str) -> JSONResponse:
        clip = next((c for c in list_demo_clips() if c["id"] == clip_id), None)
        if clip is None:
            raise HTTPException(404, "unknown clip")
        audio = decode_audio((DEMO_CLIPS_DIR / clip["file"]).read_bytes(), clip["file"])
        info = SessionInfo(
            new_session_id("replay"),
            "replay",
            clip["title"],
            meta={"clip": clip_id, "truth": clip["label"], "tool": clip.get("tool")},
        )
        session = shared.start_session(info, FileSource(audio, realtime=True))
        return JSONResponse({"session": session.info.session_id})

    # ----------------------------------------------------------------- dashboard push
    @api.websocket("/ws/dashboard")
    async def dashboard_ws(ws: WebSocket) -> None:
        await ws.accept()
        queue = shared.hub.subscribe()
        try:
            await ws.send_json({"type": "hello", "health": (await health()).body.decode()})
            for message in shared.sessions.replay_history():
                await ws.send_json({**message, "replayed": True})
            while True:
                await ws.send_json(await queue.get())
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            shared.hub.unsubscribe(queue)

    # ----------------------------------------------------------------- WebRTC
    @api.post("/offer")
    async def offer(request: Request) -> JSONResponse:
        from live_call.webrtc_harness import rtc_server

        params = await request.json()
        role = params.get("role", "caller")
        name = params.get("name") or role

        def on_track(room_name: str, participant: str, source) -> None:
            info = SessionInfo(
                new_session_id("webrtc"),
                "webrtc",
                f"{name} in room {room_name}",
                meta={"room": room_name, "participant": participant, "role": role},
            )
            shared.start_session(info, source)

        return await rtc_server.handle_offer(params, on_track=on_track)

    # ----------------------------------------------------------------- Twilio
    async def twilio_form(request: Request) -> dict:
        form = dict(await request.form())
        token = shared.dispatcher.settings.auth_token
        if shared.dispatcher.live and os.environ.get("TWILIO_VALIDATE", "1") != "0":
            base = shared.dispatcher.settings.public_base_url
            url = base + request.url.path + (f"?{request.url.query}" if request.url.query else "")
            expected = twilio_signature(token, url, form)
            if not hmac.compare_digest(expected, request.headers.get("X-Twilio-Signature", "")):
                raise HTTPException(403, "invalid Twilio signature")
        return form

    def ws_base() -> str:
        base = shared.dispatcher.settings.public_base_url
        return base.replace("https://", "wss://").replace("http://", "ws://")

    @api.post("/twilio/voice")
    async def twilio_voice(request: Request) -> Response:
        form = await twilio_form(request)
        call_sid = form.get("CallSid", "unknown")
        conference = f"call-{call_sid}"
        shared.twilio_calls[call_sid] = {"conference": conference, "from": form.get("From", "")}
        base = shared.dispatcher.settings.public_base_url
        twiml = voice_twiml(
            f"{ws_base()}/twilio/media", conference, form.get("From", ""), f"{base}/twilio/status"
        )
        if shared.dispatcher.live:

            async def bring_receiver() -> None:
                try:
                    leg = await shared.dispatcher.rest.call_receiver_into(conference)
                    shared.twilio_calls[call_sid]["receiver_call_sid"] = leg.get("sid")
                    for session in shared.sessions.all():
                        if session.info.meta.get("conference") == conference:
                            session.info.meta["receiver_call_sid"] = leg.get("sid")
                except Exception as exc:
                    log.error("could not dial the receiver: %s", exc)

            task = asyncio.create_task(bring_receiver())
            shared.tasks.add(task)
            task.add_done_callback(shared.tasks.discard)
        return Response(twiml, media_type="application/xml")

    @api.post("/twilio/join")
    async def twilio_join(request: Request, conference: str) -> Response:
        await twilio_form(request)
        return Response(join_twiml(conference), media_type="application/xml")

    @api.post("/twilio/announce")
    async def twilio_announce(request: Request, kind: str = "beep") -> Response:
        await twilio_form(request)
        beep = f"{shared.dispatcher.settings.public_base_url}/twilio/beep.wav"
        return Response(announcement_twiml(kind, beep), media_type="application/xml")

    @api.get("/twilio/beep.wav")
    async def twilio_beep() -> Response:
        return Response(warning_beep_wav(), media_type="audio/wav")

    @api.post("/twilio/status")
    async def twilio_status(request: Request) -> Response:
        form = await twilio_form(request)
        shared.hub.publish(
            {
                "type": "twilio_status",
                "status": form.get("CallStatus") or form.get("StatusCallbackEvent"),
                "call_sid": form.get("CallSid"),
            }
        )
        return Response("", status_code=204)

    @api.websocket("/twilio/media")
    async def twilio_media(ws: WebSocket) -> None:
        await ws.accept()
        source = TwilioStreamSource()
        session = None
        try:
            while True:
                event = source.feed_text(await ws.receive_text())
                if event == "start" and session is None:
                    start = source.start
                    known = shared.twilio_calls.get(start.call_sid, {})
                    meta = {
                        "call_sid": start.call_sid,
                        "stream_sid": start.stream_sid,
                        "conference": start.parameters.get("conference") or known.get("conference"),
                        "from": start.parameters.get("from") or known.get("from", ""),
                        "receiver_call_sid": known.get("receiver_call_sid"),
                    }
                    info = SessionInfo(
                        new_session_id("twilio"),
                        "twilio",
                        f"Phone call from {meta['from'] or 'unknown'}",
                        meta=meta,
                    )
                    session = shared.start_session(info, source)
                if event == "stop":
                    break
        except WebSocketDisconnect:
            pass
        finally:
            source.close()
            with contextlib.suppress(Exception):
                await ws.close()

    return api


app = create_app() if os.environ.get("DFD_NO_APP") != "1" else None
