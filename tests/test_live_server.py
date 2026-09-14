"""The demo server end to end with a stand-in detector: pages, upload, Twilio webhooks and stream."""

from __future__ import annotations

import io
import time
import wave

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from live_call import media_handler as mh  # noqa: E402
from live_call.alerts import AlertDispatcher, TwilioSettings  # noqa: E402
from live_call.detector import DetectorService, DetectorSettings  # noqa: E402
from live_call.verdict_engine import EngineConfig  # noqa: E402


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("DFD_NO_APP", "1")
    from live_call import server

    detector = DetectorService(
        DetectorSettings("", "", "cpu"), score_fn=lambda w: 0.05, config=EngineConfig(threshold=0.5)
    )
    dispatcher = AlertDispatcher(settings=TwilioSettings("", "", "", "", ""))
    api = server.create_app(server.App(detector=detector, dispatcher=dispatcher))
    with TestClient(api) as test_client:
        test_client.shared = api.state.app
        yield test_client


def _wav_bytes(seconds: float = 10.0) -> bytes:
    t = np.arange(int(16000 * seconds)) / 16000
    pcm = (0.2 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm.tobytes())
    return buffer.getvalue()


def test_pages_and_health(client):
    for path in ("/", "/dashboard", "/call"):
        response = client.get(path)
        assert response.status_code == 200 and "<html" in response.text
    health = client.get("/api/health").json()
    assert health["detector"]["threshold"] == 0.5
    assert health["twilio"]["configured"] is False


def test_upload_returns_verdict_and_window_timeline(client):
    response = client.post("/api/predict", files={"file": ("clip.wav", _wav_bytes(), "audio/wav")})
    body = response.json()
    assert response.status_code == 200
    assert body["verdict"] == "spoof" and body["windows"] == 4
    assert len(body["window_scores"]) == 4


def test_bad_upload_is_a_client_error(client):
    response = client.post("/api/predict", files={"file": ("x.wav", b"not audio", "audio/wav")})
    assert response.status_code == 400


def test_voice_webhook_streams_the_caller_and_opens_a_conference(client):
    response = client.post("/twilio/voice", data={"CallSid": "CA123", "From": "+919000000000"})
    xml = response.text
    assert response.headers["content-type"].startswith("application/xml")
    assert "<Stream" in xml and 'track="inbound_track"' in xml
    assert "<Conference" in xml and "call-CA123" in xml


def test_join_and_announce_twiml(client):
    assert "call-CA9" in client.post("/twilio/join?conference=call-CA9").text
    assert "<Say" in client.post("/twilio/announce?kind=sms").text
    beep = client.get("/twilio/beep.wav")
    assert beep.status_code == 200 and beep.content[:4] == b"RIFF"


def test_signature_is_enforced_when_twilio_is_live(monkeypatch):
    monkeypatch.setenv("DFD_NO_APP", "1")
    from live_call import server

    settings = TwilioSettings(
        "AC1", "secret", "+15550001111", "+919999999999", "https://demo.example"
    )
    dispatcher = AlertDispatcher(settings=settings, rest=object())
    detector = DetectorService(
        DetectorSettings("", "", "cpu"), score_fn=lambda w: 0.9, config=EngineConfig(threshold=0.5)
    )
    api = server.create_app(server.App(detector=detector, dispatcher=dispatcher))
    with TestClient(api) as c:
        form = {"CallSid": "CA1", "From": "+91"}
        assert c.post("/twilio/join?conference=x", data=form).status_code == 403
        good = server.twilio_signature(
            "secret", "https://demo.example/twilio/join?conference=x", form
        )
        ok = c.post("/twilio/join?conference=x", data=form, headers={"X-Twilio-Signature": good})
        assert ok.status_code == 200


def test_media_stream_becomes_a_monitored_session(client):
    t = np.arange(8000 * 20) / 8000
    payload = mh.float_to_mulaw_bytes(0.3 * np.sin(2 * np.pi * 300 * t))
    with client.websocket_connect("/twilio/media") as ws:
        ws.send_json({"event": "connected"})
        ws.send_json(
            {
                "event": "start",
                "streamSid": "MZ1",
                "start": {
                    "callSid": "CA7",
                    "streamSid": "MZ1",
                    "tracks": ["inbound"],
                    "customParameters": {"conference": "call-CA7", "from": "+91888"},
                },
            }
        )
        for n, i in enumerate(range(0, len(payload), 160)):
            ws.send_json(mh.media_message(payload[i : i + 160], "MZ1", n))
        ws.send_json({"event": "stop"})
    shared = client.shared
    deadline = time.time() + 10
    while time.time() < deadline and not any(s.ended for s in shared.sessions.all()):
        time.sleep(0.05)
    session = shared.sessions.all()[0]
    assert session.info.kind == "twilio" and session.info.meta["conference"] == "call-CA7"
    assert session.engine.summary.scored_windows >= 8
    kinds = [a.kind for a in shared.dispatcher.sent if a.channel == "announce"]
    assert "beep" in kinds  # routed to the receiver's leg (dry run without credentials)


def test_results_endpoint_reads_committed_json(client):
    body = client.get("/api/results").json()
    assert set(body) == {"systems", "ablation", "reverse_degradation", "calibration"}
