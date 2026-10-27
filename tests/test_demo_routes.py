"""The demo scenes' routes: the phone's address and QR code, the scene pages, the tunnel script."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _tunnel_script():
    spec = importlib.util.spec_from_file_location(
        "start_tunnel", ROOT / "scripts" / "start_tunnel.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_tunnel_script_reads_cloudflares_address():
    tunnel = _tunnel_script()
    log = "2026-10-27 INF |  https://quiet-river-1234.trycloudflare.com                    |"
    assert tunnel.find_url(log) == "https://quiet-river-1234.trycloudflare.com"
    assert tunnel.find_url("INF Starting tunnel") is None


def test_the_tunnel_address_file_sits_beside_the_demo_calls(monkeypatch, tmp_path):
    tunnel = _tunnel_script()
    monkeypatch.delenv("TUNNEL_URL_FILE", raising=False)
    monkeypatch.setenv("DEMO_CLIPS_DIR", str(tmp_path / "demo_calls_v2"))
    assert tunnel.url_file() == tmp_path / "tunnel_url.txt"
    monkeypatch.setenv("TUNNEL_URL_FILE", str(tmp_path / "x.txt"))
    assert tunnel.url_file() == tmp_path / "x.txt"


# --------------------------------------------------------------------------- endpoints
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from live_call.alerts import AlertDispatcher, TwilioSettings  # noqa: E402
from live_call.detector import DetectorService, DetectorSettings  # noqa: E402
from live_call.verdict_engine import EngineConfig  # noqa: E402


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("DFD_NO_APP", "1")
    monkeypatch.setenv("DEMO_CLIPS_DIR", str(tmp_path / "demo_calls_v2"))
    monkeypatch.delenv("TUNNEL_URL_FILE", raising=False)
    from live_call import server

    detector = DetectorService(
        DetectorSettings("", "", "cpu"),
        score_fn=lambda w: 0.95,
        config=EngineConfig(threshold=0.687),
    )
    app = server.App(
        detector=detector, dispatcher=AlertDispatcher(settings=TwilioSettings("", "", "", "", ""))
    )
    with TestClient(server.create_app(app)) as test_client:
        test_client.data = tmp_path
        yield test_client


def test_scene_pages_are_served(client):
    pages = (
        "/demo",
        "/threat",
        "/lanes",
        "/live",
        "/network",
        "/trust",
        "/limits",
        "/console",
        "/phone/receiver",
        "/phone/caller",
    )
    for path in pages:
        response = client.get(path)
        assert response.status_code == 200 and "<html" in response.text, path


def test_the_receiver_page_says_who_sees_the_verdict(client):
    page = client.get("/phone/receiver").text
    assert "Verdicts reach only the receiver" in page
    assert "speechSynthesis" in page  # the phone speaks the warning
    caller = client.get("/phone/caller").text
    assert "likely_fake" not in caller and "onVerdict" not in caller  # the caller is never told


def test_the_phone_address_prefers_the_tunnel(client):
    first = client.get("/api/tunnel").json()
    assert first["url"] is None and isinstance(first["lan"], list)
    (client.data / "tunnel_url.txt").write_text(
        "https://quiet-river-1234.trycloudflare.com\n", encoding="utf-8"
    )
    assert client.get("/api/tunnel").json()["url"] == "https://quiet-river-1234.trycloudflare.com"


def test_results_routes_answer_from_committed_files(client):
    real = client.get("/api/real_calls").json()
    assert len(real["table"]) == 4 and len(real["chain"]["stages"]) == 6
    assert client.get("/api/headline").json()["s1"]["genuine_called_fake_pct"]["source"]
    # no stage player prepared in this test's data directory: the page says so
    assert client.get("/api/stage_player").json() == {"available": False, "stages": []}
    assert client.get("/api/stage_player/direct/audio").status_code == 404


def test_the_stage_player_serves_prepared_stages(client):
    import json

    folder = client.data / "stage_player"
    folder.mkdir()
    (folder / "direct.wav").write_bytes(b"RIFF0000WAVE")
    index = {
        "speaker": "596505",
        "stages": [{"key": "direct", "label": "Direct", "file": "direct.wav"}],
    }
    (folder / "stage_player.json").write_text(json.dumps(index), encoding="utf-8")
    body = client.get("/api/stage_player").json()
    assert body["available"] is True and body["stages"][0]["key"] == "direct"
    assert client.get("/api/stage_player/direct/audio").content.startswith(b"RIFF")
    assert client.get("/api/stage_player/nope/audio").status_code == 404


def _call(folder: Path, seconds: float = 12.0) -> None:
    import io
    import json
    import wave

    import numpy as np

    folder.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(16000 * seconds)) / 16000
    pcm = (0.2 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm.tobytes())
    (folder / "call.wav").write_bytes(buffer.getvalue())
    index = [{"id": "call", "file": "call.wav", "title": "A call", "label": "spoof"}]
    (folder / "demo_clips.json").write_text(json.dumps(index), encoding="utf-8")


def test_shuffle_proof_scores_both_versions(client):
    _call(client.data / "demo_calls_v2")
    body = client.post("/api/shuffle/call").json()
    audio = client.get("/api/shuffle/call/audio")
    assert body["chunk_ms"] == 100.0 and body["verdict_unchanged"] is True
    assert body["original"]["scored_windows"] == body["shuffled"]["scored_windows"] == 5
    assert audio.status_code == 200 and audio.content[:4] == b"RIFF"
    assert client.post("/api/shuffle/call", params={"chunk_ms": 1}).status_code == 400
    assert client.post("/api/shuffle/nope").status_code == 404


def test_the_guided_flow_links_every_scene_in_order(client):
    stage = client.get("/static/stage.js").text
    hrefs = ["/threat", "/lanes", "/live", "/network", "/trust", "/limits"]
    positions = [stage.index(f'href: "{h}"') for h in hrefs]
    assert positions == sorted(positions), "the six scenes must be in plan order"
    assert client.get("/demo").status_code == 200 and "Start the demo" in client.get("/demo").text


def test_try_it_judges_a_recording_like_a_call(client):
    import io
    import wave

    import numpy as np

    t = np.arange(16000 * 12) / 16000
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((0.2 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2").tobytes())
    body = client.post(
        "/api/try", files={"file": ("me.wav", buffer.getvalue(), "audio/wav")}
    ).json()
    # the stand-in detector says 0.95 against 0.687: five windows, no warning
    assert body["peak_state"] == "genuine" and body["scored_windows"] == 5
    assert len(body["windows"]) == 5 and body["first_caution_s"] is None
    short = io.BytesIO()
    with wave.open(short, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)
    too_short = client.post("/api/try", files={"file": ("s.wav", short.getvalue(), "audio/wav")})
    assert too_short.status_code == 400 and "4 seconds" in too_short.json()["detail"]


def test_qr_codes_are_standalone_svg(client):
    pytest.importorskip("segno")
    response = client.get(
        "/api/qr", params={"text": "https://quiet-river-1234.trycloudflare.com/phone/receiver"}
    )
    assert response.status_code == 200 and response.headers["content-type"].startswith(
        "image/svg+xml"
    )
    assert b'xmlns="http://www.w3.org/2000/svg"' in response.content  # an <img> needs the namespace
    assert client.get("/api/qr", params={"text": "javascript:alert(1)"}).status_code == 400
