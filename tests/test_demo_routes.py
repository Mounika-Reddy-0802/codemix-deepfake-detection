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
    for path in ("/live", "/lanes", "/network", "/console", "/phone/receiver", "/phone/caller"):
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
