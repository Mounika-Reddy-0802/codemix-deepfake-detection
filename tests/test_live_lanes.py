"""The three-lane view: lanes read from the environment, one score series per lane, clear errors."""

from __future__ import annotations

import asyncio
import io
import json
import wave

import numpy as np
import pytest

from live_call.detector import DetectorService, DetectorSettings
from live_call.lanes import (
    Lane,
    LaneSet,
    LaneSpecError,
    build_lanes,
    lane_role,
    parse_lanes,
    score_lanes,
)
from live_call.verdict_engine import EngineConfig

SPEC = (
    r"S1 (English only)=checkpoints\baseline\best.pt=configs\threshold.yaml,"
    r"Deployed adapter=C:\dfdata\checkpoints\lora_norm_rvc_channel_best.pt=configs\threshold.yaml,"
    r"Final recipe=C:\dfdata\checkpoints\final_s1234_best.pt=configs\threshold_final.yaml"
)


def test_lanes_parse_from_the_plan_value():
    lanes = parse_lanes(SPEC)
    assert [lane.name for lane in lanes] == [
        "S1 (English only)",
        "Deployed adapter",
        "Final recipe",
    ]
    # a Windows drive letter carries a colon, never an equals sign: the split is safe
    assert lanes[1].checkpoint == r"C:\dfdata\checkpoints\lora_norm_rvc_channel_best.pt"
    assert lanes[2].threshold_file == r"configs\threshold_final.yaml"


def test_empty_value_means_no_extra_lanes():
    assert parse_lanes("") == []
    assert parse_lanes(" , ") == []


@pytest.mark.parametrize(
    "bad",
    [
        "S1=checkpoints/best.pt",  # no threshold file
        "S1=a.pt=b.yaml=extra",  # one field too many
        "=a.pt=b.yaml",  # no name
        "S1=a.pt=b.yaml,S1=c.pt=d.yaml",  # the same lane twice
    ],
)
def test_a_bad_spec_names_the_problem(bad):
    with pytest.raises(LaneSpecError, match="DFD_LANES"):
        parse_lanes(bad)


def test_roles_give_the_paper_colours():
    assert lane_role("S1 (English only)") == "s1"
    assert lane_role("Deployed adapter") == "deployed"
    assert lane_role("Final recipe") == "final"
    assert lane_role("SSL-AASIST") == "ssl"
    assert lane_role("Something else", "x.pt") == "other"
    lanes = build_lanes(parse_lanes(SPEC), DetectorService(DetectorSettings("", "", "cpu")))
    assert [lane.colour for lane in lanes] == ["#8a94a6", "#eb6834", "#1baf7a"]


def test_a_lane_on_the_primary_checkpoint_shares_its_model(tmp_path):
    ckpt = tmp_path / "final.pt"
    ckpt.write_bytes(b"x")
    primary = DetectorService(DetectorSettings(str(ckpt), "", "cpu"), score_fn=lambda w: 0.9)
    lanes = build_lanes(parse_lanes(f"Final recipe={ckpt}=configs/threshold_final.yaml"), primary)
    assert lanes[0].detector.score_fn() is primary.score_fn()
    assert lanes[0].detector.settings.threshold_file == "configs/threshold_final.yaml"


def test_lane_set_reads_the_environment(monkeypatch):
    primary = DetectorService(DetectorSettings("", "", "cpu"))
    monkeypatch.setenv("DFD_LANES", SPEC)
    assert len(LaneSet.from_env(primary).lanes) == 3
    monkeypatch.setenv("DFD_LANES", "")
    only = LaneSet.from_env(primary)  # nothing configured: the live model alone
    assert len(only.lanes) == 1 and only.lanes[0].detector is primary
    monkeypatch.setenv("DFD_LANES", "broken")
    broken = LaneSet.from_env(primary)
    assert broken.lanes == [] and "DFD_LANES" in broken.problems()[0]


def _lane(name: str, score: float, threshold: float = 0.5, **ladder) -> Lane:
    config = EngineConfig(threshold=threshold, **ladder)
    detector = DetectorService(
        DetectorSettings("", "", "cpu"), score_fn=lambda w: score, config=config
    )
    role = lane_role(name)
    return Lane(name, detector, role, "#000")


def _speech(seconds: float, rate: int = 16000) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    return (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_every_lane_scores_the_same_windows_with_its_own_ladder():
    lanes = [
        _lane("S1 (English only)", 0.05),
        _lane("Final recipe", 0.95, threshold=0.687, suspicious_after=3, fake_after=5),
    ]
    sent = []
    results = asyncio.run(score_lanes(_speech(20.0), lanes, emit=sent.append))
    s1, final = (r.summary() for r in results)
    assert s1["peak_state"] == "likely_fake" and final["peak_state"] == "genuine"
    # 20 s of audio, 4 s windows every 2 s: 9 windows in every lane, at the same times
    assert [w["t"] for w in results[0].windows] == [w["t"] for w in results[1].windows]
    assert len(results[0].windows) == 9
    # the English-only lane cautions at its second low window and warns at its fourth
    assert s1["first_caution_s"] == 6.0 and s1["first_likely_s"] == 10.0
    assert final["first_caution_s"] is None
    kinds = {m["type"] for m in sent}
    assert kinds == {"lane_window", "lane_event"}
    assert {m["lane"] for m in sent} == {"S1 (English only)", "Final recipe"}


def test_an_ssl_aasist_lane_loads_as_the_ssl_kind(monkeypatch, tmp_path):
    from src.inference import predict

    ckpt = tmp_path / "ssl_aasist_final_s1234.pt"
    ckpt.write_bytes(b"x")
    kinds = []

    def load(path, kind="adapter", device="cpu"):
        kinds.append(kind)
        return lambda w: 0.7

    monkeypatch.setattr(predict, "load_score_fn", load)
    lanes = build_lanes(
        parse_lanes(f"SSL-AASIST={ckpt}=configs/threshold_final.yaml"),
        DetectorService(DetectorSettings("", "", "cpu")),
    )
    assert lanes[0].role == "ssl" and lanes[0].colour == "#8b6cff"
    assert lanes[0].detector.score_fn()(np.zeros(64000, dtype=np.float32)) == 0.7
    assert kinds == ["ssl_aasist"]


def test_the_ssl_aasist_score_is_its_softmax_on_a_full_input(monkeypatch):
    torch = pytest.importorskip("torch")
    from src.baselines import aasist
    from src.inference import predict

    seen = []

    class Stand(torch.nn.Module):
        def forward(self, wave):
            seen.append(tuple(wave.shape))
            return torch.tensor([[0.0, 2.0]])

    monkeypatch.setattr(aasist, "build", lambda weights, device, kind: Stand())
    score = predict.load_score_fn("ssl.pt", kind="ssl_aasist", device="cpu")
    p = score(np.zeros(64000, dtype=np.float32))
    assert seen == [(1, aasist.NB_SAMP)]  # a 4 s window repeated to AASIST's fixed length
    assert p == pytest.approx(float(torch.softmax(torch.tensor([0.0, 2.0]), 0)[1]))


# --------------------------------------------------------------------------- endpoints
fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from live_call.alerts import AlertDispatcher, TwilioSettings  # noqa: E402


def _wav(seconds: float) -> bytes:
    pcm = (_speech(seconds) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm.tobytes())
    return buffer.getvalue()


@pytest.fixture()
def clips(tmp_path, monkeypatch):
    (tmp_path / "call.wav").write_bytes(_wav(12.0))
    index = [
        {"id": "call", "file": "call.wav", "title": "A call", "label": "spoof", "tool": "xtts_v2"}
    ]
    (tmp_path / "demo_clips.json").write_text(json.dumps(index), encoding="utf-8")
    monkeypatch.setenv("DEMO_CLIPS_DIR", str(tmp_path))
    return tmp_path


def _client(monkeypatch, lane_set: LaneSet):
    monkeypatch.setenv("DFD_NO_APP", "1")
    from live_call import server

    detector = DetectorService(
        DetectorSettings("", "", "cpu"),
        score_fn=lambda w: 0.95,
        config=EngineConfig(threshold=0.687),
    )
    dispatcher = AlertDispatcher(settings=TwilioSettings("", "", "", "", ""))
    app = server.App(detector=detector, dispatcher=dispatcher, lanes=lane_set)
    return TestClient(server.create_app(app))


def test_endpoint_returns_one_score_series_per_lane(clips, monkeypatch):
    lane_set = LaneSet([_lane("S1 (English only)", 0.05), _lane("Final recipe", 0.95)])
    with _client(monkeypatch, lane_set) as client:
        listed = client.get("/api/lanes").json()
        assert [lane["name"] for lane in listed["lanes"]] == ["S1 (English only)", "Final recipe"]
        body = client.post("/api/replay_lanes/call", params={"realtime": "false"}).json()
    assert body["clip"] == "call" and body["seconds"] == 12.0
    assert [lane["name"] for lane in body["lanes"]] == ["S1 (English only)", "Final recipe"]
    assert all(len(lane["windows"]) == 5 for lane in body["lanes"])
    assert [lane["peak_state"] for lane in body["lanes"]] == ["likely_fake", "genuine"]


def test_realtime_lanes_stream_to_the_dashboard_socket(clips, monkeypatch):
    (clips / "call.wav").write_bytes(_wav(4.5))  # one window, so the test takes 5 s
    lane_set = LaneSet([_lane("S1 (English only)", 0.05), _lane("Final recipe", 0.95)])
    with _client(monkeypatch, lane_set) as client:
        with client.websocket_connect("/ws/dashboard") as ws:
            assert ws.receive_json()["type"] == "hello"
            run = client.post("/api/replay_lanes/call").json()["run"]
            seen = []
            while not seen or seen[-1]["type"] not in ("lanes_end", "lanes_error"):
                message = ws.receive_json()
                if message.get("run") == run:
                    seen.append(message)
    types = [m["type"] for m in seen]
    assert types[0] == "lanes_start" and types[-1] == "lanes_end"
    windows = [m for m in seen if m["type"] == "lane_window"]
    assert {m["lane"] for m in windows} == {"S1 (English only)", "Final recipe"}


def test_a_bad_spec_gives_a_clear_400(clips, monkeypatch):
    broken = LaneSet.from_env(DetectorService(DetectorSettings("", "", "cpu")), "only-a-name")
    with _client(monkeypatch, broken) as client:
        response = client.post("/api/replay_lanes/call", params={"realtime": "false"})
        assert response.status_code == 400
        assert "name=checkpoint=threshold_file" in response.json()["detail"]
        assert client.get("/api/lanes").json()["problems"]


def test_a_missing_checkpoint_is_a_400_not_a_crash(clips, monkeypatch):
    primary = DetectorService(DetectorSettings("", "", "cpu"))
    lane_set = LaneSet.from_env(
        primary, "Final recipe=C:/nowhere/final.pt=configs/threshold_final.yaml"
    )
    with _client(monkeypatch, lane_set) as client:
        response = client.post("/api/replay_lanes/call", params={"realtime": "false"})
    assert response.status_code == 400 and "checkpoint not found" in response.json()["detail"]


def test_an_unknown_call_is_a_404(clips, monkeypatch):
    with _client(monkeypatch, LaneSet([_lane("Final recipe", 0.95)])) as client:
        assert (
            client.post("/api/replay_lanes/nope", params={"realtime": "false"}).status_code == 404
        )
