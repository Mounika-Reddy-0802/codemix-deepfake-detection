"""The demo rehearsal: every call the final-review demo plays does what the demo says it does.

The first half runs everywhere: the rehearsal file is well formed and the comparison
logic judges rows correctly. The second half replays every demo call through the exact
live path with the final recipe (``POST /api/replay/{id}`` against a test client) and
asserts each observed peak verdict equals the expected one. It needs the checkpoint,
the demo calls and the audio libraries, so it is skipped in CI and wherever they are
missing; set ``DFD_CHECKPOINT`` and ``DEMO_CLIPS_DIR`` to point at them.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from live_call import rehearsal

PLAN = rehearsal.load()


def test_the_rehearsal_file_covers_every_call_and_lane():
    ids = [c["id"] for c in PLAN["calls"]]
    assert len(ids) == len(set(ids)) == 8
    for call in PLAN["calls"]:
        assert call["truth"] in ("bonafide", "spoof")
        assert set(call["lanes"]) == {"s1", "deployed", "final"}
        # the final-recipe lane and the live model are the same model and threshold
        assert call["lanes"]["final"]["peak"] == call["primary"]["peak"]
    assert PLAN["primary"]["threshold_file"] == "configs/threshold_final.yaml"


def test_scene_two_calls_show_what_the_presenter_says():
    calls = {c["id"]: c for c in PLAN["calls"]}
    genuine, tortoise = calls["genuine_new_speaker"], calls["tortoise_caught"]
    assert genuine["lanes"]["s1"] == {"peak": "likely_fake", "max_first_likely_s": 10}
    assert genuine["lanes"]["final"]["peak"] == "genuine"
    assert tortoise["lanes"]["final"]["peak"] == "likely_fake"
    assert tortoise["lanes"]["deployed"]["peak"] == "suspicious"  # a caution, never likely cloned


def test_check_passes_a_match_and_names_every_mismatch():
    expect = {"peak": "likely_fake", "max_first_likely_s": 12}
    good = rehearsal.check(
        "x", "primary", expect, {"peak_state": "likely_fake", "first_likely_s": 12.0}
    )
    assert good.ok and good.cells()[6] == "PASS"
    late = rehearsal.check(
        "x", "primary", expect, {"peak_state": "likely_fake", "first_likely_s": 14.0}
    )
    assert not late.ok and "expected by 12" in late.why
    wrong = rehearsal.check("x", "s1", {"peak": "genuine"}, {"peak_state": "suspicious"})
    assert not wrong.ok and "peak suspicious" in wrong.why
    assert not rehearsal.check("x", "final", {"peak": "genuine"}, None).ok


def test_rows_report_a_missing_lane_as_a_failure():
    plan = {
        "calls": [
            {"id": "a", "primary": {"peak": "genuine"}, "lanes": {"s1": {"peak": "likely_fake"}}}
        ]
    }
    out = rehearsal.rows(plan, {"a": {"peak_state": "genuine"}}, {"a": []})
    assert [r.ok for r in out] == [True, False]
    assert "not scored" in rehearsal.table(out)


def test_the_demo_folder_check_refuses_team_audio_and_unplanned_calls():
    plan = {"calls": [{"id": "genuine"}, {"id": "xtts_clone"}]}
    clips = [{"id": "genuine", "speaker": "596505"}, {"id": "mine", "speaker": "P2"}]
    problems = rehearsal.check_clips(clips, plan)
    assert any("xtts_clone" in p and "not in the demo folder" in p for p in problems)
    assert any("'mine'" in p and "no expected verdict" in p for p in problems)
    assert any("team member" in p for p in problems)


# --------------------------------------------------------------------------- real models
CHECKPOINT = os.environ.get("DFD_CHECKPOINT", r"C:\dfdata\checkpoints\final_s1234_best.pt")
CLIPS = os.environ.get("DEMO_CLIPS_DIR", r"C:\dfdata\demo_calls_v2")


@pytest.fixture(scope="module")
def live_client():
    pytest.importorskip("soundfile")  # audio wheels, not installed in CI
    pytest.importorskip("librosa")
    pytest.importorskip("torch")
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    if not Path(CHECKPOINT).is_file() or Path(CHECKPOINT).name != PLAN["primary"]["checkpoint"]:
        pytest.skip(f"final-recipe checkpoint not on this machine ({CHECKPOINT})")
    if not (Path(CLIPS) / "demo_clips.json").is_file():
        pytest.skip(f"demo calls not on this machine ({CLIPS})")
    from fastapi.testclient import TestClient

    from live_call import server
    from live_call.alerts import AlertDispatcher, TwilioSettings
    from live_call.detector import DetectorService, DetectorSettings

    mp = pytest.MonkeyPatch()
    mp.setenv("DEMO_CLIPS_DIR", CLIPS)
    detector = DetectorService(
        DetectorSettings(CHECKPOINT, PLAN["primary"]["threshold_file"], "cpu")
    )
    app = server.App(
        detector=detector, dispatcher=AlertDispatcher(settings=TwilioSettings("", "", "", "", ""))
    )
    with TestClient(server.create_app(app)) as client:
        yield client
    mp.undo()


def test_the_demo_folder_matches_the_rehearsal_file(live_client):
    assert rehearsal.check_clips(live_client.get("/api/demo-clips").json(), PLAN) == []


@pytest.mark.parametrize("call", [c["id"] for c in PLAN["calls"]])
def test_every_demo_call_replays_to_its_expected_verdict(live_client, call):
    expect = next(c for c in PLAN["calls"] if c["id"] == call)["primary"]
    observed = live_client.post(
        f"/api/replay/{call}", params={"realtime": "false", "wait": "true"}
    ).json()
    row = rehearsal.check(call, "primary", expect, observed)
    assert row.ok, row.why
