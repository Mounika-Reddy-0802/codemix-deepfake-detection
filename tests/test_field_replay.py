"""Field recordings and corpus replay calls: matched to their plans by name, labels from the plan."""

from __future__ import annotations

import pandas as pd
import pytest

from src.reporting import field_replay as fr


def _touch(folder, *names):
    folder.mkdir(parents=True, exist_ok=True)
    for n in names:
        (folder / n).write_bytes(b"x")


def _plans(tmp_path):
    field = pd.DataFrame(
        {
            "take_id": ["P1_F01", "P1_K01", "P1_F02"],
            "batch": [1, 1, 1],
            "kind": ["live_read", "clone_replay", "live_read"],
            "tool": ["none", "xtts_v2", "none"],
            "label": ["bonafide", "spoof", "bonafide"],
        }
    )
    replay = pd.DataFrame(
        {
            "file": ["R01.wav", "R02.wav"],
            "kind": ["genuine", "indicf5"],
            "tool": ["none", "indicf5"],
            "label": ["bonafide", "spoof"],
        }
    )
    field.to_csv(tmp_path / "take_plan.csv", index=False)
    replay.to_csv(tmp_path / "replay_plan.csv", index=False)
    return str(tmp_path / "take_plan.csv"), str(tmp_path / "replay_plan.csv")


def test_recordings_match_their_plans_and_missing_ones_are_listed(tmp_path):
    fp, rp = _plans(tmp_path)
    _touch(tmp_path / "field", "p1_f01.m4a", "P1_K01.wav", "notes.txt")
    _touch(tmp_path / "replay", "R02.m4a")
    rows, missing = fr.manifest(tmp_path / "field", tmp_path / "replay", fp, rp)
    assert sorted(rows["take"]) == ["P1_F01", "P1_K01", "R02"]
    assert dict(zip(rows["take"], rows["label"], strict=True))["R02"] == "spoof"
    assert set(rows["set"]) == {"field_b1", "corpus_replay"}
    assert missing == ["P1_F02", "R01"]


def test_a_recording_outside_the_plan_stops(tmp_path):
    fp, rp = _plans(tmp_path)
    _touch(tmp_path / "replay", "R03.m4a")
    with pytest.raises(ValueError):
        fr.manifest(None, tmp_path / "replay", fp, rp)


def test_the_report_counts_too_short_as_wrong():
    scored = pd.DataFrame(
        {
            "set": ["corpus_replay"] * 4,
            "kind": ["genuine", "genuine", "indicf5", "indicf5"],
            "label": ["bonafide", "bonafide", "spoof", "spoof"],
            "verdict": ["real", "no speech", "cloned", "real"],
            "front_end": ["g711"] * 4,
        }
    )
    text = fr.report(scored, ["R05"])
    assert "| corpus_replay: genuine | 1 of 2 | 50.0 |" in text
    assert "1 too short" in text
    assert "| **corpus_replay: all cloned** | 1 of 2 |" in text
    assert "Planned but not recorded (1): R05." in text
