"""Voice check v2 on clean files: speakers left out, both runs on the same recordings."""

from __future__ import annotations

import pandas as pd
import pytest

from src.reporting import voice_check_v2_clean as vc


def _run(path, rows) -> str:
    cols = ["filepath", "label", "tool", "set", "split", "front_end", "verdict", "p_genuine"]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
    return str(path)


ROWS = [
    ("${DATA_ROOT}/c/a_1.wav", "bonafide", "none", "cm07", "test", "g711", "real", 0.9),
    ("${DATA_ROOT}/c/a_2.wav", "spoof", "indicf5", "cm07", "test", "g711", "cloned", 0.01),
    ("${DATA_ROOT}/c/t_1.wav", "spoof", "indicf5", "cm07", "test", "g711", "cloned", 0.01),
    ("${DATA_ROOT}/c/b_1.wav", "bonafide", "none", "hiacc", "test", "g711", "no speech", None),
    ("${DATA_ROOT}/c/b_2.wav", "bonafide", "none", "hiacc", "test", "none", "real", 0.9),
]
SPEAKERS = {"a_1.wav": "a", "a_2.wav": "a", "t_1.wav": "t", "b_1.wav": "b", "b_2.wav": "b"}


def test_training_speakers_and_other_front_ends_are_left_out(tmp_path):
    d = vc.load(_run(tmp_path / "r.csv", ROWS), SPEAKERS, {"t"})
    assert set(d["name"]) == {"a_1.wav", "a_2.wav", "b_1.wav"}
    assert set(d["kind"]) == {"genuine", "indicf5"}


def test_a_file_with_no_known_speaker_stops_the_table(tmp_path):
    with pytest.raises(ValueError):
        vc.load(_run(tmp_path / "r.csv", ROWS), {"a_1.wav": "a"}, set())


def test_runs_are_compared_on_the_recordings_both_scored(tmp_path):
    a = vc.load(_run(tmp_path / "a.csv", ROWS), SPEAKERS, set())
    b = vc.load(_run(tmp_path / "b.csv", ROWS[:2]), SPEAKERS, set())
    out = vc.paired({"before": a, "v2": b})
    assert len(out["before"]) == len(out["v2"]) == 2


def test_cells_count_too_short_and_genuine_called_cloned():
    d = pd.DataFrame(
        {
            "label": ["bonafide"] * 4,
            "kind": ["genuine"] * 4,
            "verdict": ["real", "real", "no speech", "cloned"],
        }
    )
    assert vc.cell(d) == "50.0 % of 4 (25 % too short; 1 called cloned)"
    assert vc.cell(d.iloc[:0]) == "-"


def test_a_name_under_two_speakers_is_refused(tmp_path):
    pd.DataFrame({"filepath": ["x/a_1.wav"], "speaker": ["a"]}).to_csv(
        tmp_path / "m1.csv", index=False
    )
    pd.DataFrame({"filepath": ["y/a_1.wav"], "speaker": ["z"]}).to_csv(
        tmp_path / "m2.csv", index=False
    )
    with pytest.raises(ValueError):
        vc.speaker_map(str(tmp_path / "*.csv"))
