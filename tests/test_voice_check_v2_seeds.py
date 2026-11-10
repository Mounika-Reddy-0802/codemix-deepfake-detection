"""Voice check v2 over four seeds (P-045): collected from run folders, reported without a rule."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.reporting import voice_check_v2_seeds as seeds

TRAIN = {"t1"}


def _folder(path, seed, scores, status="ok", data="vcv2"):
    """One notebook's results folder: runs.json and the CM09 score file."""
    path.mkdir(parents=True)
    run = {"data": data, "seed": seed, "status": status}
    run.update(control_fa=0.001, hiacc_fa=0.003, iv_test_fa=0.002)
    (path / "runs.json").write_text(json.dumps([run]), encoding="utf-8")
    rows = [
        ("t1", "bonafide", "none", 9.0),  # a training speaker: left out of every EER
        ("t1", "spoof", "chatterbox", 9.5),
        ("h1", "bonafide", "none", scores[0]),
        ("h2", "bonafide", "none", scores[1]),
        ("h1", "spoof", "chatterbox", scores[2]),
        ("h2", "spoof", "chatterbox", scores[3]),
    ]
    pd.DataFrame(rows, columns=["speaker", "label", "tool", "score"]).to_csv(
        path / f"vcv2_s{seed}__cm09_scores.csv", index=False
    )
    return path


def test_runs_are_collected_with_training_speakers_left_out(tmp_path):
    a = _folder(tmp_path / "a", 1234, [5.0, 6.0, -5.0, -6.0])
    b = _folder(tmp_path / "b", 7, [5.0, -7.0, -5.0, -6.0])
    table = seeds.collect(a, b, train=TRAIN)
    assert sorted(table) == [7, 1234]
    assert table[1234]["chatterbox"] == 0.0
    assert table[7]["chatterbox"] > 0.0
    assert table[1234]["indicf5"] is None
    assert table[7]["hiacc_fa"] == 0.003


def test_a_seed_found_twice_stops_and_other_runs_are_ignored(tmp_path):
    a = _folder(tmp_path / "a", 1234, [5.0, 6.0, -5.0, -6.0])
    b = _folder(tmp_path / "b", 1234, [5.0, 6.0, -5.0, -6.0])
    with pytest.raises(ValueError):
        seeds.collect(a, b, train=TRAIN)
    c = _folder(tmp_path / "c", 2025, [5.0, 6.0, -5.0, -6.0], data="sslgs_control")
    d = _folder(tmp_path / "d", 42, [5.0, 6.0, -5.0, -6.0], status="failed")
    assert seeds.collect(c, d, train=TRAIN) == {}


def test_summary_and_table_report_every_seed_and_name_the_missing():
    table = {
        1234: {"chatterbox": 0.01, "control_fa": 0.0},
        2025: {"chatterbox": 0.03, "control_fa": 0.002},
        7: {"chatterbox": 0.02, "control_fa": None},
    }
    summ = seeds.summary(table)
    assert summ["chatterbox"]["mean"] == pytest.approx(0.02)
    assert summ["chatterbox"]["sd"] == pytest.approx(0.01)
    assert (summ["chatterbox"]["min"], summ["chatterbox"]["max"]) == (0.01, 0.03)
    assert summ["control_fa"]["n"] == 2
    assert "indicf5" not in summ
    text = seeds.markdown(table, summ, excluded=31)
    assert "Missing seeds: 42" in text
    assert (
        "| Chatterbox clones (CM09), G.711 | 1.00 | 3.00 | 2.00 | 2.00 ± 1.00 | 1.00-3.00 |" in text
    )
