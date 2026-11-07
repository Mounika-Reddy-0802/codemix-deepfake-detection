"""The generator study's decision rule on hand-made numbers (no real score file is read)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.reporting import generator_study as rep

CONTROL = {"chatterbox": 0.30, "indicf5": 0.22, "new_tortoise": 0.09, "control_fa": 0.005}


def _table(fold_a: float, fold_b: float, tortoise_shift: float = 0.0, seeds=rep.SEEDS) -> dict:
    table = {s: {} for s in rep.SYSTEMS}
    for i, seed in enumerate(seeds):
        jitter = 0.002 * i
        table["control"][seed] = {k: v + jitter for k, v in CONTROL.items()}
        table["fold_a"][seed] = {**table["control"][seed], "chatterbox": fold_a + jitter}
        table["fold_b"][seed] = {**table["control"][seed], "indicf5": fold_b + jitter}
        for fold in ("fold_a", "fold_b"):
            table[fold][seed]["new_tortoise"] += tortoise_shift
    return table


def test_both_folds_clearly_better_carries_over():
    out = rep.judge(_table(fold_a=0.10, fold_b=0.08))
    assert out["answer"] == "carries over"
    assert out["folds"]["fold_a"]["mann_whitney_p"] < rep.MAX_P


def test_one_fold_better_is_partly():
    out = rep.judge(_table(fold_a=0.10, fold_b=0.25))
    assert out["folds"]["fold_b"]["conditions"]["1_lower_in_4_of_5"] is False
    assert out["answer"] == "partly"


def test_a_tortoise_cost_above_one_point_blocks_both():
    out = rep.judge(_table(fold_a=0.10, fold_b=0.08, tortoise_shift=0.011))
    assert out["answer"] == "does not carry over"


def test_fewer_than_five_seeds_is_incomplete():
    assert rep.judge(_table(0.10, 0.08, seeds=rep.SEEDS[:4]))["answer"] == "incomplete"


def test_eer_ignores_the_training_half_and_other_tools():
    scores = pd.DataFrame(
        {
            "speaker": ["a", "a", "b", "b", "b"],
            "label": ["bonafide", "spoof", "bonafide", "spoof", "spoof"],
            "tool": ["none", "chatterbox", "none", "chatterbox", "tortoise"],
            "score": [0.1, 0.9, 0.9, 0.1, 0.95],
        }
    )
    # speaker a (training half) is inverted; held out, b is perfectly separated
    assert rep.held_out_eer(scores, "chatterbox", {"a"}) == 0.0
    assert rep.held_out_eer(scores, "chatterbox", set()) > 0.0


def test_collect_reads_scores_and_false_alarms_of_finished_runs(tmp_path):
    runs = [
        {"system": "control", "seed": 7, "status": "ok", "control_fa": 0.004},
        {"system": "fold_a", "seed": 7, "status": "failed"},
    ]
    (tmp_path / "runs.json").write_text(json.dumps(runs))
    pd.DataFrame(
        {
            "speaker": ["b", "b"],
            "label": ["bonafide", "spoof"],
            "tool": ["none", "indicf5"],
            "score": [0.9, 0.1],
        }
    ).to_csv(tmp_path / "control_s7__cm07_scores.csv", index=False)
    table = rep.collect(tmp_path, train=set())
    assert table["control"][7]["indicf5"] == 0.0
    assert table["control"][7]["control_fa"] == 0.004
    assert table["control"][7]["chatterbox"] is None
    assert table["fold_a"] == {}


def _write_part(folder, runs):
    folder.mkdir()
    (folder / "runs.json").write_text(json.dumps(runs))
    for r in runs:
        pd.DataFrame(
            {
                "speaker": ["b", "b"],
                "label": ["bonafide", "spoof"],
                "tool": ["none", "indicf5"],
                "score": [0.9, 0.1],
            }
        ).to_csv(folder / f"{r['system']}_s{r['seed']}__cm07_scores.csv", index=False)


def test_both_kaggle_parts_are_read_and_a_partial_run_still_counts(tmp_path):
    _write_part(
        tmp_path / "a", [{"system": "control", "seed": 7, "status": "ok", "control_fa": 0.0}]
    )
    _write_part(
        tmp_path / "b", [{"system": "control", "seed": 42, "status": "partial", "control_fa": 0.0}]
    )
    table = rep.collect(tmp_path / "a", tmp_path / "b", train=set())
    assert set(table["control"]) == {7, 42}


def test_the_same_run_in_two_parts_is_refused(tmp_path):
    runs = [{"system": "control", "seed": 7, "status": "ok", "control_fa": 0.0}]
    _write_part(tmp_path / "a", runs)
    _write_part(tmp_path / "b", runs)
    with pytest.raises(ValueError):
        rep.collect(tmp_path / "a", tmp_path / "b", train=set())
