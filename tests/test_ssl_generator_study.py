"""The generator study on SSL-AASIST (P-044): its manifests and its rule (hand-made numbers)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.data import generator_study as gs
from src.data import spont_recipe
from src.data import ssl_generator_study as data
from src.data import voice_check_v2 as v2
from src.reporting import ssl_generator_study as rule

TRAIN = {"t1", "t2"}


def _run(chatterbox=0.10, indicf5=0.10, tortoise=0.05, fa=0.002) -> dict:
    return {
        "chatterbox": chatterbox,
        "indicf5": indicf5,
        "new_tortoise": tortoise,
        "control_fa": fa,
    }


def _table(**folds) -> dict:
    table = {"control": {s: _run() for s in rule.SEEDS}}
    for fold in rule.UNSEEN:
        table[fold] = {s: folds.get(fold, _run(chatterbox=0.05, indicf5=0.05)) for s in rule.SEEDS}
    return table


def _boot(high=-0.01) -> dict:
    return {f: {s: {"interval": [-0.05, high]} for s in rule.SEEDS} for f in rule.UNSEEN}


def test_both_folds_confirmed_carries_over():
    out = rule.judge(_table(), _boot())
    assert out["answer"] == "carries over"
    assert all(f["decision"] == "confirmed" for f in out["folds"].values())


def test_each_condition_can_fail_on_its_own():
    worse = _run(chatterbox=0.12, indicf5=0.05)
    assert rule.judge(_table(fold_a=worse), _boot())["answer"] == "partly"
    assert rule.judge(_table(), _boot(high=0.001))["answer"] == "does not carry over"
    tortoise = _run(chatterbox=0.05, indicf5=0.05, tortoise=0.0601)
    out = rule.judge(_table(fold_b=tortoise), _boot())
    assert out["folds"]["fold_b"]["conditions"]["3_tortoise_within_1_point"] is False
    alarms = _run(chatterbox=0.05, indicf5=0.05, fa=0.0071)
    out = rule.judge(_table(fold_a=alarms), _boot())
    assert out["folds"]["fold_a"]["conditions"]["4_false_alarms_within_half_point"] is False


def test_a_missing_run_is_incomplete():
    table = _table()
    del table["fold_b"][2025]
    assert rule.judge(table, _boot())["answer"] == "incomplete"


def _scores(rng, shift: float) -> pd.DataFrame:
    rows = []
    for spk in ["t1", "h1", "h2", "h3", "h4", "h5"]:
        for i in range(20):
            fake = i % 2 == 1
            score = rng.normal(-1.0 + shift if fake else 1.0)
            rows.append(
                (
                    f"{spk}_{i}.wav",
                    spk,
                    "spoof" if fake else "bonafide",
                    "chatterbox" if fake else "none",
                    score,
                )
            )
    return pd.DataFrame(rows, columns=["filepath", "speaker", "label", "tool", "score"])


def test_the_bootstrap_leaves_out_training_speakers_and_finds_a_real_gap():
    rng = np.random.default_rng(0)
    base = _scores(rng, 0.0)
    better = base.copy()
    better.loc[better["label"] == "spoof", "score"] -= 3.0  # the fold separates far better
    worse = base.copy()
    worse.loc[worse["label"] == "spoof", "score"] += 1.5
    out = rule.bootstrap_difference(better, worse, "chatterbox", TRAIN, n=500)
    assert out["speakers"] == 5 and out["clips"] == 100  # t1 left out
    assert out["interval"][1] < 0 and out["difference"] < 0
    same = rule.bootstrap_difference(base, base, "chatterbox", TRAIN, n=200)
    assert same["interval"] == [0.0, 0.0]


def test_different_clips_are_refused():
    rng = np.random.default_rng(1)
    a = _scores(rng, 0.0)
    try:
        rule.bootstrap_difference(a, a.iloc[:-1], "chatterbox", TRAIN, n=10)
    except ValueError:
        return
    raise AssertionError("a fold and a control scored on different clips must be refused")


def test_collect_reads_runs_json_and_score_files(tmp_path):
    rng = np.random.default_rng(2)
    rows = []
    for name in rule.RUN_NAMES.values():
        for s in rule.SEEDS:
            run = f"{name}_s{s}"
            rows.append({"name": run, "data": name, "seed": s, "status": "ok", "control_fa": 0.001})
            for stem in ("cm09", "cm07", "new_speakers"):
                tool = {"cm09": "chatterbox", "cm07": "indicf5", "new_speakers": "tortoise"}[stem]
                f = _scores(rng, 0.0)
                f.loc[f["label"] == "spoof", "tool"] = tool
                f.to_csv(tmp_path / f"{run}__{stem}_scores.csv", index=False)
    (tmp_path / "runs.json").write_text(json.dumps(rows))
    table, frames = rule.collect(tmp_path, train=TRAIN)
    assert set(table) == set(rule.RUN_NAMES) and all(len(v) == 2 for v in table.values())
    assert rule.complete(table)
    assert set(frames["fold_a"][1234]) == {"cm09", "cm07"}


def test_the_manifests_are_the_spontaneous_recipe_plus_each_systems_rows():
    base = pd.read_csv(f"data/manifests/{spont_recipe.TRAIN_OUT}.csv", dtype={"speaker": str})
    train = gs.train_speakers()
    tools = {"control": set(), "fold_a": {"indicf5"}, "fold_b": {"chatterbox"}}
    for system in data.SYSTEMS:
        out = pd.read_csv(f"data/manifests/{data.train_out(system)}.csv", dtype={"speaker": str})
        assert out.equals(v2.build(system=system))
        extra = out.iloc[len(base) :]
        assert extra["speaker"].isin(train).all()
        assert set(extra.loc[extra["label"] == "spoof", "tool"].str.lower()) == tools[system]
        assert not out["tool"].str.lower().eq("tortoise").any()
    control = pd.read_csv(f"data/manifests/{data.train_out('control')}.csv", dtype={"speaker": str})
    vcv2 = pd.read_csv(f"data/manifests/{v2.TRAIN_OUT}.csv", dtype={"speaker": str})
    assert len(vcv2) - len(control) == 282 + 265
