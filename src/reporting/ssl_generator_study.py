"""The generator study on SSL-AASIST (P-044), judged by the rule written before it was run.

The adapter study (P-041) answered "partly": with the final recipe's adapter, training also on
Chatterbox lowered the EER on IndicF5, never heard, in all five seeds (confirmed); training also
on IndicF5 lowered Chatterbox's in all five but raised false alarms past the limit. P-044 asks
the same question of the stronger detector, SSL-AASIST on the spontaneous recipe (the voice
check's), with the same systems (``src/data/ssl_generator_study``) and two seeds, 1234 and 2025.
Fold C, both tools, is voice check v2, already trained; it is reported, not judged.

Every number is computed from per-clip scores **on speakers outside ``gen_train``**
(``generator_study.held_out_eer``), for the control and the folds alike. Two seeds cannot carry
an exact rank test (the smallest two-against-two p is 0.33), so the clip-sampling uncertainty is
taken from the held-out speakers instead: in each seed, 10,000 resamples of those speakers, with
replacement, the EER difference fold minus control recomputed on the same resampled clips.

A fold's hypothesis is **confirmed** only if all four hold:

1. lower EER on its never-heard tool than the control, in both seeds;
2. in both seeds, the 95% speaker-bootstrap interval of that difference lies wholly below zero;
3. new-speaker Tortoise (G.711; in no training set) mean EER at most 1.0 point higher;
4. genuine language-control clips called fake at most 0.5 points more often (mean).

Conditions 3 and 4 are the adapter study's, unchanged. The answer: **carries over** if both
folds are confirmed, **partly** if one is, **does not carry over** if neither, **incomplete**
if any of the six runs is missing. Two seeds see training variance only twice; the page says so::

    python -m src.reporting.ssl_generator_study --results <folder> [<folder> ...]
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd

from src.data import generator_study as gs
from src.reporting import generator_study as adapter_study

SEEDS = (1234, 2025)
#: system -> the run name its notebook writes (fold C is voice check v2's runs)
RUN_NAMES = {
    "control": "sslgs_control",
    "fold_a": "sslgs_fold_a",
    "fold_b": "sslgs_fold_b",
    "fold_c": "vcv2",
}
UNSEEN = adapter_study.UNSEEN  # fold_a -> chatterbox, fold_b -> indicf5
MEASURES = adapter_study.MEASURES
MAX_TORTOISE_RISE = adapter_study.MAX_TORTOISE_RISE
MAX_FALSE_ALARM_RISE = adapter_study.MAX_FALSE_ALARM_RISE
BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20261108
INTERVAL = (2.5, 97.5)
#: the adapter study's decisions (``experiments/results/generator_study_v1.json``), for agreement
ADAPTER_DECISIONS = {"fold_a": "not confirmed", "fold_b": "confirmed"}


def _clips(scores: pd.DataFrame, tool: str, train: set[str]) -> pd.DataFrame:
    keep = gs.test_speakers(scores["speaker"], train)
    label = scores["label"].str.lower()
    keep &= label.eq("bonafide") | scores["tool"].str.lower().eq(tool)
    return scores.loc[keep, ["filepath", "speaker", "label", "score"]]


def bootstrap_difference(
    fold: pd.DataFrame,
    control: pd.DataFrame,
    tool: str,
    train: set[str],
    n: int = BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Speaker-level bootstrap of EER(fold) - EER(control) on the same held-out clips."""
    from src.training.metrics import compute_eer

    a, b = _clips(fold, tool, train), _clips(control, tool, train)
    pair = a.merge(b[["filepath", "score"]], on="filepath", suffixes=("", "_control"))
    if len(pair) != len(a) or len(pair) != len(b):
        raise ValueError(f"{tool}: the fold and the control did not score the same clips")
    pair = pair.reset_index(drop=True)
    y = pair["label"].str.lower().eq("bonafide").to_numpy(int)
    sa = pair["score"].to_numpy(np.float64)
    sb = pair["score_control"].to_numpy(np.float64)
    groups = list(pair.groupby(pair["speaker"].astype(str), sort=True).indices.values())
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        yy = y[idx]
        if yy.all() or not yy.any():
            continue
        diffs.append(compute_eer(sa[idx], yy)[0] - compute_eer(sb[idx], yy)[0])
    low, high = (float(v) for v in np.percentile(diffs, INTERVAL))
    point = compute_eer(sa, y)[0] - compute_eer(sb, y)[0]
    return {
        "difference": float(point),
        "interval": [low, high],
        "resamples": len(diffs),
        "speakers": len(groups),
        "clips": len(pair),
    }


def judge_fold(table: dict, boot: dict, fold: str) -> dict:
    """The registered decision for one fold against the control, both seeds."""
    tool = UNSEEN[fold]
    fold_v = [table[fold][s][tool] for s in SEEDS]
    ctrl = [table["control"][s][tool] for s in SEEDS]
    rises = {
        key: mean(table[fold][s][key] for s in SEEDS)
        - mean(table["control"][s][key] for s in SEEDS)
        for key in ("new_tortoise", "control_fa")
    }
    conditions = {
        "1_lower_in_both_seeds": all(f < c for f, c in zip(fold_v, ctrl, strict=True)),
        "2_bootstrap_below_zero": all(boot[fold][s]["interval"][1] < 0 for s in SEEDS),
        "3_tortoise_within_1_point": rises["new_tortoise"] <= MAX_TORTOISE_RISE,
        "4_false_alarms_within_half_point": rises["control_fa"] <= MAX_FALSE_ALARM_RISE,
    }
    decision = "confirmed" if all(conditions.values()) else "not confirmed"
    return {
        "unseen_tool": tool,
        "fold": fold_v,
        "control": ctrl,
        "bootstrap": {str(s): boot[fold][s] for s in SEEDS},
        "tortoise_mean_rise": rises["new_tortoise"],
        "false_alarm_mean_rise": rises["control_fa"],
        "conditions": conditions,
        "decision": decision,
        "same_as_adapter_study": decision == ADAPTER_DECISIONS[fold],
    }


def complete(table: dict) -> bool:
    """Every judged quantity of the six registered runs is present."""
    needed = (*UNSEEN.values(), "new_tortoise", "control_fa")
    return all(
        table.get(system, {}).get(s, {}).get(k) is not None
        for system in ("control", *UNSEEN)
        for s in SEEDS
        for k in needed
    )


def judge(table: dict, boot: dict) -> dict:
    """The study's answer from ``table[system][seed][measure]`` and the bootstrap results."""
    out: dict = {"seeds": list(SEEDS)}
    if not complete(table) or any(s not in boot.get(f, {}) for f in UNSEEN for s in SEEDS):
        out["answer"] = "incomplete"
        return out
    out["folds"] = {fold: judge_fold(table, boot, fold) for fold in UNSEEN}
    confirmed = sum(f["decision"] == "confirmed" for f in out["folds"].values())
    out["answer"] = {2: "carries over", 1: "partly", 0: "does not carry over"}[confirmed]
    return out


def collect(*folders: Path, train: set[str]) -> tuple[dict, dict]:
    """``table[system][seed]`` and the score frames of the unseen-tool sets, from run folders.

    Each folder holds a notebook's ``runs.json`` and per-clip score files; a (system, seed)
    found twice stops the reading rather than letting one silently replace the other.
    """
    by_name = {v: k for k, v in RUN_NAMES.items()}
    table: dict = {s: {} for s in RUN_NAMES}
    frames: dict = {s: {} for s in RUN_NAMES}
    for folder in folders:
        for r in json.loads((folder / "runs.json").read_text(encoding="utf-8")):
            system, seed = by_name.get(r.get("data")), int(r["seed"])
            if system is None or r.get("status") not in adapter_study.SCORED:
                continue
            if seed in table[system]:
                raise ValueError(f"{system} seed {seed} appears in two results folders")
            run = f"{RUN_NAMES[system]}_s{seed}"
            row = adapter_study.measure_run(folder, run, train)
            row["control_fa"] = r.get("control_fa")
            table[system][seed] = row
            frames[system][seed] = {
                stem: pd.read_csv(folder / f"{run}__{stem}_scores.csv", dtype={"speaker": str})
                for stem, _ in (MEASURES[t] for t in UNSEEN.values())
                if (folder / f"{run}__{stem}_scores.csv").is_file()
            }
    return table, frames


def bootstraps(frames: dict, train: set[str]) -> dict:
    """``boot[fold][seed]`` for every fold and seed whose score files are present."""
    out: dict = {}
    for fold, tool in UNSEEN.items():
        stem = MEASURES[tool][0]
        out[fold] = {}
        for s in SEEDS:
            f, c = frames[fold].get(s, {}).get(stem), frames["control"].get(s, {}).get(stem)
            if f is not None and c is not None:
                out[fold][s] = bootstrap_difference(f, c, tool, train)
    return out


def main() -> None:
    """CLI: judge the run folders; write one JSON."""
    import argparse

    parser = argparse.ArgumentParser(description="The generator study on SSL-AASIST (P-044)")
    parser.add_argument("--results", nargs="+", required=True, help="run folders")
    parser.add_argument("--pools", default=f"data/manifests/{gs.POOLS_OUT}")
    parser.add_argument("--out", default="experiments/results/ssl_generator_study_v1.json")
    args = parser.parse_args()
    train = gs.train_speakers(args.pools)
    table, frames = collect(*map(Path, args.results), train=train)
    report = judge(table, bootstraps(frames, train))
    report["means"] = adapter_study.means(table)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for fold, f in report.get("folds", {}).items():
        print(
            f"{fold} on never-heard {f['unseen_tool']}: fold {f['fold']} control {f['control']}, "
            f"{f['decision']}"
        )
    print("ANSWER:", report["answer"])


if __name__ == "__main__":
    main()
