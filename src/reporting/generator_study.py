"""The generator study (P-041), judged by the rule written before it was run.

Does training on one more cloning tool catch a tool the detector has never heard? Each
fold adds one tool's clones of the ``gen_train`` speakers to the final recipe; the control
adds only those speakers' genuine clips. Every number here is computed from per-clip
scores **on speakers outside ``gen_train``** (``src/data/generator_study``), for the
control and the folds alike, so the comparison is on voices none of them heard.

For each fold, its **unseen** tool is the primary measure: Chatterbox (CM09, G.711) for
fold A, which trained on IndicF5; IndicF5 (CM07, G.711) for fold B, which trained on
Chatterbox. A fold's hypothesis is **confirmed** only if all four hold, against the
control's five paired seeds:

1. lower EER on its unseen tool in at least 4 of 5 seeds;
2. exact two-sided Mann-Whitney p <= 0.025 on the five-against-five EERs (0.05 split over
   the two hypotheses);
3. new-speaker Tortoise (G.711; never in any training) mean EER at most 1.0 point higher;
4. genuine language-control clips called fake at most 0.5 points more often (mean).

The study's answer: **carries over** if both folds are confirmed, **partly** if one is,
**does not carry over** if neither; **incomplete** with fewer than five seeds of any of
the three systems. Fold C (both tools, the demo model) is reported, not judged here::

    python -m src.reporting.generator_study --results genstudy_results
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd

from src.data import generator_study as gs
from src.reporting.variants import mann_whitney_exact

SEEDS = (1234, 2025, 7, 42, 99)
SYSTEMS = ("control", "fold_a", "fold_b", "fold_c")
UNSEEN = {"fold_a": "chatterbox", "fold_b": "indicf5"}
MIN_LOWER = 4
MAX_P = 0.025
MAX_TORTOISE_RISE = 0.010
MAX_FALSE_ALARM_RISE = 0.005
#: measure -> (score file of the run, the spoof tool it is EER against genuine of)
MEASURES = {
    "chatterbox": ("cm09", "chatterbox"),
    "indicf5": ("cm07", "indicf5"),
    "indicf5_amr": ("cm07_amr", "indicf5"),
    "new_tortoise": ("new_speakers", "tortoise"),
    "new_tortoise_amr": ("new_speakers_amr", "tortoise"),
    "new_xtts": ("new_speakers", "xtts_v2"),
    "cm04": ("cm04", "tortoise"),
    "cm04_amr": ("cm04_amr", "tortoise"),
    "eval_pool": ("eval_pool", "xtts_v2"),
    "rvc_test": ("rvc_test", "rvc"),
}


def held_out_eer(scores: pd.DataFrame, tool: str, train: set[str]) -> float | None:
    """EER of one tool against the genuine clips, on speakers outside ``train`` only."""
    from src.training.metrics import compute_eer

    keep = gs.test_speakers(scores["speaker"], train)
    label = scores["label"].str.lower()
    keep &= label.eq("bonafide") | scores["tool"].str.lower().eq(tool)
    sub = scores[keep]
    y = sub["label"].str.lower().eq("bonafide").to_numpy(int)
    if y.all() or not y.any():
        return None
    return float(compute_eer(sub["score"].to_numpy(np.float64), y)[0])


def measure_run(results: Path, run: str, train: set[str]) -> dict[str, float | None]:
    """Every measure of one run (``<system>_s<seed>``) from its per-clip score files."""
    out: dict[str, float | None] = {}
    for name, (stem, tool) in MEASURES.items():
        path = results / f"{run}__{stem}_scores.csv"
        out[name] = (
            held_out_eer(pd.read_csv(path, dtype={"speaker": str}), tool, train)
            if path.is_file()
            else None
        )
    return out


def _paired(table: dict, system: str, key: str, seeds: list[int]) -> list[float]:
    return [table[system][s][key] for s in seeds]


def judge_fold(table: dict, fold: str, seeds: list[int]) -> dict:
    """The registered decision for one fold against the control."""
    tool = UNSEEN[fold]
    fold_v, ctrl = _paired(table, fold, tool, seeds), _paired(table, "control", tool, seeds)
    lower = sum(f < c for f, c in zip(fold_v, ctrl, strict=True))
    p = mann_whitney_exact(fold_v, ctrl)
    rises = {
        key: mean(_paired(table, fold, key, seeds)) - mean(_paired(table, "control", key, seeds))
        for key in ("new_tortoise", "control_fa")
    }
    conditions = {
        "1_lower_in_4_of_5": lower >= MIN_LOWER,
        "2_mann_whitney": p <= MAX_P,
        "3_tortoise_within_1_point": rises["new_tortoise"] <= MAX_TORTOISE_RISE,
        "4_false_alarms_within_half_point": rises["control_fa"] <= MAX_FALSE_ALARM_RISE,
    }
    return {
        "unseen_tool": tool,
        "fold": fold_v,
        "control": ctrl,
        "lower_in_seeds": lower,
        "mann_whitney_p": p,
        "tortoise_mean_rise": rises["new_tortoise"],
        "false_alarm_mean_rise": rises["control_fa"],
        "conditions": conditions,
        "decision": "confirmed" if all(conditions.values()) else "not confirmed",
    }


def judge(table: dict) -> dict:
    """The study's answer from ``table[system][seed][measure]``."""
    seeds = [s for s in SEEDS if all(s in table.get(sys_, {}) for sys_ in ("control", *UNSEEN))]
    needed = [k for k in (*UNSEEN.values(), "new_tortoise", "control_fa")]
    seeds = [
        s
        for s in seeds
        if all(table[sys_][s].get(k) is not None for sys_ in ("control", *UNSEEN) for k in needed)
    ]
    out: dict = {"seeds": seeds, "planned_seeds": list(SEEDS)}
    if len(seeds) < len(SEEDS):
        out["answer"] = "incomplete"
        return out
    out["folds"] = {fold: judge_fold(table, fold, seeds) for fold in UNSEEN}
    confirmed = sum(f["decision"] == "confirmed" for f in out["folds"].values())
    out["answer"] = {2: "carries over", 1: "partly", 0: "does not carry over"}[confirmed]
    return out


def means(table: dict) -> dict:
    """Mean and SD of every measure per system (reported, not judged)."""
    out: dict = {}
    for system, runs in table.items():
        out[system] = {}
        for key in (*MEASURES, "control_fa"):
            vals = [r[key] for r in runs.values() if r.get(key) is not None]
            if vals:
                sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                out[system][key] = {"mean": mean(vals), "sd": sd, "n": len(vals)}
    return out


def collect(results: Path, train: set[str]) -> dict:
    """``table[system][seed]`` from a results folder (its runs.json gives control_fa)."""
    runs = json.loads((results / "runs.json").read_text(encoding="utf-8"))
    table: dict = {s: {} for s in SYSTEMS}
    for r in runs:
        system, seed = r.get("system"), int(r["seed"])
        if system not in table or r.get("status") != "ok":
            continue
        row = measure_run(results, f"{system}_s{seed}", train)
        row["control_fa"] = r.get("control_fa")
        table[system][seed] = row
    return table


def main() -> None:
    """CLI: judge a results folder; write one JSON."""
    import argparse

    parser = argparse.ArgumentParser(description="The generator study, by its registered rule")
    parser.add_argument("--results", required=True, help="folder with runs.json and score files")
    parser.add_argument("--pools", default=f"data/manifests/{gs.POOLS_OUT}")
    parser.add_argument("--out", default="experiments/results/generator_study_v1.json")
    args = parser.parse_args()
    table = collect(Path(args.results), gs.train_speakers(args.pools))
    report = judge(table)
    report["means"] = means(table)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for fold, f in report.get("folds", {}).items():
        print(
            f"{fold} on unseen {f['unseen_tool']}: lower in {f['lower_in_seeds']}/5, "
            f"p = {f['mann_whitney_p']:.4f}, tortoise {100 * f['tortoise_mean_rise']:+.2f}, "
            f"false alarms {100 * f['false_alarm_mean_rise']:+.2f} -> {f['decision']}"
        )
    print("ANSWER:", report["answer"])


if __name__ == "__main__":
    main()
