"""W9-T1 ablation: XTTS-only against XTTS+RVC adaptation, read against shortcut floors.

The W10 Kaggle run trained four LoRA adapters on the normalised bundle -- two data
mixes (XTTS only, XTTS + the RVC training half) in two conditions (clean, G.711 at
20 dB) -- and scored each on three sets: the eval pool (XTTS, the seen tool), CM04
(Tortoise, never trained on) and the RVC test half (speakers disjoint from the RVC
training half, P-026).

A raw EER on this corpus means nothing without the low-level-cue floor measured on
the *same* set in the *same* condition (``lowlevel_cue_check_v1.md``): eight cheap
signal statistics already reach it. So every cell here carries its floor and the
margin below it, and a cell whose EER is not below its floor is marked as not
evidence of learning, however low it looks.

The ablation's question (plan, W9-T1): does attack diversity improve unseen-tool EER?
:func:`build` answers it from the numbers rather than by hand.

Pure json/pandas, so CI runs it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SUMMARY = "results/w10/w10_summary.json"
OUT = "experiments/results/ablations.json"

#: Adapter run name -> (training mix, condition).
ADAPTERS: dict[str, tuple[str, str]] = {
    "lora_norm_clean": ("xtts", "clean"),
    "lora_norm_rvc_clean": ("xtts+rvc", "clean"),
    "lora_norm_channel": ("xtts", "channel"),
    "lora_norm_rvc_channel": ("xtts+rvc", "channel"),
}

#: Scored set -> what it tests.
SETS: dict[str, str] = {
    "eval_pool": "seen tool (XTTS)",
    "cm04": "unseen tool (Tortoise)",
    "rvc_test": "RVC, unseen speakers",
}

#: (set, condition) -> low-level-cue gate JSON on that set, normalised audio.
FLOORS: dict[tuple[str, str], str] = {
    ("eval_pool", "clean"): "experiments/results/lowlevel_cue_check_normalised_bundle.json",
    ("eval_pool", "channel"): "experiments/results/lowlevel_cue_check_normalised_channel20.json",
    ("cm04", "clean"): "experiments/results/lowlevel_cue_check_cm04_normalised.json",
    ("cm04", "channel"): "experiments/results/lowlevel_cue_check_cm04_normalised_channel20.json",
    ("rvc_test", "clean"): "experiments/results/lowlevel_cue_check_rvc_holdout_normalised.json",
    ("rvc_test", "channel"): (
        "experiments/results/lowlevel_cue_check_rvc_holdout_normalised_channel20.json"
    ),
}


#: Matched floors with paired, speaker-level verdicts (``src/training/floor_compare.py``,
#: ``docs/results/shortcut_floor_v2.md``): each detector's floor is fitted on its own
#: training data and scored on its own clips. Supersedes the per-set floors above as
#: the evidence test; those stay in every cell as ``floor_v1``.
MATCHED = "experiments/results/floor_compare_v2.json"


class AblationDataError(RuntimeError):
    """Raised when a run or a floor the table needs is missing."""


def load_matched(path: str = MATCHED) -> dict[tuple[str, str], dict]:
    """(run, set) -> matched floor (%), paired difference interval (%) and its verdict."""
    from src.training.floor_compare import SYSTEMS

    if not Path(path).is_file():
        raise AblationDataError(f"matched floors not computed: {path}")
    run_of = {system.name: Path(system.scores_prefix).name for system in SYSTEMS}
    out = {}
    for row in json.loads(Path(path).read_text())["results"]:
        out[(run_of[row["system"]], row["test_set"])] = {
            "floor": _pct(row["eer_b"]),
            "diff_ci": [_pct(row["diff_ci"][0]), _pct(row["diff_ci"][1])],
            "verdict": row["verdict"],
        }
    return out


def floor_fields(eer: float, ci_high: float, floor_v1: float, matched: dict | None) -> dict:
    """The floor columns of one cell.

    With a matched entry, the evidence test is the paired speaker-level verdict against
    the matched floor; the old per-set floor is kept beside it as ``floor_v1``. Without
    one, the old rule: the clip-level interval's upper bound below the per-set floor.
    """
    if matched is None:
        return {
            "floor": floor_v1,
            "margin": round(floor_v1 - eer, 2),
            "clears_floor": ci_high < floor_v1,
        }
    from src.training.floor_compare import BEATS_FLOOR

    return {
        "floor": matched["floor"],
        "margin": round(matched["floor"] - eer, 2),
        "clears_floor": matched["verdict"] == BEATS_FLOOR,
        "floor_verdict": matched["verdict"],
        "diff_ci": matched["diff_ci"],
        "floor_v1": floor_v1,
    }


def _pct(x: float) -> float:
    return round(100.0 * float(x), 2)


def load_floors(floors: dict[tuple[str, str], str] = FLOORS) -> dict[tuple[str, str], float]:
    """Floor EER (%) per (set, condition), read from the committed gate JSONs."""
    out = {}
    for key, path in floors.items():
        if not Path(path).is_file():
            raise AblationDataError(f"shortcut floor for {key} not measured: {path}")
        out[key] = _pct(json.loads(Path(path).read_text())["eer"])
    return out


def table(
    summary: dict,
    floors: dict[tuple[str, str], float],
    matched: dict[tuple[str, str], dict] | None = None,
) -> pd.DataFrame:
    """One row per adapter x set: EER, CI, floor, margin, and whether it clears the floor.

    With ``matched`` (:func:`load_matched`), each cell is judged against its matched
    floor by the paired speaker-level test; without it, against the per-set floor.
    """
    rows = []
    for run, (mix, condition) in ADAPTERS.items():
        for ev in SETS:
            key = f"{run}__{ev}"
            if key not in summary:
                raise AblationDataError(f"run missing from the summary: {key}")
            r = summary[key]
            eer = _pct(r["eer"])
            entry = None if matched is None else matched.get((run, ev))
            if matched is not None and entry is None:
                raise AblationDataError(f"no matched floor for {run} on {ev}")
            rows.append(
                {
                    "adapter": run,
                    "mix": mix,
                    "condition": condition,
                    "set": ev,
                    "eer": eer,
                    "eer_ci": [_pct(r["eer_ci_low"]), _pct(r["eer_ci_high"])],
                    "auc": round(float(r["auc"]), 4),
                    "clips": int(r["clips"]),
                    **floor_fields(eer, _pct(r["eer_ci_high"]), floors[(ev, condition)], entry),
                }
            )
    return pd.DataFrame(rows)


def rvc_effect(frame: pd.DataFrame) -> pd.DataFrame:
    """EER change from adding RVC to training, per condition and set (negative = better)."""
    wide = frame.pivot_table(index=["condition", "set"], columns="mix", values="eer")
    wide["delta"] = (wide["xtts+rvc"] - wide["xtts"]).round(2)
    return wide.reset_index()[["condition", "set", "xtts", "xtts+rvc", "delta"]]


def verdict(effect: pd.DataFrame) -> str:
    """The W9-T1 answer, stated from the deltas."""
    unseen = effect[effect["set"] == "cm04"]
    rvc = effect[effect["set"] == "rvc_test"]
    worse = bool((unseen["delta"] > 0).all())
    better = bool((rvc["delta"] < 0).all())
    if worse and better:
        return (
            "No. Adding RVC fixes the family it adds (RVC on unseen speakers improves in "
            "both conditions) but makes the unseen tool worse in both conditions: attack "
            "diversity trades unseen-tool EER for seen-family coverage."
        )
    if not worse and better:
        return "Yes. Adding RVC improves RVC detection without hurting the unseen tool."
    return "Mixed: see the per-condition deltas."


def build(summary_path: str = SUMMARY, out: str = OUT) -> dict:
    """Write ``ablations.json`` and return it."""
    summary = json.loads(Path(summary_path).read_text())
    frame = table(summary, load_floors(), load_matched())
    effect = rvc_effect(frame)
    result = {
        "question": "W9-T1: does attack diversity (XTTS+RVC) improve unseen-tool EER?",
        "answer": verdict(effect),
        "source": summary_path,
        "floor_rule": "matched floor, paired speaker-level test (floor_compare_v2.json)",
        "floors_v1_pct": {f"{s}/{c}": v for (s, c), v in load_floors().items()},
        "rvc_effect_pct": effect.to_dict(orient="records"),
        "cells": frame.to_dict(orient="records"),
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    """CLI: ``python -m src.reporting.ablation`` -- writes experiments/results/ablations.json."""
    import argparse

    parser = argparse.ArgumentParser(description="XTTS vs XTTS+RVC ablation table")
    parser.add_argument("--summary", default=SUMMARY)
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()
    result = build(args.summary, args.out)
    cells = pd.DataFrame(result["cells"])
    print(cells.pivot(index="adapter", columns="set", values="eer").to_string())
    print(pd.DataFrame(result["rvc_effect_pct"]).to_string(index=False))
    print(result["answer"])


if __name__ == "__main__":
    main()
