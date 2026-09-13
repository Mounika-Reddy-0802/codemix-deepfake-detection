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


class AblationDataError(RuntimeError):
    """Raised when a run or a floor the table needs is missing."""


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


def table(summary: dict, floors: dict[tuple[str, str], float]) -> pd.DataFrame:
    """One row per adapter x set: EER, CI, floor, margin, and whether it clears the floor."""
    rows = []
    for run, (mix, condition) in ADAPTERS.items():
        for ev in SETS:
            key = f"{run}__{ev}"
            if key not in summary:
                raise AblationDataError(f"run missing from the summary: {key}")
            r = summary[key]
            floor = floors[(ev, condition)]
            eer = _pct(r["eer"])
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
                    "floor": floor,
                    "margin": round(floor - eer, 2),
                    "clears_floor": _pct(r["eer_ci_high"]) < floor,
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
    frame = table(summary, load_floors())
    effect = rvc_effect(frame)
    result = {
        "question": "W9-T1: does attack diversity (XTTS+RVC) improve unseen-tool EER?",
        "answer": verdict(effect),
        "source": summary_path,
        "floors_pct": {f"{s}/{c}": v for (s, c), v in load_floors().items()},
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
