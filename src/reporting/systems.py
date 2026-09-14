"""The three-system comparison: S1 English-only, S2 LoRA, S3 native (W9-T2, owner M).

The paper's claims table (plan, section 5) needs three numbers side by side for each
evaluation set, and one more question answered: does training on code-mixed audio
cost English performance (W7-T2, reverse degradation)?

- **S2** is the four W10 LoRA adapters (``results/w10/w10_summary.json``).
- **S3** is the four native models trained on exactly the same manifests
  (``docs/results/s3_native/s3_summary.json``, P-028).
- **English** for S2 and S3 was scored in the same Kaggle session on the 71,237-clip
  ASVspoof 2019 LA eval partition; S1's English number is the W8-T1 retention run.

Every code-mixed cell is read against the shortcut floor of its own set and
condition (``ablation.FLOORS``), so a cell only counts as evidence when its CI upper
bound is below that floor. English has no floor: ASVspoof LA is the benchmark the
floor idea was borrowed from, where AffectDF's replica sits at chance.

S1 was never scored on the normalised code-mixed sets, so those cells are ``None``
and reported as unmeasured rather than filled with the pre-normalisation numbers.

Pure json/pandas, so CI runs it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.reporting.ablation import SETS, load_floors

S2_SUMMARY = "results/w10/w10_summary.json"
S3_SUMMARY = "docs/results/s3_native/s3_summary.json"
S1_ENGLISH = "experiments/asvspoof_retention_summary.json"
OUT = "experiments/results/systems_s1_s2_s3.json"
REVERSE_OUT = "experiments/results/s3_reverse_degradation.json"

#: (variant, condition) -> (S2 run, S3 run). Same manifests within a row.
PAIRS: dict[tuple[str, str], tuple[str, str]] = {
    ("xtts", "clean"): ("lora_norm_clean", "s3_native_clean"),
    ("xtts", "channel"): ("lora_norm_channel", "s3_native_channel"),
    ("xtts+rvc", "clean"): ("lora_norm_rvc_clean", "s3_native_rvc_clean"),
    ("xtts+rvc", "channel"): ("lora_norm_rvc_channel", "s3_native_rvc_channel"),
}


class SystemsDataError(RuntimeError):
    """Raised when a run the comparison needs is missing."""


def _pct(x: float | None) -> float | None:
    return None if x is None else round(100.0 * float(x), 2)


def _cell(summary: dict, key: str, floor: float | None) -> dict:
    if key not in summary:
        raise SystemsDataError(f"missing run: {key}")
    r = summary[key]
    eer, hi = _pct(r["eer"]), _pct(r["eer_ci_high"])
    return {
        "eer": eer,
        "eer_ci": [_pct(r["eer_ci_low"]), hi],
        "auc": round(float(r["auc"]), 4),
        "clips": int(r["clips"]),
        "floor": floor,
        "clears_floor": None if floor is None else hi < floor,
    }


def build_table(
    s2: dict, s3: dict, s1_english: float, floors: dict[tuple[str, str], float]
) -> pd.DataFrame:
    """One row per (system, variant, condition, set), including English."""
    rows = []
    for (variant, condition), (s2_run, s3_run) in PAIRS.items():
        for system, summary, run in (("S2 LoRA", s2, s2_run), ("S3 native", s3, s3_run)):
            for ev in SETS:
                cell = _cell(summary, f"{run}__{ev}", floors[(ev, condition)])
                rows.append(
                    {
                        "system": system,
                        "variant": variant,
                        "condition": condition,
                        "set": ev,
                        "run": run,
                        **cell,
                    }
                )
            # English lives in the S3 session's summary for both systems.
            cell = _cell(s3, f"{run}__english", None)
            rows.append(
                {
                    "system": system,
                    "variant": variant,
                    "condition": condition,
                    "set": "english",
                    "run": run,
                    **cell,
                }
            )
    for condition in ("clean", "channel"):
        rows.append(
            {
                "system": "S1 English-only",
                "variant": "asvspoof",
                "condition": condition,
                "set": "english",
                "run": "stage1_baseline",
                "eer": s1_english,
                "eer_ci": None,
                "auc": None,
                "clips": 71237,
                "floor": None,
                "clears_floor": None,
            }
        )
    return pd.DataFrame(rows)


def reverse_degradation(frame: pd.DataFrame, s1_english: float) -> dict:
    """English EER per model, and how far each system moved from S1."""
    english = frame[frame["set"] == "english"]
    models = []
    for _, row in english[english["system"] != "S1 English-only"].iterrows():
        models.append(
            {
                "system": row["system"],
                "run": row["run"],
                "variant": row["variant"],
                "condition": row["condition"],
                "english_eer": row["eer"],
                "english_ci": row["eer_ci"],
                "vs_s1_pp": round(row["eer"] - s1_english, 2),
            }
        )
    s2 = [m["english_eer"] for m in models if m["system"] == "S2 LoRA"]
    s3 = [m["english_eer"] for m in models if m["system"] == "S3 native"]
    at_chance = all(e >= 45.0 for e in s3)
    return {
        "question": "W7-T2: does native code-mixed training wreck English performance?",
        "s1_english_eer": s1_english,
        "s2_english_range": [min(s2), max(s2)],
        "s3_english_range": [min(s3), max(s3)],
        "answer": (
            "Yes. Every S3 model scores English at or near chance "
            f"({min(s3):.2f}-{max(s3):.2f}% EER) while S2 adapters stay at "
            f"{min(s2):.2f}-{max(s2):.2f}%: starting from the English checkpoint and "
            "adapting a small set of weights keeps the conventional task, training the "
            "encoder natively does not."
            if at_chance
            else "Partly: see the per-model English EERs."
        ),
        "models": models,
    }


def build(out: str = OUT, reverse_out: str = REVERSE_OUT) -> dict:
    """Write the comparison and the reverse-degradation artefact; return the comparison."""
    s2 = json.loads(Path(S2_SUMMARY).read_text())
    s3 = json.loads(Path(S3_SUMMARY).read_text())
    s1 = round(
        100.0 * json.loads(Path(S1_ENGLISH).read_text())["results"]["stage1_baseline"]["eer"], 2
    )
    frame = build_table(s2, s3, s1, load_floors())
    reverse = reverse_degradation(frame, s1)
    result = {
        "sources": {"s2": S2_SUMMARY, "s3": S3_SUMMARY, "s1_english": S1_ENGLISH},
        "s1_codemix_normalised": "not measured",
        "cells": json.loads(frame.to_json(orient="records")),
    }
    for path, payload in ((out, result), (reverse_out, reverse)):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2) + "\n")
    return result


def main() -> None:
    """CLI: ``python -m src.reporting.systems``."""
    result = build()
    frame = pd.DataFrame(result["cells"])
    frame["label"] = frame["system"] + " | " + frame["variant"] + " | " + frame["condition"]
    print(frame.pivot_table(index="label", columns="set", values="eer").to_string())
    print(json.loads(Path(REVERSE_OUT).read_text())["answer"])


if __name__ == "__main__":
    main()
