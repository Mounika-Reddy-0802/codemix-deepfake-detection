"""English over the phone line: what the deployed detector costs English callers.

External review comment 4 asks what channel-matched Hinglish adaptation costs on
English. EER alone hides the part a deployment feels, so this module reads the four
score files in ``experiments/results/english_channel_v1/`` (deployed S2 and the
English-only S1, each on the same 6,004-clip English sample, clean and over the G.711
phone line) and reports, per model and condition:

- the EER and its interval, copied from the evaluate JSON;
- at the deployed threshold (P(genuine) < 0.5 is called fake): the share of genuine
  English clips called fake, with a 95% speaker-bootstrap interval
  (``src.training.false_alarm.summarise``), and the share of English fakes caught;
- the EER of each of the 13 ASVspoof attacks, to see which attacks the phone line
  hides.

A score file not yet produced is reported as ``None``. Pure pandas, so CI runs it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.training.false_alarm import DEPLOYED_THRESHOLD, summarise

RESULTS_DIR = "experiments/results/english_channel_v1"
MODELS = {"deployed_s2": "Deployed S2", "s1": "S1 (English only)"}
CONDITIONS = {"clean": "clean", "channel": "phone line"}


def fakes_caught(scores: pd.DataFrame, threshold: float = DEPLOYED_THRESHOLD) -> float:
    """Share of spoof clips whose P(genuine) falls below ``threshold``."""
    spoof = scores.loc[scores["label"].astype(str).str.lower() == "spoof", "score"]
    if spoof.empty:
        raise ValueError("no spoof clips to measure")
    return float((spoof < threshold).mean())


def cell(results_dir: str | Path, model: str, condition: str) -> dict | None:
    """Every number for one model in one condition, or None if it was not scored."""
    stem = Path(results_dir) / f"{model}__english_{condition}"
    json_path, scores_path = stem.with_suffix(".json"), Path(f"{stem}_scores.csv")
    if not (json_path.is_file() and scores_path.is_file()):
        return None
    result = json.loads(json_path.read_text())
    pooled = result["pooled"]
    scores = pd.read_csv(scores_path)
    genuine = summarise(scores)
    return {
        "eer": round(100 * pooled["eer"], 2),
        "eer_ci": [round(100 * pooled["eer_ci_low"], 2), round(100 * pooled["eer_ci_high"], 2)],
        "auc": round(pooled["auc"], 3),
        "genuine_called_fake": round(100 * genuine["false_alarm_rate"], 2),
        "genuine_called_fake_ci": [round(100 * x, 2) for x in genuine["false_alarm_ci"]],
        "fakes_caught": round(100 * fakes_caught(scores), 2),
        "per_attack_eer": {
            a["attack"]: round(100 * a["eer"], 2) for a in result.get("per_attack", [])
        },
        "clips": int(pooled["clips"]),
        "speakers": int(scores["speaker"].nunique()),
    }


def build(results_dir: str | Path = RESULTS_DIR) -> dict:
    """All model x condition cells, plus the threshold they were read at."""
    return {
        "threshold": DEPLOYED_THRESHOLD,
        "cells": {
            model: {cond: cell(results_dir, model, cond) for cond in CONDITIONS} for model in MODELS
        },
    }


def _num(cell: dict | None, key: str, ci_key: str | None = None) -> str:
    if cell is None:
        return "not measured"
    text = f"{cell[key]:.2f}"
    if ci_key:
        lo, hi = cell[ci_key]
        text += f" [{lo:.2f}, {hi:.2f}]"
    return text


def markdown(result: dict) -> str:
    """The main table (one row per model and condition) and the per-attack table."""
    lines = [
        "| Model | Condition | EER (%) | Genuine called fake at 0.5 (%) | Fakes caught at 0.5 (%) |",
        "|---|---|---:|---:|---:|",
    ]
    for model, label in MODELS.items():
        for cond, cond_label in CONDITIONS.items():
            c = result["cells"][model][cond]
            lines.append(
                f"| {label} | {cond_label} | {_num(c, 'eer', 'eer_ci')} | "
                f"{_num(c, 'genuine_called_fake', 'genuine_called_fake_ci')} | "
                f"{_num(c, 'fakes_caught')} |"
            )
    columns = [
        (model, cond)
        for model in MODELS
        for cond in CONDITIONS
        if result["cells"][model][cond] is not None
    ]
    if not columns:
        return "\n".join(lines)
    attacks = sorted(
        {a for m, c in columns for a in result["cells"][m][c]["per_attack_eer"]},
        key=lambda a: int(a.lstrip("A")) if a.lstrip("A").isdigit() else 999,
    )
    header = " | ".join(f"{MODELS[m]}, {CONDITIONS[c]}" for m, c in columns)
    lines += ["", f"| Attack | {header} |", "|---|" + "---:|" * len(columns)]
    for attack in attacks:
        row = " | ".join(
            f"{result['cells'][m][c]['per_attack_eer'].get(attack, float('nan')):.2f}"
            for m, c in columns
        )
        lines.append(f"| {attack} | {row} |")
    return "\n".join(lines)


def main() -> None:
    """CLI: ``python -m src.reporting.english_channel``."""
    import argparse

    parser = argparse.ArgumentParser(description="English over the phone line")
    parser.add_argument("--results", default=RESULTS_DIR)
    parser.add_argument("--out", default=f"{RESULTS_DIR}/summary.json")
    args = parser.parse_args()
    result = build(args.results)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(markdown(result))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
