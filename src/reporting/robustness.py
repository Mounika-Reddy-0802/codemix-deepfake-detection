"""The robustness and evasion sweep, scored (plan task A4).

``src.data.robustness`` renders 1,122 clips (genuine new speakers, XTTS-v2, Tortoise and
IndicF5 clones) under 15 post-processing transforms before the phone line. This module
reads one per-clip score file per transform and reports, for each transform:

- the EER of each clone tool against the genuine clips (the same clips under the same
  transform), with :func:`src.training.metrics.eer`;
- false alarms (genuine called fake) and misses per tool at the fixed clip rule, fake when
  P(genuine) < 0.5 (``src.reporting.det_curves.clip_metrics``);
- the change from the untransformed reference (``none``).

A transform helps an attacker when misses or EER rise on the clones; it hurts a genuine
caller when false alarms rise. Nothing is chosen here. Score files are named
``<scores-dir>/sweep_<transform>_scores.csv``::

    python -m src.reporting.robustness --scores-dir experiments/robustness_v1/final_s1234 \\
        --model "Final adapter" --out experiments/results/robustness_v1_final.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.robustness import TRANSFORMS
from src.reporting.det_curves import clip_metrics
from src.training.metrics import eer

FAKE_GROUPS = ("xtts_v2", "tortoise", "indicf5")


def measure(scores: pd.DataFrame) -> dict:
    """EER per clone tool against the genuine clips, and the rule's error rates."""
    genuine = scores[scores["group"] == "genuine"]
    g_scores = genuine["score"].to_numpy(dtype=float)
    out = {"false_alarm": clip_metrics(g_scores, np.ones(len(g_scores), dtype=int))["false_alarm"]}
    for tool in FAKE_GROUPS:
        fake = scores.loc[scores["group"] == tool, "score"].to_numpy(dtype=float)
        if len(fake) == 0:
            continue
        both = np.concatenate([g_scores, fake])
        labels = np.concatenate([np.ones(len(g_scores)), np.zeros(len(fake))]).astype(int)
        out[f"eer_{tool}"] = float(eer(both, labels)) * 100
        out[f"miss_{tool}"] = clip_metrics(fake, np.zeros(len(fake), dtype=int))["miss"]
    return out


def table(scores_dir: str | Path) -> pd.DataFrame:
    """One row per transform found, with the change from ``none`` for every measure."""
    rows = []
    for name in TRANSFORMS:
        path = Path(scores_dir) / f"sweep_{name}_scores.csv"
        if path.is_file():
            rows.append({"transform": name, **measure(pd.read_csv(path))})
    frame = pd.DataFrame(rows).set_index("transform")
    if "none" not in frame.index:
        raise ValueError(f"no reference (sweep_none_scores.csv) in {scores_dir}")
    deltas = (frame - frame.loc["none"]).add_prefix("delta_")
    return frame.join(deltas).reset_index()


def markdown(frame: pd.DataFrame, model: str) -> str:
    """The table for a results document: values with their change from ``none``."""
    cols = ["false_alarm"] + [c for c in frame.columns if c.startswith(("eer_", "miss_"))]
    names = {"false_alarm": "False alarm"} | {
        c: c.replace("eer_", "EER ").replace("miss_", "Miss ") for c in cols[1:]
    }
    lines = [
        f"**{model}** (%; change from no transform in brackets)",
        "",
        "| Transform | " + " | ".join(names[c] for c in cols) + " |",
        "|---|" + "---:|" * len(cols),
    ]
    for _, row in frame.iterrows():
        cells = []
        for c in cols:
            value, delta = row[c], row[f"delta_{c}"]
            cells.append(
                f"{value:.1f}" if row["transform"] == "none" else f"{value:.1f} ({delta:+.1f})"
            )
        lines.append(f"| {row['transform']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    """CLI: score files of one model -> JSON and a markdown table."""
    import argparse

    parser = argparse.ArgumentParser(description="Robustness sweep report")
    parser.add_argument("--scores-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    frame = table(args.scores_dir)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(
            {"model": args.model, "rows": json.loads(frame.to_json(orient="records"))}, indent=1
        ),
        encoding="utf-8",
    )
    print(markdown(frame, args.model))


if __name__ == "__main__":
    main()
