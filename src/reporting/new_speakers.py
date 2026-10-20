"""Results on the week-13 sets: new speakers and a second unseen tool.

The paper's original test pool has 15 speakers. Week 13 added a second, larger test
population and a second unseen generator:

* **new speakers** - 60 MUCS speakers never used anywhere before (``eval_ext`` pool):
  2,400 genuine clips, 1,770 XTTS clones (CM05, the tool the S2 systems adapt on) and
  222 Tortoise clones (CM06, a tool no system trains on);
* **IndicF5** - 526 clones by a modern flow-matching Hindi TTS (CM07) against 651
  all-Hindi genuine clips of the same 54 test speakers; no system trains on it.

Every model's per-clip scores (``experiments/results/ext_scoring_v1``) give one EER per
(model, condition, attack), each against the shared genuine set of its speakers, with a
95% interval from resampling whole speakers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.training.metrics import eer

SCORES = "experiments/results/ext_scoring_v1"
MODELS = {
    "s1": "S1 (English only)",
    "s2_clean": "S2 LoRA, XTTS, trained clean",
    "s2_channel": "S2 LoRA, XTTS, trained phone line",
    "s2_rvc_clean": "S2 LoRA, XTTS+RVC, trained clean",
    "deployed_s2": "S2 LoRA, XTTS+RVC, trained phone line (deployed)",
}
ATTACKS = {  # column -> (scoring set, tool)
    "new XTTS (CM05)": ("score_ext", "xtts_v2"),
    "new Tortoise (CM06)": ("score_ext", "tortoise"),
    "IndicF5 (CM07)": ("score_cm07", "indicf5"),
}
CONDITIONS = ("clean", "channel")


def attack_eer(scores: pd.DataFrame, tool: str, n_boot: int = 1000, seed: int = 0) -> dict:
    """EER of one attack against the genuine rows, with a speaker-bootstrap interval."""
    genuine = scores["label"].str.lower() == "bonafide"
    part = scores[genuine | (scores["tool"] == tool)].reset_index(drop=True)
    labels = (part["label"].str.lower() == "bonafide").to_numpy().astype(int)
    values = part["score"].to_numpy(dtype=float)
    speakers = part["speaker"].astype(str).to_numpy()
    point = float(eer(values, labels))
    rng = np.random.default_rng(seed)
    unique = np.unique(speakers)
    members = [np.flatnonzero(speakers == s) for s in unique]
    boot = []
    for _ in range(n_boot):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(unique), len(unique))])
        if labels[idx].min() == labels[idx].max():
            continue
        boot.append(eer(values[idx], labels[idx]))
    low, high = np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan)
    return {
        "eer": point,
        "ci_low": float(low),
        "ci_high": float(high),
        "spoof_clips": int((labels == 0).sum()),
        "genuine_clips": int(labels.sum()),
    }


def analyse(scores_dir: str = SCORES, n_boot: int = 1000) -> list[dict]:
    rows = []
    for tag, name in MODELS.items():
        for cond in CONDITIONS:
            row = {"model": name, "tag": tag, "condition": cond}
            for column, (scoring_set, tool) in ATTACKS.items():
                path = Path(scores_dir) / f"{tag}__{scoring_set}_{cond}_scores.csv"
                if path.is_file():
                    scores = pd.read_csv(path)
                    if (scores["tool"] == tool).any():
                        row[column] = attack_eer(scores, tool, n_boot)
                        row[column]["per_speaker"] = per_speaker(scores, tool)
            if len(row) > 3:
                rows.append(row)
    return rows


def per_speaker(scores: pd.DataFrame, tool: str) -> dict:
    """EER of one attack computed separately for every speaker; the spread over speakers."""
    genuine = scores["label"].str.lower() == "bonafide"
    part = scores[genuine | (scores["tool"] == tool)]
    values = []
    for _, group in part.groupby(part["speaker"].astype(str)):
        labels = (group["label"].str.lower() == "bonafide").to_numpy().astype(int)
        if labels.min() != labels.max():
            values.append(eer(group["score"].to_numpy(dtype=float), labels))
    v = np.asarray(values)
    return {
        "speakers": int(len(v)),
        "median": float(np.median(v)),
        "min": float(v.min()),
        "max": float(v.max()),
        "above_20pct": int((v > 0.20).sum()),
    }


def markdown(rows: list[dict]) -> str:
    lines = [
        "# Results on the week-13 sets: new speakers and a second unseen tool (v1)",
        "",
        "EER (%) with a 95% speaker-bootstrap interval. New speakers: 60 speakers never",
        "used before, 2,400 genuine clips. IndicF5: 54 test speakers, 651 all-Hindi genuine",
        "clips. XTTS is the tool the S2 systems adapt on; Tortoise and IndicF5 are unseen by",
        "every system. Built by `src/reporting/new_speakers.py` from",
        "`experiments/results/ext_scoring_v1`.",
        "",
        "| Model | Audio | " + " | ".join(ATTACKS) + " |",
        "|---|---|" + "---|" * len(ATTACKS),
    ]
    for row in rows:
        cells = []
        for column in ATTACKS:
            r = row.get(column)
            cells.append(
                "-"
                if r is None
                else f"{100 * r['eer']:.1f} [{100 * r['ci_low']:.1f}, {100 * r['ci_high']:.1f}]"
            )
        audio = "phone line" if row["condition"] == "channel" else "clean"
        lines.append(f"| {row['model']} | {audio} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Per-speaker spread (each speaker's EER on their own clips):",
        "",
        "| Model | Audio | Attack | Speakers | Median | Min | Max | Above 20% |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        audio = "phone line" if row["condition"] == "channel" else "clean"
        for column in ATTACKS:
            s = (row.get(column) or {}).get("per_speaker")
            if s:
                lines.append(
                    f"| {row['model']} | {audio} | {column} | {s['speakers']} | "
                    f"{100 * s['median']:.1f} | {100 * s['min']:.1f} | {100 * s['max']:.1f} | "
                    f"{s['above_20pct']} |"
                )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Results on the week-13 sets")
    parser.add_argument("--scores", default=SCORES)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--out", default="experiments/results/new_speakers_v1.json")
    parser.add_argument("--doc", default="docs/results/new_speakers_v1.md")
    args = parser.parse_args()
    rows = analyse(args.scores, args.n_boot)
    Path(args.out).write_text(json.dumps(rows, indent=2))
    Path(args.doc).write_text(markdown(rows), encoding="utf-8")
    print(markdown(rows))


if __name__ == "__main__":
    main()
