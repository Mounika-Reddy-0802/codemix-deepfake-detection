"""How far a model's clip scores separate genuine spontaneous speech from each tool's clones.

Threshold-free: one EER per tool, genuine spontaneous clips (HiACC, the IndicVoices test
speakers) against that tool's clones, from per-clip score files. An operating point cannot
move it (``operating_points_spont_v1.md``); training can (``spontaneous_training_v1.md``).
It chooses nothing. Genuine scores from one file are ranked against clone scores from
another, so every file of one separation must hold one model's scores on one scale: the
adapters write P(genuine), SSL-AASIST log-odds, and a mix is refused. A score file without a
``label`` column (HiACC's false-alarm scores) is genuine throughout; ``name=path@tool`` keeps
one tool of a mixed clone set::

    python -m src.reporting.spontaneous_separation \\
        --genuine hiacc_scores.csv iv_test_scores.csv \\
        --clones xtts_v2=iv_test_scores.csv tortoise=cm04_scores.csv indicf5=cm07_scores.csv \\
        tortoise_new=new_speakers_scores.csv@tortoise --out separation.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def genuine_scores(paths: list[str]) -> np.ndarray:
    """Every genuine clip's score from the given files."""
    parts = []
    for path in paths:
        frame = pd.read_csv(path)
        if "label" in frame:
            frame = frame[frame["label"] == "bonafide"]
        parts.append(frame["score"].to_numpy(float))
    return np.concatenate(parts)


def clone_scores(path: str, tool: str | None = None) -> np.ndarray:
    """The clones' scores in one file, optionally of one tool only."""
    frame = pd.read_csv(path)
    keep = frame["label"] == "spoof"
    if tool:
        keep &= frame["tool"] == tool
    return frame.loc[keep, "score"].to_numpy(float)


def separation(genuine: np.ndarray, clones: dict[str, np.ndarray]) -> dict:
    """EER (%) of the genuine clips against each named clone set, all on one score scale."""
    from src.training.metrics import compute_eer

    scales = {bool(((a >= 0) & (a <= 1)).all()) for a in (genuine, *clones.values()) if a.size}
    if len(scales) > 1:
        raise ValueError("score files mix P(genuine) and log-odds: use one model's files")
    out = {"genuine_clips": int(genuine.size), "tools": {}}
    for name, fake in clones.items():
        scores = np.concatenate([genuine, fake])
        labels = np.concatenate([np.ones(genuine.size), np.zeros(fake.size)])
        eer, _ = compute_eer(scores, labels)
        out["tools"][name] = {"clones": int(fake.size), "eer_pct": round(100 * float(eer), 2)}
    return out


def main() -> None:
    """CLI: genuine files, ``name=path[@tool]`` clone sets, one JSON out."""
    import argparse

    parser = argparse.ArgumentParser(description="Spontaneous genuine speech against clones")
    parser.add_argument("--genuine", nargs="+", required=True)
    parser.add_argument("--clones", nargs="+", required=True, help="name=path[@tool]")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    clones = {}
    for spec in args.clones:
        name, _, rest = spec.partition("=")
        path, _, tool = rest.partition("@")
        clones[name] = clone_scores(path, tool or None)
    report = separation(genuine_scores(args.genuine), clones)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, v in report["tools"].items():
        print(
            f"{name:14s} {v['eer_pct']:6.2f}% EER  ({v['clones']} clones, {report['genuine_clips']} genuine)"
        )


if __name__ == "__main__":
    main()
