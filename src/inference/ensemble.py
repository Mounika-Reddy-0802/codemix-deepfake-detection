"""Average two SSL-AASIST checkpoints on the live path (plan task A1).

The larger encoder's real-call result rested on one checkpoint, and two trainings of the
same recipe differ a lot on the hardest sets (Chatterbox, spontaneous speech). The
simplest way to use both is to average them: for every window the live path scores, the
ensemble's P(genuine) is the mean of the two checkpoints' P(genuine), and the mean is
smoothed exactly as the live path smooths a single model (``real_calls.as_silence`` with
nothing masked, the same moving average).

Its operating point is chosen like every other one in the project: on the 506 dev clips
only, by ``python -m src.inference.calibrate choose`` on the averaged dev windows. Nothing
is chosen on a call or a test set.

Both checkpoints must have scored the same windows. The silence gate depends on the
audio's level, not on the model, so the two window files skip the same windows; a
mismatch means the files were not made from the same audio and is an error. Usage::

    python -m src.inference.ensemble --a A_windows.csv --b B_windows.csv --out AVG.csv

Per-clip score files (the clip scorer's log-odds of genuine, one 4 s crop per clip) are
averaged the same way, as P(genuine), with ``--clips``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.inference.real_calls import as_silence

KEYS = ["filepath", "window"]


def average_windows(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """The mean of two window files' scores, re-smoothed as the live path does."""
    if len(a) != len(b):
        raise ValueError(f"the window files differ in length: {len(a)} and {len(b)}")
    joined = a.merge(b[[*KEYS, "score"]], on=KEYS, how="outer", suffixes=("", "_b"), indicator=True)
    if (joined["_merge"] != "both").any():
        raise ValueError("the window files do not cover the same windows")
    skipped_a, skipped_b = joined["score"].isna(), joined["score_b"].isna()
    if (skipped_a != skipped_b).any():
        raise ValueError("the window files skip different windows: not the same audio")
    joined["score"] = (joined["score"] + joined["score_b"]) / 2.0
    out = joined.drop(columns=["score_b", "_merge"])
    out = as_silence(out, pd.Series(False, index=out.index))
    return out[a.columns]


def average_clips(a: pd.DataFrame, b: pd.DataFrame, key: str = "filepath") -> pd.DataFrame:
    """The mean P(genuine) of two SSL-AASIST clip score files (log-odds in, P out).

    Both files must hold exactly the same clips; ``a``'s columns and row order are kept.
    """
    joined = a.merge(
        b[[key, "score"]],
        on=key,
        how="left",
        suffixes=("", "_b"),
        indicator=True,
        validate="one_to_one",
    )
    if (joined["_merge"] != "both").any() or len(a) != len(b):
        raise ValueError("the clip score files do not cover the same clips")
    p_a = 1.0 / (1.0 + np.exp(-joined["score"].to_numpy(dtype=np.float64)))
    p_b = 1.0 / (1.0 + np.exp(-joined["score_b"].to_numpy(dtype=np.float64)))
    joined["score"] = (p_a + p_b) / 2.0
    return joined[a.columns]


def main() -> None:
    """CLI: write the averaged window file (or, with ``--clips``, clip score file)."""
    import argparse

    parser = argparse.ArgumentParser(description="Average two checkpoints' window scores")
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--clips", action="store_true", help="per-clip log-odds files")
    args = parser.parse_args()
    if args.clips:
        out = average_clips(pd.read_csv(args.a), pd.read_csv(args.b))
        out.to_csv(args.out, index=False)
        print(f"{len(out)} clips -> {args.out}; mean P(genuine) {out['score'].mean():.4f}")
        return
    out = average_windows(pd.read_csv(args.a), pd.read_csv(args.b))
    out.to_csv(args.out, index=False)
    scored = out["smoothed"].notna()
    print(f"{len(out)} windows ({int(scored.sum())} scored) -> {args.out}")
    print(f"mean P(genuine): {np.nanmean(out['score']):.4f}")


if __name__ == "__main__":
    main()
