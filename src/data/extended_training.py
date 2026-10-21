"""The corpus-overfit fix: the deployed recipe trained on many more genuine speakers.

The deployed S2 adapts on 21 genuine speakers (1,217 clips). Its false alarms on new
recordings and its unseen-tool errors could come from having seen too few voices, so
this run keeps everything else fixed (the recipe, the dev set, the Stage-1 start) and
only grows the training manifest with:

* the 227 extra adaptation speakers (``adapt_ext`` pool, 3,405 genuine clips), and
* their XTTS clones (CM08, 1,759 clips that passed the screen).

Both come from pools carved before any of their clips was scored, and neither shares
a speaker with an evaluation, extension-test or control pool or with the dev set;
``build`` checks this again rather than trusting the file names.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

BASE = "codemix_adapt_train_rvc_channel20"
ADDITIONS = ("adapt_ext_genuine_channel20", "cm08_channel20")
DEV = "codemix_adapt_dev_channel20"
OUT = "codemix_adapt_train_ext_rvc_channel20"
POOLS_EXT = "speaker_pools_ext.csv"
TRAINING_POOL = "adapt_ext"


def build(manifests_dir: str) -> pd.DataFrame:
    """The base training manifest plus the additions, after the speaker checks."""
    mdir = Path(manifests_dir)

    def read(stem: str) -> pd.DataFrame:
        return pd.read_csv(mdir / f"{stem}.csv", dtype={"speaker": str})

    base = read(BASE)
    pools = pd.read_csv(mdir / POOLS_EXT, dtype={"speaker": str})
    allowed = set(pools.loc[pools["pool"] == TRAINING_POOL, "speaker"])
    dev_speakers = set(read(DEV)["speaker"])
    parts = [base]
    for stem in ADDITIONS:
        extra = read(stem)
        outside = set(extra["speaker"]) - allowed
        if outside:
            raise ValueError(f"{stem}: {len(outside)} speaker(s) outside the {TRAINING_POOL} pool")
        if set(extra["speaker"]) & dev_speakers:
            raise ValueError(f"{stem} shares a speaker with the dev set")
        if set(extra["split"]) != {"train"}:
            raise ValueError(f"{stem} holds rows not marked for training")
        parts.append(extra[base.columns])
    out = pd.concat(parts, ignore_index=True)
    if out["filepath"].duplicated().any():
        raise ValueError("a clip appears twice")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the extended training manifest")
    parser.add_argument("--manifests", default="data/manifests")
    args = parser.parse_args()
    out = build(args.manifests)
    out.to_csv(Path(args.manifests) / f"{OUT}.csv", index=False)
    counts = out.groupby("label")["speaker"].agg(["size", "nunique"])
    print(f"{OUT}: {len(out)} clips\n{counts}")


if __name__ == "__main__":
    main()
