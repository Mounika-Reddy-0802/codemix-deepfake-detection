"""Training on two telephone codecs: the deployed recipe's clips over G.711 and AMR-NB.

The deployed S2 trains on clips passed through a simulated G.711 line. Scored through
the real AMR-NB mobile codec it loses most of that training's benefit
(``docs/results/amr_channel_v1.md``): it learned one chain, not telephone audio in
general. The direct test is to train on both. This manifest holds every clip of the
deployed training manifest twice, once over G.711 (as published) and once through
AMR-NB at 12.2 kbit/s, and nothing else: same speakers, same attacks, same dev set.

The two halves must be the same clips; ``build`` checks the clip names match one to one
and that each half carries a single, different channel condition.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

G711 = "codemix_adapt_train_rvc_channel20"
AMR = "codemix_adapt_train_rvc_amr"
OUT = "codemix_adapt_train_rvc_mixcodec"


def build(manifests_dir: str) -> pd.DataFrame:
    """The G.711 and AMR-NB halves of the deployed training manifest, stacked."""
    mdir = Path(manifests_dir)
    g711 = pd.read_csv(mdir / f"{G711}.csv", dtype={"speaker": str})
    amr = pd.read_csv(mdir / f"{AMR}.csv", dtype={"speaker": str})

    def names(frame: pd.DataFrame) -> list[str]:  # utt_id is empty for some clips; names never are
        return sorted(frame["filepath"].map(lambda p: Path(str(p)).name))

    if names(g711) != names(amr):
        raise ValueError("the G.711 and AMR-NB halves hold different clips")
    if g711["condition"].nunique() != 1 or amr["condition"].nunique() != 1:
        raise ValueError("each half must carry one channel condition")
    if g711["condition"].iloc[0] == amr["condition"].iloc[0]:
        raise ValueError("both halves carry the same channel condition")
    out = pd.concat([g711, amr[g711.columns]], ignore_index=True)
    if out["filepath"].duplicated().any():
        raise ValueError(
            "a clip path appears twice: the AMR-NB copies must sit in their own folder"
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the two-codec training manifest")
    parser.add_argument("--manifests", default="data/manifests")
    args = parser.parse_args()
    out = build(args.manifests)
    out.to_csv(Path(args.manifests) / f"{OUT}.csv", index=False)
    print(f"{OUT}: {len(out)} clips, conditions {out['condition'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
