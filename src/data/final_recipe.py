"""The final training recipe: many voices and two telephone codecs together.

Two week-13 runs each fixed one weakness of the deployed S2:

* training on 248 genuine speakers instead of 21 (``extended_training``) improved every
  seed on new speakers, both unseen tools and false alarms;
* training on G.711 and AMR-NB together (``codec_mix``) removed the loss on a real
  mobile codec.

This manifest combines them: the 248-speaker training manifest over the simulated G.711
line, plus every one of its clips again through AMR-NB at 12.2 kbit/s (16,036 clips).
Nothing else changes: same dev set, start, schedule. The AMR-NB half is checked to be
exactly the G.711 half's clips, in a different condition, in their own folder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data import codec_mix, extended_training

AMR_PARTS = ("codemix_adapt_train_rvc_amr", "adapt_ext_genuine_amr", "cm08_amr")
OUT = "codemix_adapt_train_final"


def build(manifests_dir: str) -> pd.DataFrame:
    """G.711 248-speaker manifest stacked with its AMR-NB copy."""
    mdir = Path(manifests_dir)
    g711 = pd.read_csv(mdir / f"{extended_training.OUT}.csv", dtype={"speaker": str})
    amr = pd.concat(
        [pd.read_csv(mdir / f"{stem}.csv", dtype={"speaker": str}) for stem in AMR_PARTS],
        ignore_index=True,
    )
    amr["split"] = "train"
    return codec_mix.stack(g711, amr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the final training manifest")
    parser.add_argument("--manifests", default="data/manifests")
    args = parser.parse_args()
    out = build(args.manifests)
    out.to_csv(Path(args.manifests) / f"{OUT}.csv", index=False)
    print(
        f"{OUT}: {len(out)} clips, {out['speaker'].nunique()} speakers, "
        f"conditions {out['condition'].value_counts().to_dict()}"
    )


if __name__ == "__main__":
    main()
