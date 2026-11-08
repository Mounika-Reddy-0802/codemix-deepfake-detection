"""Voice check v2: the voice check's detector recipe plus the newest tools' clones (P-042).

The voice check runs SSL-AASIST. Its strongest recipe so far, the spontaneous recipe
(``spont_recipe``), still lets most clean IndicF5 and Chatterbox clones through
(``docs/voice_check.md``). v2 trains the same model on the same data plus the generator
study's training half (P-041): the IndicF5 and Chatterbox clones of the 31 ``gen_train``
speakers, over G.711, and those speakers' genuine clips over both codecs, so their voices are
not "always fake" in training. Exactly the rows the study's fold C adds to the final recipe,
added here to the spontaneous recipe instead.

**In this model IndicF5 and Chatterbox are seen tools.** Every voice-check number of v2 is
measured on speakers outside ``gen_train`` (``generator_study.test_speakers``), and the
paper's unseen-tool numbers, from detectors that never heard these tools, are unchanged.
Tortoise stays out of training (``config_guard``), so it remains an unseen tool here too.
The dev set is the spontaneous recipe's, unchanged. The adoption rule is in
``src/reporting/voice_check_v2.py`` and ``docs/results/voice_check_v2.md``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data import generator_study as gs
from src.data import spont_recipe

TRAIN_OUT = "codemix_adapt_train_vcv2"
DEV = spont_recipe.DEV_OUT  # unchanged: the 506 dev clips plus the 20 iv_dev speakers


def build(manifests_dir: str = "data/manifests", system: str = "fold_c") -> pd.DataFrame:
    """The spontaneous recipe's training rows plus one study system's extra rows.

    ``fold_c`` (both tools) is voice check v2; ``control``, ``fold_a`` and ``fold_b`` are the
    same recipe for the generator study on SSL-AASIST (``src/data/ssl_generator_study``).
    """
    mdir = Path(manifests_dir)
    base = pd.read_csv(mdir / f"{spont_recipe.TRAIN_OUT}.csv", dtype={"speaker": str})
    study = gs.build(system, manifests_dir)
    extra = study[study["speaker"].isin(gs.train_speakers(str(mdir / gs.POOLS_OUT)))]
    out = pd.concat([base, extra.reindex(columns=base.columns)], ignore_index=True)
    out["split"] = "train"
    if out["filepath"].duplicated().any():
        raise ValueError(f"duplicate clips in the {system} training manifest")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice check v2 training manifest")
    parser.add_argument("--manifests", default="data/manifests")
    args = parser.parse_args()
    out = build(args.manifests)
    out.to_csv(Path(args.manifests) / f"{TRAIN_OUT}.csv", index=False)
    print(f"{TRAIN_OUT}: {len(out)} clips, {out['speaker'].nunique()} speakers")
    print(out.groupby(["label", "tool"]).size().to_dict())


if __name__ == "__main__":
    main()
