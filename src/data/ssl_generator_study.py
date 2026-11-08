"""The generator study on SSL-AASIST: the three training manifests (P-044).

The generator study (P-041) asked, with the final recipe's adapter, whether training on one
more cloning tool catches a tool the detector never heard. P-044 asks the same of the
stronger detector, SSL-AASIST, on the recipe the voice check runs (the spontaneous recipe):

=========  ===========================================================================
control    the spontaneous recipe + the 31 ``gen_train`` speakers' genuine clips
fold_a     those genuine clips + their IndicF5 clones     (Chatterbox never heard)
fold_b     those genuine clips + their Chatterbox clones  (IndicF5 never heard)
fold_c     those genuine clips + both: voice check v2, already trained (reported)
=========  ===========================================================================

Exactly the generator study's extra rows, added to the spontaneous recipe as voice check v2
adds fold C's (``voice_check_v2.build``). Tortoise is in none of them (``config_guard``).
Every number is measured on speakers outside ``gen_train``; the rule is
``src/reporting/ssl_generator_study.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data import voice_check_v2 as v2

SYSTEMS = ("control", "fold_a", "fold_b")


def train_out(system: str) -> str:
    """Manifest stem of one system's training rows."""
    if system not in SYSTEMS:
        raise ValueError(f"unknown system {system!r}; expected one of {SYSTEMS}")
    return f"codemix_adapt_train_sslgs_{system}"


def main() -> None:
    parser = argparse.ArgumentParser(description="SSL-AASIST generator study manifests")
    parser.add_argument("--manifests", default="data/manifests")
    args = parser.parse_args()
    for system in SYSTEMS:
        out = v2.build(args.manifests, system=system)
        out.to_csv(Path(args.manifests) / f"{train_out(system)}.csv", index=False)
        print(
            f"{train_out(system)}: {len(out)} clips",
            out.groupby(["label", "tool"]).size().to_dict(),
        )


if __name__ == "__main__":
    main()
