"""The generator study: does training on more cloning tools catch a tool never seen?

Every unseen-tool result so far comes from detectors trained on XTTS-v2 (and RVC). The
field's open question for this setting is whether more generators in training carry over
to a generator the detector has never heard. IndicF5 (CM07) and Chatterbox (CM09) are the
only other tools there are clones of, and they cloned **test speakers** (15 of the eval
pool and 46 of ``eval_ext``). So, as for RVC (P-025), the speakers are split:

* the 61 cloned speakers are cut into two frozen halves, ``gen_train`` and ``gen_test``,
  alternated over sorted ids within each pool so every machine gets the same split
  (P-016); the split is written once to ``generator_pools.csv`` and its hash frozen here;
* a study model trains on the ``gen_train`` half only -- its IndicF5 and/or Chatterbox
  clones **and its genuine clips** (fakes never enter alone: without them every voice of
  that half would be fake in training);
* **every test number of a study model is reported on speakers outside ``gen_train``**,
  so no test speaker's voice was heard in training, in any set (``test_speakers``).

Four systems, each the final recipe (``codemix_adapt_train_final``) plus:

=========  =========================================================
control    the ``gen_train`` genuine clips only
fold_a     those genuine clips + IndicF5 clones   (Chatterbox unseen)
fold_b     those genuine clips + Chatterbox clones (IndicF5 unseen)
fold_c     those genuine clips + both              (the demo model)
=========  =========================================================

The control carries the same extra genuine speech, so a fold differs from it only by the
generator's clones. **Tortoise is never in any of them** (``config_guard``). The genuine clips come over both
codecs, as in the final recipe; the clones over G.711 only, because Chatterbox exists only
there, so folds A and B each add one tool's clips in the same condition (282 and 265). Team decision P-041; the hypotheses and decision rule are in
``docs/results/generator_study_v1.md``.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from src.data import final_recipe

POOLS_OUT = "generator_pools.csv"
#: sha256 of the canonical generator_pools.csv (CRLF, as frame_hash in speaker_pools)
FROZEN_SHA256: str | None = "8174f5446a97fab60d99f08404ac1f6b73b533ff38d7761d70bb7e2951806137"
CLONED = "score_cm09_channel20"  # Chatterbox's set: all 61 cloned speakers
GENUINE = ("score_cm07_channel20", "score_cm07_amr")
INDICF5 = ("score_cm07_channel20",)
CHATTERBOX = ("score_cm09_channel20",)
SYSTEMS = {
    "control": (),
    "fold_a": ("indicf5",),
    "fold_b": ("chatterbox",),
    "fold_c": ("indicf5", "chatterbox"),
}
POOL_FILES = ("speaker_pools.csv", "speaker_pools_ext.csv")


def _read(mdir: Path, stem: str) -> pd.DataFrame:
    return pd.read_csv(mdir / f"{stem}.csv", dtype={"speaker": str})


def split_speakers(manifests_dir: str = "data/manifests") -> pd.DataFrame:
    """The 61 cloned speakers, each with its frozen pool and its half."""
    mdir = Path(manifests_dir)
    cloned = set(_read(mdir, CLONED).query("label == 'spoof'")["speaker"])
    pools = pd.concat([_read(mdir, Path(f).stem) for f in POOL_FILES], ignore_index=True)
    pools = pools[pools["speaker"].isin(cloned)][["speaker", "pool"]].drop_duplicates()
    if set(pools["speaker"]) != cloned:
        raise ValueError(f"cloned speakers outside the frozen pools: {cloned - set(pools.speaker)}")
    rows = []
    for pool, group in sorted(pools.groupby("pool")):
        for i, speaker in enumerate(sorted(group["speaker"])):
            rows.append(
                {
                    "speaker": speaker,
                    "pool": pool,
                    "half": "gen_train" if i % 2 == 0 else "gen_test",
                }
            )
    return pd.DataFrame(rows).sort_values(["half", "pool", "speaker"]).reset_index(drop=True)


def frame_hash(frame: pd.DataFrame) -> str:
    """sha256 of the split, independent of the platform's line ending."""
    ordered = frame[["speaker", "pool", "half"]].sort_values(["half", "pool", "speaker"])
    canonical = ordered.to_csv(index=False, lineterminator="\r\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def train_speakers(pools_csv: str = f"data/manifests/{POOLS_OUT}") -> set[str]:
    """The ``gen_train`` half: the only test-pool speakers a study model may hear."""
    frame = pd.read_csv(pools_csv, dtype={"speaker": str})
    if FROZEN_SHA256 is not None and frame_hash(frame) != FROZEN_SHA256:
        raise ValueError(f"{pools_csv} differs from the frozen split")
    return set(frame.loc[frame["half"] == "gen_train", "speaker"])


def test_speakers(speakers: pd.Series, train: set[str]) -> pd.Series:
    """Mask of the rows a study model may be scored on: speakers it never heard."""
    return ~speakers.astype(str).isin(train)


def _rows(
    mdir: Path, stems: tuple[str, ...], label: str, tool: str, train: set[str]
) -> pd.DataFrame:
    frame = pd.concat([_read(mdir, s) for s in stems], ignore_index=True)
    keep = (
        frame["label"].eq(label) & frame["tool"].str.lower().eq(tool) & frame["speaker"].isin(train)
    )
    return frame[keep]


def build(system: str, manifests_dir: str = "data/manifests") -> pd.DataFrame:
    """Training manifest of one study system (the final recipe plus its extra rows)."""
    if system not in SYSTEMS:
        raise ValueError(f"unknown system {system!r}; expected one of {sorted(SYSTEMS)}")
    mdir = Path(manifests_dir)
    train = train_speakers(str(mdir / POOLS_OUT))
    parts = [_read(mdir, final_recipe.OUT), _rows(mdir, GENUINE, "bonafide", "none", train)]
    if "indicf5" in SYSTEMS[system]:
        parts.append(_rows(mdir, INDICF5, "spoof", "indicf5", train))
    if "chatterbox" in SYSTEMS[system]:
        parts.append(_rows(mdir, CHATTERBOX, "spoof", "chatterbox", train))
    out = pd.concat(parts, ignore_index=True)
    out["split"] = "train"
    if out["filepath"].duplicated().any():
        raise ValueError(f"{system}: duplicate clips in the training manifest")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generator study: speaker split and manifests")
    parser.add_argument("--manifests", default="data/manifests")
    parser.add_argument("--split", action="store_true", help="write generator_pools.csv first")
    args = parser.parse_args()
    mdir = Path(args.manifests)
    if args.split:
        pools = split_speakers(args.manifests)
        pools.to_csv(mdir / POOLS_OUT, index=False)
        print(
            f"{POOLS_OUT}: {pools.groupby(['half', 'pool']).size().to_dict()} sha256 {frame_hash(pools)}"
        )
    for system in SYSTEMS:
        out = build(system, args.manifests)
        stem = f"genstudy_train_{system}"
        out.to_csv(mdir / f"{stem}.csv", index=False)
        extra = out[out["speaker"].isin(train_speakers(str(mdir / POOLS_OUT)))]
        print(
            f"{stem}: {len(out)} clips; extra {extra.groupby(['label', 'tool', 'condition']).size().to_dict()}"
        )


if __name__ == "__main__":
    main()
