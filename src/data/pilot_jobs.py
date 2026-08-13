"""Assemble the W3-T5 XTTS-v2 pilot: 20 clone jobs across a script/language matrix.

Week 3 (W3-T5, owner SK). The pilot answers two questions before ~4,000 clips are
generated in Week 4:

1. **Which script?** Hinglish is written in Devanagari, in Latin, and in a mix of
   both. XTTS-v2 may handle them very differently, and picking wrong wastes the
   whole Week-4 run.
2. **Which language tag?** XTTS takes a language code. ``hi`` and ``en`` produce
   different phonetisations of the same code-mixed sentence.

So the pilot is a 4-cell matrix, five clips each. The transcripts are **real MUCS
utterances** chosen for their script composition rather than machine-transliterated
text -- a transliterator would introduce its own errors and we would be testing
the transliterator instead of XTTS.

Two hard guards, both tested:

- reference speakers come from the **train pool only**, read from the frozen
  ``speaker_pools.csv``. A reference drawn from the adaptation or eval pool would
  put a cloned voice on the wrong side of the split and void the gap matrix.
- the job table records the pool with every row, so the firewall is auditable
  after the fact rather than trusted at generation time.

Generation itself does not run here: Coqui TTS has no distribution for Python
3.13 and this machine has no CUDA. The output of this module is a self-contained
pack (reference wavs + ``generation_jobs.csv``) to carry to a Kaggle GPU session.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

#: Devanagari code block, used to classify a transcript's script.
DEVANAGARI_START, DEVANAGARI_END = "\u0900", "\u097f"

#: The pilot matrix. Five clips per cell -> 20 jobs.
PILOT_CELLS = (
    ("deva_hi", "mostly_devanagari", "hi", "Devanagari script, Hindi tag - the expected default"),
    (
        "latn_hi",
        "mostly_latin",
        "hi",
        "romanised script, Hindi tag - how Hinglish is usually typed",
    ),
    (
        "latn_en",
        "mostly_latin",
        "en",
        "romanised script, English tag - does the en tag switch better?",
    ),
    ("mixed_hi", "mixed", "hi", "genuinely mixed script, Hindi tag - the realistic case"),
)

CLIPS_PER_CELL = 5

#: A cloning reference needs enough clean speech for XTTS to condition on.
MIN_REFERENCE_SECONDS = 6.0
#: Transcripts shorter than this produce clips too brief to judge.
MIN_TRANSCRIPT_CHARS = 40

JOB_COLUMNS = [
    "job_id",
    "cell",
    "speaker",
    "pool",
    "reference_wav",
    "transcript",
    "script_class",
    "language",
    "tool",
    "output_path",
    "seed",
]


class PilotError(AssertionError):
    """Raised when the pilot would violate the speaker-pool firewall."""


@dataclass(frozen=True)
class PilotConfig:
    """Pilot size and thresholds."""

    clips_per_cell: int = CLIPS_PER_CELL
    min_reference_seconds: float = MIN_REFERENCE_SECONDS
    min_transcript_chars: int = MIN_TRANSCRIPT_CHARS
    seed: int = 1234
    tool: str = "xtts_v2"


# --------------------------------------------------------------------------- #
# Script classification
# --------------------------------------------------------------------------- #
def devanagari_fraction(text: str) -> float:
    """Fraction of the alphabetic characters that are Devanagari."""
    letters = [c for c in str(text) if c.isalpha()]
    if not letters:
        return 0.0
    return sum(DEVANAGARI_START <= c <= DEVANAGARI_END for c in letters) / len(letters)


def script_class(text: str) -> str:
    """Bucket a transcript as ``mostly_latin`` / ``mixed`` / ``mostly_devanagari``."""
    fraction = devanagari_fraction(text)
    if fraction < 0.20:
        return "mostly_latin"
    if fraction > 0.70:
        return "mostly_devanagari"
    return "mixed"


# --------------------------------------------------------------------------- #
# Pool firewall
# --------------------------------------------------------------------------- #
def train_pool_speakers(pools: pd.DataFrame) -> set[str]:
    """Speakers permitted as cloning references for the seen-attack track."""
    return set(pools.loc[pools["pool"] == "train", "speaker"].astype(str))


def assert_train_pool_only(jobs: pd.DataFrame, pools: pd.DataFrame) -> None:
    """Raise :class:`PilotError` if any job clones a non-train-pool speaker."""
    allowed = train_pool_speakers(pools)
    used = set(jobs["speaker"].astype(str))
    intruders = used - allowed
    if intruders:
        lookup = pools.set_index(pools["speaker"].astype(str))["pool"].to_dict()
        detail = ", ".join(f"{s} (pool={lookup.get(s, 'unknown')})" for s in sorted(intruders))
        raise PilotError(f"pilot would clone speakers outside the train pool: {detail}")


# --------------------------------------------------------------------------- #
# Reference selection
# --------------------------------------------------------------------------- #
def choose_references(
    index: pd.DataFrame, pools: pd.DataFrame, config: PilotConfig | None = None
) -> pd.DataFrame:
    """Longest qualifying clip per train-pool speaker, best speakers first.

    One reference per speaker keeps the pilot's variable the *script*, not the
    reference audio.
    """
    cfg = config or PilotConfig()
    allowed = train_pool_speakers(pools)
    candidates = index[
        index["speaker"].astype(str).isin(allowed)
        & (index["duration_seconds"] >= cfg.min_reference_seconds)
    ]
    if candidates.empty:
        return candidates
    return (
        candidates.sort_values("duration_seconds", ascending=False)
        .drop_duplicates("speaker")
        .reset_index(drop=True)
    )


def choose_transcripts(
    index: pd.DataFrame, wanted: str, n: int, config: PilotConfig | None = None
) -> pd.DataFrame:
    """Pick ``n`` transcripts of a given script class, longest first.

    Longest-first because a two-word utterance cannot exercise a code-switch
    boundary, which is the thing the pilot is judging.
    """
    cfg = config or PilotConfig()
    frame = index.dropna(subset=["transcript"]).copy()
    frame["transcript"] = frame["transcript"].astype(str)
    frame = frame[frame["transcript"].str.len() >= cfg.min_transcript_chars]
    if frame.empty:
        return frame
    frame["script_class"] = frame["transcript"].map(script_class)
    matching = frame[frame["script_class"] == wanted]
    return matching.sort_values("transcript", key=lambda s: s.str.len(), ascending=False).head(n)


# --------------------------------------------------------------------------- #
# Job assembly
# --------------------------------------------------------------------------- #
def build_pilot_jobs(
    index: pd.DataFrame,
    pools: pd.DataFrame,
    out_dir: str = "data/generated/pilot",
    config: PilotConfig | None = None,
) -> pd.DataFrame:
    """Build the 4-cell pilot job table.

    **The same speakers appear in every cell.** The pilot's variable is the
    script and the language tag, so the reference voice has to be held constant
    across cells -- otherwise a difference between ``deva_hi`` and ``latn_en``
    could just as easily be a difference between two speakers, and the pilot
    would answer nothing. Five speakers x four cells = twenty jobs, and each
    speaker is heard once per cell.
    """
    cfg = config or PilotConfig()
    references = choose_references(index, pools, cfg).head(cfg.clips_per_cell)
    if references.empty:
        raise PilotError("no train-pool speaker has a long enough reference clip")

    pool_of = pools.set_index(pools["speaker"].astype(str))["pool"].to_dict()
    rows: list[dict[str, object]] = []
    job_id = 0

    for cell, wanted_script, language, _ in PILOT_CELLS:
        transcripts = choose_transcripts(index, wanted_script, cfg.clips_per_cell, cfg)
        for position in range(cfg.clips_per_cell):
            if position >= len(transcripts):
                break
            # Indexed by position within the cell, not by a running counter, so
            # speaker N is the same person in every cell.
            reference = references.iloc[position % len(references)]
            transcript_row = transcripts.iloc[position]
            speaker = str(reference["speaker"])
            job_id += 1
            rows.append(
                {
                    "job_id": job_id,
                    "cell": cell,
                    "speaker": speaker,
                    "pool": pool_of.get(speaker, "unknown"),
                    "reference_wav": f"refs/{speaker}.wav",
                    "transcript": str(transcript_row["transcript"]),
                    "script_class": wanted_script,
                    "language": language,
                    "tool": cfg.tool,
                    "output_path": f"{out_dir}/{cell}_{job_id:02d}_{speaker}.wav",
                    "seed": cfg.seed + job_id,
                }
            )

    jobs = pd.DataFrame(rows, columns=JOB_COLUMNS)
    assert_train_pool_only(jobs, pools)
    return jobs


def pilot_summary(jobs: pd.DataFrame) -> dict:
    """What the pilot will actually produce."""
    if jobs.empty:
        return {"jobs": 0}
    return {
        "jobs": int(len(jobs)),
        "cells": {c: int(n) for c, n in jobs["cell"].value_counts().sort_index().items()},
        "speakers": int(jobs["speaker"].nunique()),
        "pools_used": sorted(jobs["pool"].unique()),
        "languages": sorted(jobs["language"].unique()),
        "median_transcript_chars": int(jobs["transcript"].str.len().median()),
    }


def main() -> None:
    """CLI: ``python -m src.data.pilot_jobs --data-root C:/dfdata``."""
    import argparse
    import json
    import os
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Assemble the XTTS-v2 pilot jobs")
    parser.add_argument("--index", default="data/manifests/clip_index.csv")
    parser.add_argument("--pools", default="data/manifests/speaker_pools.csv")
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "data"))
    parser.add_argument("--pack-dir", default=None, help="where to write the pilot pack")
    parser.add_argument("--jobs-out", default="data/manifests/pilot_generation_jobs.csv")
    parser.add_argument(
        "--no-refs", action="store_true", help="write the job table without cutting reference wavs"
    )
    args = parser.parse_args()

    pack_dir = Path(args.pack_dir or (Path(args.data_root) / "generated" / "pilot"))
    index = pd.read_csv(args.index)
    pools = pd.read_csv(args.pools)

    jobs = build_pilot_jobs(index, pools, out_dir="outputs")
    Path(args.jobs_out).parent.mkdir(parents=True, exist_ok=True)
    jobs.to_csv(args.jobs_out, index=False)
    print(json.dumps(pilot_summary(jobs), indent=2))
    print(f"\nwrote {args.jobs_out}")

    if args.no_refs:
        return

    from src.data.corpora import load_clip
    from src.utils.audio_utils import TARGET_SR, save_wav

    references = choose_references(index, pools).set_index(
        choose_references(index, pools)["speaker"].astype(str)
    )
    refs_dir = pack_dir / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for speaker in sorted(set(jobs["speaker"].astype(str))):
        row = references.loc[speaker]
        audio, sample_rate = load_clip(row.to_dict(), target_sr=TARGET_SR)
        save_wav(str(refs_dir / f"{speaker}.wav"), audio, sample_rate)
        written += 1
    jobs.to_csv(pack_dir / "generation_jobs.csv", index=False)
    print(f"wrote {written} reference wav(s) -> {refs_dir}")
    print(f"pilot pack ready at {pack_dir} -- copy it to a Kaggle GPU session")


if __name__ == "__main__":
    main()
