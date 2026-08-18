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

This module only assembles the pack (reference wavs + ``generation_jobs.csv``);
``src.data.spoof_generation`` synthesises from it. The pack is self-contained and
path-portable, so the same one runs on a Colab GPU or, slowly, on a CPU laptop --
the `coqui-tts` fork does ship for Python 3.13, contrary to the original note here.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.transliteration import to_roman

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

#: XTTS-v2's per-language character limit. Text longer than this is **silently
#: truncated** -- the model warns on stderr and synthesises the opening fragment.
#: The first pilot run hit this on 19 of 20 jobs (median 278 chars against a
#: 150-char Hindi limit) because transcripts were chosen longest-first. Raters
#: would then have been scoring truncation instead of code-switch quality, which
#: is the one thing the pilot exists to measure.
XTTS_CHAR_LIMITS = {"en": 250, "hi": 150}
#: Used for any language not listed above; XTTS's own default for unknown codes.
DEFAULT_CHAR_LIMIT = 250

#: Language tags for which XTTS's text cleaner cannot expand digits into words.
#: It calls ``num2words(n, lang=...)`` unconditionally, and num2words has no Hindi
#: implementation, so a transcript containing a digit raises ``NotImplementedError``
#: mid-synthesis. MUCS is NPTEL lecture speech and is full of numbers, so this is
#: common rather than exotic -- it killed the second pilot run at clip 7.
NUMERALS_UNSUPPORTED = {"hi"}


def char_limit(language: str) -> int:
    """XTTS-v2's maximum synthesisable transcript length for a language tag."""
    return XTTS_CHAR_LIMITS.get(str(language).lower(), DEFAULT_CHAR_LIMIT)


def is_synthesisable(text: str, language: str) -> bool:
    """Whether XTTS can actually say this transcript under this language tag.

    Length and digits are the two ways a real MUCS utterance fails. Both are
    checked here rather than discovered one crash at a time during a 4,000-clip
    Week-4 run.
    """
    text = str(text)
    if len(text) > char_limit(language):
        return False
    return not (str(language).lower() in NUMERALS_UNSUPPORTED and any(c.isdigit() for c in text))


JOB_COLUMNS = [
    "job_id",
    "cell",
    "speaker",
    "pool",
    "reference_wav",
    "transcript",  # what XTTS is handed -- romanised when the pack is romanised
    "transcript_source",  # what MUCS/HiACC actually wrote, kept for provenance
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
    #: Transliterate Devanagari to Latin before handing text to XTTS (team
    #: decision, 17 Aug 2026: the corpus is written in one script, and it is
    #: Latin). The source text is preserved in ``transcript_source``.
    romanise: bool = False


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
    # Duration alone does not order these uniquely: MUCS segment durations are
    # whole seconds, so ties are the rule rather than the exception (177570 and
    # 850754 are both exactly 20.0 s). With only `duration_seconds` as the key the
    # winner fell out of `clip_index.csv` row order, which follows the filesystem
    # walk and therefore differs per machine -- the CPU laptop and the GPU laptop
    # built the same pilot with speakers 4 and 5 swapped in every cell, so
    # `deva_hi_04` was a different person on each and the rating sheet would have
    # mis-attributed every score. `utt_id` is unique, so appending it makes both
    # the speaker ordering AND the chosen clip per speaker reproducible anywhere.
    return (
        candidates.sort_values(
            ["duration_seconds", "speaker", "utt_id"],
            ascending=[False, True, True],
            kind="stable",
        )
        .drop_duplicates("speaker")
        .reset_index(drop=True)
    )


def pilot_char_limit() -> int:
    """The transcript length every pilot cell must share.

    The tightest limit across the tags the pilot uses. Letting each cell run to
    its own limit would give ``latn_en`` 250 characters and the ``hi`` cells 150,
    so ``latn_en`` vs ``latn_hi`` would differ in tag *and* in utterance length --
    and a rating difference between them could be either. The pilot's one job is
    to isolate script and tag, so length is held constant instead.
    """
    return min(char_limit(language) for _, _, language, _ in PILOT_CELLS)


def spoken_text(text: str, romanise: bool) -> str:
    """The string XTTS is actually handed for a transcript.

    One function so that job assembly and the length/sayability filters can never
    disagree about what the model receives -- the bug class P-010 documents.
    """
    return to_roman(text) if romanise else str(text)


def choose_transcripts(
    index: pd.DataFrame,
    wanted: str,
    n: int,
    config: PilotConfig | None = None,
    language: str = "hi",
    max_chars: int | None = None,
    romanise: bool = False,
) -> pd.DataFrame:
    """Pick ``n`` transcripts of a given script class: the longest XTTS can say.

    Longest-first, because a two-word utterance cannot exercise a code-switch
    boundary and that boundary is the thing the pilot judges. But restricted to
    what :func:`is_synthesisable` allows: past the character limit XTTS truncates
    and synthesises only the opening fragment, and a digit under a ``hi`` tag
    crashes it outright. Unfiltered, "longest" quietly becomes "most truncated".

    Two different strings are in play once ``romanise`` is on, and they are
    measured for different things:

    - **cell membership** uses the SOURCE script, because that is what makes a
      sentence Hindi-heavy or English-heavy. After transliteration every
      transcript is Latin, so classifying the output would collapse all cells
      into one and the pilot would compare nothing.
    - **length and sayability** use the TRANSLITERATED text, because that is what
      XTTS receives. Romanisation inflates length by ~13% on average and up to
      58%: measured on this corpus, 14 of the 20 pilot transcripts cross the
      150-character Hindi limit once romanised. Filtering on the Devanagari
      length would hand XTTS text it silently truncates -- exactly the failure
      P-010 records, re-introduced through the back door.
    """
    cfg = config or PilotConfig()
    cap = char_limit(language) if max_chars is None else min(max_chars, char_limit(language))
    frame = index.dropna(subset=["transcript"]).copy()
    frame["transcript"] = frame["transcript"].astype(str)
    frame["script_class"] = frame["transcript"].map(script_class)  # source script
    frame["spoken_text"] = frame["transcript"].map(lambda t: spoken_text(t, romanise))
    lengths = frame["spoken_text"].str.len()
    frame = frame[(lengths >= cfg.min_transcript_chars) & (lengths <= cap)]
    frame = frame[frame["spoken_text"].map(lambda t: is_synthesisable(t, language))]
    if frame.empty:
        return frame
    matching = frame[frame["script_class"] == wanted].copy()
    # Same reproducibility trap as choose_references: thousands of MUCS
    # transcripts share a character count, so ordering on length alone let the
    # filesystem walk decide which ones the pilot used. `utt_id` is unique, so it
    # settles every tie identically on any machine.
    matching["_length"] = matching["spoken_text"].str.len()
    return (
        matching.sort_values(["_length", "utt_id"], ascending=[False, True], kind="stable")
        .drop(columns="_length")
        .head(n)
    )


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

    shared_limit = pilot_char_limit()  # same utterance length in every cell
    for cell, wanted_script, language, _ in PILOT_CELLS:
        transcripts = choose_transcripts(
            index,
            wanted_script,
            cfg.clips_per_cell,
            cfg,
            language,
            max_chars=shared_limit,
            romanise=cfg.romanise,
        )
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
                    "transcript": str(transcript_row["spoken_text"]),
                    "transcript_source": str(transcript_row["transcript"]),
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


def load_pilot_jobs(jobs_csv: str, pack_dir: str, out_dir: str = "outputs") -> list:
    """Turn the pilot job table into :class:`CloneJob` objects for the generator.

    Kept here rather than in the Colab notebook so the notebook holds no pipeline
    logic (CLAUDE.md section 9) and the path rewriting is unit-tested. Reference
    paths in the CSV are relative to the pack (``refs/<speaker>.wav``) so the pack
    can be moved between machines without editing anything.
    """
    from pathlib import Path

    from src.data.spoof_generation import CloneJob

    frame = pd.read_csv(jobs_csv)
    pack = Path(pack_dir)
    jobs = []
    for row in frame.itertuples(index=False):
        jobs.append(
            CloneJob(
                speaker=str(row.speaker),
                reference_wav=str(pack / str(row.reference_wav)),
                transcript=str(row.transcript),
                output_path=str(Path(out_dir) / Path(str(row.output_path)).name),
                pool=str(row.pool),
                language=str(row.language),
                tool=str(row.tool),
                seed=int(row.seed),
            )
        )
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
    parser.add_argument(
        "--romanise",
        action="store_true",
        help="transliterate Devanagari to Latin before synthesis (Hinglish transcripts)",
    )
    args = parser.parse_args()

    pack_dir = Path(args.pack_dir or (Path(args.data_root) / "generated" / "pilot"))
    index = pd.read_csv(args.index)
    pools = pd.read_csv(args.pools)

    config = PilotConfig(romanise=args.romanise)
    jobs = build_pilot_jobs(index, pools, out_dir="outputs", config=config)
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
