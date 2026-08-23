"""Assemble the W4-T1 generation run: ~4,000 XTTS-v2 clones from the train pool.

The pilot answered *how* to generate (P-014: romanised Hinglish; P-015: the
character limit applies to the romanised text). This module answers *what* to
generate, at scale.

**Cross-pairing is the point.** The 25 train-pool speakers have 1,404 unique
synthesisable transcripts between them, which is fewer than the 4,000 clips the
plan wants. Rather than repeat whole utterances verbatim, each transcript is
spoken by several different voices — 1,404 texts x 25 speakers = 35,100 possible
(speaker, text) pairs, so 4,000 is a *choice*, not a ceiling. A detector that
learned "this sentence = spoof" would be learning the transcript, not the
artefact, so no (speaker, text) pair is ever generated twice.

**The pairing is a stride, not a shuffle.** Job ``i`` takes
``speakers[i % S]`` and ``texts[i % T]``. Because ``gcd(25, 1404) = 1``, this
gives every speaker exactly ``n / S`` jobs, spreads each speaker's texts across
the whole corpus rather than clustering them at the front, and cannot repeat a
pair until ``lcm(S, T) = 35,100`` jobs — far beyond any run size we plan. It is
also completely deterministic, so two machines build the same table.

The speaker-pool firewall is the same one the pilot uses: **train pool only**.
A reference drawn from the adaptation or eval pool would put a clone of an
evaluation voice into training data and void the gap matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd

import pandas as pd

from src.data.pilot_jobs import (
    JOB_COLUMNS,
    MIN_TRANSCRIPT_CHARS,
    PilotError,
    choose_references,
    is_synthesisable,
    script_class,
    spoken_text,
    train_pool_speakers,
)

#: The Week-4 target from the v2 plan (W4-T1).
DEFAULT_TARGET = 4000


@dataclass(frozen=True)
class ScaleConfig:
    """Size and shape of the generation run."""

    n_target: int = DEFAULT_TARGET
    language: str = "hi"
    romanise: bool = True
    tool: str = "xtts_v2"
    seed: int = 4000
    min_reference_seconds: float = 6.0
    min_transcript_chars: int = MIN_TRANSCRIPT_CHARS


def usable_transcripts(
    index: pd.DataFrame, pools: pd.DataFrame, config: ScaleConfig | None = None
) -> pd.DataFrame:
    """Unique transcripts from train-pool speakers that XTTS can actually say.

    Deduplicated on the **spoken** text: the same sentence appearing under two
    utterance ids is one transcript, and generating both would be a duplicate
    dressed up as two clips.
    """
    cfg = config or ScaleConfig()
    allowed = train_pool_speakers(pools)
    frame = index.dropna(subset=["transcript"]).copy()
    frame = frame[frame["speaker"].astype(str).isin(allowed)]
    if frame.empty:
        return frame.assign(spoken_text=[], script_source=[])

    frame["transcript"] = frame["transcript"].astype(str)
    frame["spoken_text"] = frame["transcript"].map(lambda t: spoken_text(t, cfg.romanise))
    lengths = frame["spoken_text"].str.len()
    frame = frame[lengths >= cfg.min_transcript_chars]
    frame = frame[frame["spoken_text"].map(lambda t: is_synthesisable(t, cfg.language))]
    if frame.empty:
        return frame

    frame["script_source"] = frame["transcript"].map(script_class)
    # utt_id is unique, so this ordering is identical on any machine (P-016).
    return (
        frame.sort_values("utt_id", kind="stable")
        .drop_duplicates("spoken_text")
        .reset_index(drop=True)
    )


def build_scale_jobs(
    index: pd.DataFrame,
    pools: pd.DataFrame,
    out_dir: str = "outputs",
    config: ScaleConfig | None = None,
) -> pd.DataFrame:
    """Build the ~4,000-row generation table, train pool only."""
    cfg = config or ScaleConfig()
    references = choose_references(index, pools)
    if references.empty:
        raise PilotError("no train-pool speaker has a long enough reference clip")

    texts = usable_transcripts(index, pools, cfg)
    if texts.empty:
        raise PilotError("no train-pool transcript survives the length/digit filters")

    speakers = sorted(set(references["speaker"].astype(str)))
    n_speakers = len(speakers)

    # The stride visits lcm(S, T) distinct pairs, NOT S x T. When S and T share a
    # factor it wraps early: 5 speakers over 40 texts repeats after 40 jobs, not
    # 200. Dropping the tail of the transcript list until the two are coprime
    # costs at most a handful of transcripts and makes the whole S x T space
    # reachable, so the capacity figure below is the truth rather than an
    # optimistic bound.
    n_texts = len(texts)
    while n_texts > 1 and gcd(n_speakers, n_texts) != 1:
        n_texts -= 1
    texts = texts.head(n_texts)

    capacity = n_speakers * n_texts
    if cfg.n_target > capacity:
        raise PilotError(
            f"asked for {cfg.n_target} clips but only {capacity} unique "
            f"(speaker, transcript) pairs exist ({n_speakers} speakers x {n_texts} texts)"
        )

    pool_of = pools.set_index(pools["speaker"].astype(str))["pool"].to_dict()
    rows: list[dict[str, object]] = []
    for i in range(cfg.n_target):
        speaker = speakers[i % n_speakers]
        text_row = texts.iloc[i % n_texts]
        job_id = i + 1
        rows.append(
            {
                "job_id": job_id,
                "cell": f"scale_{cfg.language}",
                "speaker": speaker,
                "pool": pool_of.get(speaker, "unknown"),
                "reference_wav": f"refs/{speaker}.wav",
                "transcript": str(text_row["spoken_text"]),
                "transcript_source": str(text_row["transcript"]),
                "script_class": str(text_row["script_source"]),
                "language": cfg.language,
                "tool": cfg.tool,
                "output_path": f"{out_dir}/{cfg.tool}_{speaker}_{job_id:05d}.wav",
                "seed": cfg.seed + job_id,
            }
        )

    jobs = pd.DataFrame(rows, columns=JOB_COLUMNS)
    assert_scale_invariants(jobs, pools)
    return jobs


def assert_scale_invariants(jobs: pd.DataFrame, pools: pd.DataFrame) -> None:
    """The three things that make this corpus usable. Checked before generating."""
    # 1. Pool firewall.
    allowed = train_pool_speakers(pools)
    intruders = set(jobs["speaker"].astype(str)) - allowed
    if intruders:
        lookup = pools.set_index(pools["speaker"].astype(str))["pool"].to_dict()
        detail = ", ".join(f"{s} (pool={lookup.get(s, 'unknown')})" for s in sorted(intruders))
        raise PilotError(f"scale run would clone speakers outside the train pool: {detail}")

    # 2. No repeated (speaker, transcript) pair -- a duplicate clip teaches the
    #    detector the sentence rather than the artefact.
    pairs = jobs[["speaker", "transcript"]].astype(str)
    if pairs.duplicated().any():
        n = int(pairs.duplicated().sum())
        raise PilotError(f"{n} duplicate (speaker, transcript) pair(s) in the job table")

    # 3. Nothing XTTS will silently truncate (P-010/P-015).
    over = [
        (row.language, len(row.transcript))
        for row in jobs.itertuples(index=False)
        if not is_synthesisable(str(row.transcript), str(row.language))
    ]
    if over:
        raise PilotError(f"{len(over)} transcript(s) are not synthesisable under their tag")


def scale_summary(jobs: pd.DataFrame) -> dict:
    """What the run will produce, for the log and the datasheet."""
    if jobs.empty:
        return {"jobs": 0}
    per_speaker = jobs["speaker"].value_counts()
    return {
        "jobs": int(len(jobs)),
        "speakers": int(jobs["speaker"].nunique()),
        "per_speaker_min": int(per_speaker.min()),
        "per_speaker_max": int(per_speaker.max()),
        "unique_transcripts": int(jobs["transcript"].nunique()),
        "mean_uses_per_transcript": round(len(jobs) / jobs["transcript"].nunique(), 2),
        "pools_used": sorted(jobs["pool"].unique()),
        "languages": sorted(jobs["language"].unique()),
        "median_transcript_chars": int(jobs["transcript"].str.len().median()),
        "source_script_mix": {k: int(v) for k, v in jobs["script_class"].value_counts().items()},
    }


def main() -> None:
    """CLI: ``python -m src.data.scale_jobs --data-root C:/dfdata --n 4000``."""
    import argparse
    import json
    import os
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Assemble the W4-T1 generation job table")
    parser.add_argument("--index", default="data/manifests/clip_index.csv")
    parser.add_argument("--pools", default="data/manifests/speaker_pools.csv")
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "data"))
    parser.add_argument("--pack-dir", default=None)
    parser.add_argument("--jobs-out", default="data/manifests/scale_generation_jobs.csv")
    parser.add_argument("--n", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--language", default="hi")
    parser.add_argument("--no-romanise", action="store_true")
    parser.add_argument("--no-refs", action="store_true", help="job table only, no wav cutting")
    args = parser.parse_args()

    pack_dir = Path(args.pack_dir or (Path(args.data_root) / "generated" / "xtts_v2"))
    index = pd.read_csv(args.index)
    pools = pd.read_csv(args.pools)

    config = ScaleConfig(n_target=args.n, language=args.language, romanise=not args.no_romanise)
    jobs = build_scale_jobs(index, pools, out_dir="outputs", config=config)
    Path(args.jobs_out).parent.mkdir(parents=True, exist_ok=True)
    jobs.to_csv(args.jobs_out, index=False)
    print(json.dumps(scale_summary(jobs), indent=2, ensure_ascii=False))
    print(f"\nwrote {args.jobs_out}")

    if args.no_refs:
        return

    from src.data.corpora import load_clip
    from src.utils.audio_utils import TARGET_SR, save_wav

    references = choose_references(index, pools)
    references = references.set_index(references["speaker"].astype(str))
    refs_dir = pack_dir / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for speaker in sorted(set(jobs["speaker"].astype(str))):
        target = refs_dir / f"{speaker}.wav"
        if target.exists():
            written += 1
            continue
        audio, sample_rate = load_clip(references.loc[speaker].to_dict(), target_sr=TARGET_SR)
        save_wav(str(target), audio, sample_rate)
        written += 1
    jobs.to_csv(pack_dir / "generation_jobs.csv", index=False)
    print(f"{written} reference wav(s) ready -> {refs_dir}")
    print(f"pack ready at {pack_dir}")


if __name__ == "__main__":
    main()
