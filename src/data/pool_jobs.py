"""Assemble XTTS-v2 generation jobs for the **adaptation** or **eval** pool.

``scale_jobs.py`` builds the ~4,000-clip train-pool run (the S3 native-training
attack). This module is the same recipe -- reference selection, deterministic
cross-pairing stride, the synthesisability filters -- pointed at the other two
pools instead, because Stage-3 LoRA adaptation and its held-out evaluation need
their own code-mixed spoof clips, and generating them from train-pool speakers
would put an evaluation voice on the training side of the S3 firewall.

**Never accepts ``pool="train"``.** That run already exists, is already tested,
and already has 4,434 clips downstream of it (the gap matrix); this module
exists so nobody re-derives it by hand with the wrong pool and quietly breaks
the disjointness the whole gap-closure result depends on.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd

import pandas as pd

from src.data.pilot_jobs import (
    JOB_COLUMNS,
    MIN_TRANSCRIPT_CHARS,
    PilotError,
    assert_pool_only,
    choose_references,
    is_synthesisable,
    pool_speakers,
    script_class,
    spoken_text,
)

#: Speaker-count-proportional to the train-pool run (4,000 clips / 25 speakers =
#: 160/speaker). Smaller pools get the same per-speaker density rather than a
#: fixed count, so a 10-speaker adaptation pool and a 15-speaker eval pool are
#: each generated at the same rate the train pool was.
CLIPS_PER_SPEAKER = 160

ALLOWED_POOLS = ("adaptation", "eval")


@dataclass(frozen=True)
class PoolJobConfig:
    """Size and shape of a pool's generation run."""

    pool: str = "adaptation"
    n_target: int | None = None  # None -> len(speakers) * CLIPS_PER_SPEAKER
    language: str = "hi"
    romanise: bool = True
    tool: str = "xtts_v2"
    seed: int = 7000
    min_reference_seconds: float = 6.0
    min_transcript_chars: int = MIN_TRANSCRIPT_CHARS

    def __post_init__(self) -> None:
        if self.pool not in ALLOWED_POOLS:
            raise PilotError(
                f"pool_jobs only builds {ALLOWED_POOLS}; got {self.pool!r} "
                "-- the train-pool run is scale_jobs.py, not this module"
            )


def usable_transcripts(
    index: pd.DataFrame, pools: pd.DataFrame, config: PoolJobConfig
) -> pd.DataFrame:
    """Unique transcripts from ``config.pool`` speakers that XTTS can actually say."""
    allowed = pool_speakers(pools, config.pool)
    frame = index.dropna(subset=["transcript"]).copy()
    frame = frame[frame["speaker"].astype(str).isin(allowed)]
    if frame.empty:
        return frame.assign(spoken_text=[], script_source=[])

    frame["transcript"] = frame["transcript"].astype(str)
    frame["spoken_text"] = frame["transcript"].map(lambda t: spoken_text(t, config.romanise))
    lengths = frame["spoken_text"].str.len()
    frame = frame[lengths >= config.min_transcript_chars]
    frame = frame[frame["spoken_text"].map(lambda t: is_synthesisable(t, config.language))]
    if frame.empty:
        return frame

    frame["script_source"] = frame["transcript"].map(script_class)
    # utt_id is unique, so this ordering is identical on any machine (P-016).
    return (
        frame.sort_values("utt_id", kind="stable")
        .drop_duplicates("spoken_text")
        .reset_index(drop=True)
    )


def build_pool_jobs(
    index: pd.DataFrame,
    pools: pd.DataFrame,
    out_dir: str = "outputs",
    config: PoolJobConfig | None = None,
) -> pd.DataFrame:
    """Build the generation table for one pool. See :func:`usable_transcripts`."""
    cfg = config or PoolJobConfig()
    references = choose_references(index, pools, pool=cfg.pool)
    if references.empty:
        raise PilotError(f"no {cfg.pool}-pool speaker has a long enough reference clip")

    texts = usable_transcripts(index, pools, cfg)
    if texts.empty:
        raise PilotError(f"no {cfg.pool}-pool transcript survives the length/digit filters")

    speakers = sorted(set(references["speaker"].astype(str)))
    n_speakers = len(speakers)
    n_target = cfg.n_target if cfg.n_target is not None else n_speakers * CLIPS_PER_SPEAKER

    # Same coprime-stride reasoning as scale_jobs.py: the stride visits
    # lcm(S, T) distinct pairs, not S x T, so texts are trimmed until the two
    # counts share no factor and the full S x T space is reachable.
    n_texts = len(texts)
    while n_texts > 1 and gcd(n_speakers, n_texts) != 1:
        n_texts -= 1
    texts = texts.head(n_texts)

    capacity = n_speakers * n_texts
    if n_target > capacity:
        raise PilotError(
            f"asked for {n_target} clips but only {capacity} unique (speaker, "
            f"transcript) pairs exist in the {cfg.pool} pool "
            f"({n_speakers} speakers x {n_texts} texts)"
        )

    pool_of = pools.set_index(pools["speaker"].astype(str))["pool"].to_dict()
    rows: list[dict[str, object]] = []
    for i in range(n_target):
        speaker = speakers[i % n_speakers]
        text_row = texts.iloc[i % n_texts]
        job_id = i + 1
        rows.append(
            {
                "job_id": job_id,
                "cell": f"{cfg.pool}_{cfg.language}",
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
    assert_pool_job_invariants(jobs, pools, cfg.pool)
    return jobs


def assert_pool_job_invariants(jobs: pd.DataFrame, pools: pd.DataFrame, pool: str) -> None:
    """The same three checks scale_jobs.py runs, against the given pool."""
    assert_pool_only(jobs, pools, pool)

    pairs = jobs[["speaker", "transcript"]].astype(str)
    if pairs.duplicated().any():
        n = int(pairs.duplicated().sum())
        raise PilotError(f"{n} duplicate (speaker, transcript) pair(s) in the job table")

    over = [
        (row.language, len(row.transcript))
        for row in jobs.itertuples(index=False)
        if not is_synthesisable(str(row.transcript), str(row.language))
    ]
    if over:
        raise PilotError(f"{len(over)} transcript(s) are not synthesisable under their tag")


def pool_job_summary(jobs: pd.DataFrame) -> dict:
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
        "pools_used": sorted(jobs["pool"].unique()),
        "languages": sorted(jobs["language"].unique()),
    }


def main() -> None:
    """CLI: ``python -m src.data.pool_jobs --pool adaptation --data-root C:/dfdata``."""
    import argparse
    import json
    import os
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Assemble a Stage-3 generation job table for the adaptation or eval pool"
    )
    parser.add_argument("--pool", required=True, choices=ALLOWED_POOLS)
    parser.add_argument("--index", default="data/manifests/clip_index.csv")
    parser.add_argument("--pools", default="data/manifests/speaker_pools.csv")
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "data"))
    parser.add_argument("--pack-dir", default=None)
    parser.add_argument("--jobs-out", default=None)
    parser.add_argument("--n", type=int, default=None, help="default: speakers x 160")
    parser.add_argument("--language", default="hi")
    parser.add_argument(
        "--tool",
        default="xtts_v2",
        help="generator id recorded on every row; 'tortoise' builds the CM04 held-out set",
    )
    parser.add_argument("--no-romanise", action="store_true")
    parser.add_argument("--no-refs", action="store_true", help="job table only, no wav cutting")
    args = parser.parse_args()

    pack_dir = Path(args.pack_dir or (Path(args.data_root) / "generated" / f"xtts_v2_{args.pool}"))
    jobs_out = args.jobs_out or (
        "data/manifests/heldout_generation_jobs.csv"
        if args.tool == "tortoise"
        else f"data/manifests/{args.pool}_generation_jobs.csv"
    )
    index = pd.read_csv(args.index)
    pools = pd.read_csv(args.pools)

    config = PoolJobConfig(
        pool=args.pool,
        n_target=args.n,
        language=args.language,
        romanise=not args.no_romanise,
        tool=args.tool,
    )
    jobs = build_pool_jobs(index, pools, out_dir="outputs", config=config)
    Path(jobs_out).parent.mkdir(parents=True, exist_ok=True)
    jobs.to_csv(jobs_out, index=False)
    print(json.dumps(pool_job_summary(jobs), indent=2, ensure_ascii=False))
    print(f"\nwrote {jobs_out}")

    if args.no_refs:
        return

    from src.data.corpora import load_clip
    from src.utils.audio_utils import TARGET_SR, save_wav

    references = choose_references(index, pools, pool=args.pool)
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
