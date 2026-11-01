"""XTTS-v2 clone jobs for the IndicVoices speaker pools (plan task A2.3).

Spontaneous genuine speech in training needs clones of the **same** speakers, or the
detector could learn "spontaneous means genuine" (the speaking style as a shortcut). This
module builds the clone tables with the project's existing recipe
(``src.data.pool_jobs.build_pool_jobs``: one reference clip per speaker, the longest of at
least 6 s; transcripts from the same pool's speakers, romanised as in every earlier XTTS
run; a coprime stride over speakers and texts), 20 clones per speaker, one table per pool.

The clips are the ones ``src.data.indicvoices_download`` keeps, computed from metadata
alone (the frozen row groups and pools, the same selection and the same clip ids), so the
tables can be built before any audio is downloaded. Transcripts never cross pools: the
test pool's clones read only test-pool text.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.indicvoices_download import select_rows, utt_id
from src.data.indicvoices_pools import GROUPS_PATH, POOLS_PATH
from src.data.pool_jobs import PoolJobConfig, build_pool_jobs

CLIPS_PER_SPEAKER = 20
#: one seed block per pool (seed + job id), disjoint from each other and from the
#: earlier XTTS and IndicF5 runs (7000-9999)
SEEDS = {"iv_train": 20000, "iv_dev": 30000, "iv_test": 40000}
JOBS_PATH = "data/manifests/indicvoices_{pool}_xtts_jobs.csv"


def planned_clips(cache: str, groups_path: str = GROUPS_PATH) -> pd.DataFrame:
    """The clips the download keeps, as a clip index: speaker, utt_id, transcript, duration."""
    pools = pd.read_csv(POOLS_PATH)
    rows = []
    for g in pd.read_csv(groups_path).itertuples(index=False):
        meta = pd.read_parquet(Path(cache) / g.shard)
        part = meta.iloc[int(g.first_row) : int(g.first_row) + int(g.rows)].reset_index(drop=True)
        kept = select_rows(part, pools)
        for position, row in kept.iterrows():
            rows.append(
                {
                    "speaker": row["speaker"],
                    "pool": row["split"],
                    "utt_id": utt_id(str(row["speaker_id"]), g.shard, int(g.first_row) + position),
                    "transcript": str(row["text"]),
                    "duration_seconds": float(row["duration"]),
                }
            )
    return pd.DataFrame(rows)


def build_jobs(clips: pd.DataFrame, pools: pd.DataFrame, pool: str) -> pd.DataFrame:
    """The clone table of one pool, 20 jobs per speaker."""
    n_speakers = int((pools["pool"] == pool).sum())
    config = PoolJobConfig(pool=pool, n_target=n_speakers * CLIPS_PER_SPEAKER, seed=SEEDS[pool])
    return build_pool_jobs(clips, pools, out_dir="outputs", config=config)


def write_pack(clips: pd.DataFrame, clean_root: str, pack_dir: str) -> pd.DataFrame:
    """The generation pack: one folder per pool, as ``src.data.generate_cli`` reads it.

    ``<pack_dir>/xtts_<pool>/generation_jobs.csv`` and ``refs/<speaker>.wav``, one reference
    per speaker: the clip the job builder chose (``choose_references``: the longest of at
    least 6 s), copied from the downloaded clean clips. ``references.csv`` lists them.
    """
    import shutil

    from src.data.pilot_jobs import choose_references

    pools = pd.read_csv(POOLS_PATH)
    chosen = []
    for pool in SEEDS:
        folder = Path(pack_dir) / f"xtts_{pool}"
        (folder / "refs").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(JOBS_PATH.format(pool=pool), folder / "generation_jobs.csv")
        for row in choose_references(clips, pools, pool=pool).itertuples(index=False):
            src = Path(clean_root) / "clips" / f"{row.utt_id}.wav"
            shutil.copyfile(src, folder / "refs" / f"{row.speaker}.wav")
            chosen.append({"speaker": row.speaker, "pool": pool, "reference_utt_id": row.utt_id})
    table = pd.DataFrame(chosen)
    table.to_csv(Path(pack_dir) / "references.csv", index=False)
    return table


def main() -> None:
    """CLI: write one XTTS-v2 job table per IndicVoices pool (and the pack, with --pack)."""
    import argparse

    parser = argparse.ArgumentParser(description="XTTS-v2 jobs for the IndicVoices pools")
    parser.add_argument("--cache", default="data/indicvoices/meta")
    parser.add_argument("--pack", default=None, help="also write the generation pack here")
    parser.add_argument("--clean-root", default="D:/dfdata/indicvoices_bundle/clean")
    args = parser.parse_args()
    clips = planned_clips(args.cache)
    pools = pd.read_csv(POOLS_PATH)
    print(f"planned clips: {len(clips)} from {clips['speaker'].nunique()} speakers")
    for pool in SEEDS:
        jobs = build_jobs(clips, pools, pool)
        out = JOBS_PATH.format(pool=pool)
        jobs.to_csv(out, index=False)
        print(f"{pool}: {len(jobs)} jobs, {jobs['speaker'].nunique()} speakers -> {out}")
    if args.pack:
        refs = write_pack(clips, args.clean_root, args.pack)
        print(f"pack: {len(refs)} references -> {args.pack}")


if __name__ == "__main__":
    main()
