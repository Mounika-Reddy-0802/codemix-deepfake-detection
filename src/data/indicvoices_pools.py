"""Speaker pools for IndicVoices Hindi spontaneous speech (plan task A2.2).

Every Hindi-English training set so far is read lecture speech, and the detectors call
much of genuine spontaneous speech fake (``docs/results/hiacc_final_v1.md``). IndicVoices
holds extempore (unscripted) Hindi from 2,356 speakers. This module carves speakers, once
and before any audio is downloaded or cloned, into three disjoint pools:

- ``iv_test`` -- evaluation only, a second spontaneous test set beside HiACC;
- ``iv_dev`` -- development only, so model selection can see spontaneous speech;
- ``iv_train`` -- genuine spontaneous speakers for adaptation, with seen-tool clones of
  the same speakers so that speaking style cannot become the shortcut.

**Why row groups.** Each speaker's clips are spread over about 18 of the 83 parquet
shards, and audio can only be read a whole row group at a time (about 1,100 clips and
97 MB). Fetching every clip of 200 speakers would mean reading most of the 50 GB. So a
fixed, seeded set of row groups is chosen first (40 groups, about 3.9 GB), and a speaker
is eligible with at least ``min_clips`` extempore clips of 2-15 s **inside those groups**.
Both choices use metadata only (``src.data.indicvoices_inventory``), and both are frozen:
the groups in ``indicvoices_row_groups.csv``, the pools in ``indicvoices_pools.csv``.

The test pool is carved first, as for ``speaker_pools_ext.csv``. Speaker ids are
prefixed ``iv_`` so they can never collide with MUCS or HiACC ids.

Using ``iv_train`` for training is team decision D1 (it amends the golden dataset rule,
as P-025 and P-028 did); until that decision is recorded, the pools are carved but no
training config may use them. HiACC stays evaluation-only either way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import speaker_pools as sp

POOLS_PATH = "data/manifests/indicvoices_pools.csv"
#: canonical hash of the frozen pools (`speaker_pools.frame_hash`), checked by the tests
FROZEN_SHA256 = "f8ba8c36ce87111ae38e5d5ef5d5ed6c6825626562556a6d8cd170fbca355a5b"
GROUPS_PATH = "data/manifests/indicvoices_row_groups.csv"
PREFIX = "iv_"
#: never in a training or dev manifest
EVAL_ONLY = ("iv_test",)
#: allowed in a dev manifest, never in a training manifest
DEV_ONLY = ("iv_dev",)
#: allowed in a training manifest, never in a dev manifest (decision D1, P-038)
TRAIN_ONLY = ("iv_train",)
N_GROUPS = 40
MIN_CLIPS = 20
MIN_SECONDS, MAX_SECONDS = 2.0, 15.0


def row_group_table(subset: str = "hindi", token: str | None = None) -> pd.DataFrame:
    """``shard, group, first_row, rows`` for every row group of a subset (footers only)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    from src.data.indicvoices_inventory import REPO_ID, list_shards

    fs = HfFileSystem(token=token)
    rows = []
    for path in list_shards(subset, token):
        with fs.open(f"datasets/{REPO_ID}/{path}", "rb") as handle:
            meta = pq.ParquetFile(handle).metadata
        first = 0
        for g in range(meta.num_row_groups):
            n = meta.row_group(g).num_rows
            rows.append({"shard": path.split("/")[-1], "group": g, "first_row": first, "rows": n})
            first += n
    return pd.DataFrame(rows)


def choose_groups(groups: pd.DataFrame, n: int = N_GROUPS, seed: int = 1234) -> pd.DataFrame:
    """A seeded draw of ``n`` row groups, in a stable order."""
    order = np.random.default_rng(seed).permutation(len(groups))[:n]
    return groups.iloc[sorted(order)].reset_index(drop=True)


def assign_groups(clips: pd.DataFrame, groups: pd.DataFrame) -> pd.Series:
    """The row group of every clip, from its shard and row number within the shard."""
    out = pd.Series(-1, index=clips.index, dtype="int64")
    for shard, part in groups.groupby("shard"):
        mine = clips["shard"] == shard
        bounds = part.sort_values("first_row")
        index = np.searchsorted(bounds["first_row"].to_numpy(), clips.loc[mine, "row"], "right") - 1
        out.loc[mine] = bounds["group"].to_numpy()[index]
    return out


def usable(clips: pd.DataFrame) -> pd.Series:
    """Extempore clips of 2-15 s: the speech the pools are built from."""
    style = clips["scenario"].astype(str).str.lower().str.contains("extempore")
    return style & clips["duration"].astype(float).between(MIN_SECONDS, MAX_SECONDS)


def eligible_counts(clips: pd.DataFrame, chosen: pd.DataFrame) -> pd.Series:
    """Usable clips per speaker inside the chosen row groups."""
    keys = set(zip(chosen["shard"], chosen["group"], strict=True))
    inside = [(s, g) in keys for s, g in zip(clips["shard"], clips["group"], strict=True)]
    picked = clips[usable(clips) & np.array(inside, dtype=bool)]
    return picked.groupby(picked["speaker_id"].astype(str)).size()


def carve(
    counts: pd.Series,
    n_train: int = 150,
    n_dev: int = 20,
    n_test: int = 30,
    min_clips: int = MIN_CLIPS,
    seed: int = 1234,
) -> pd.DataFrame:
    """``speaker, pool, source`` for a seeded draw of eligible speakers (test pool first).

    ``counts`` maps a speaker to their usable clips inside the chosen row groups.
    """
    eligible = sorted(counts[counts >= min_clips].index.astype(str))
    needed = n_train + n_dev + n_test
    if len(eligible) < needed:
        raise ValueError(
            f"only {len(eligible)} speakers have {min_clips}+ usable extempore clips, "
            f"need {needed}"
        )
    order = np.random.default_rng(seed).permutation(len(eligible))
    drawn = [eligible[i] for i in order[:needed]]
    pools = {
        "iv_test": drawn[:n_test],
        "iv_dev": drawn[n_test : n_test + n_dev],
        "iv_train": drawn[n_test + n_dev :],
    }
    rows = [
        {"speaker": f"{PREFIX}{speaker}", "pool": pool, "source": "indicvoices"}
        for pool, members in pools.items()
        for speaker in members
    ]
    return pd.DataFrame(rows, columns=sp.POOL_COLUMNS)


def _speakers_of(pools: tuple[str, ...], path: str) -> set[str]:
    from pathlib import Path

    if not Path(path).is_file():
        return set()
    frame = pd.read_csv(path)
    return set(frame.loc[frame["pool"].isin(pools), "speaker"].astype(str))


def evaluation_only_speakers(path: str = POOLS_PATH) -> set[str]:
    """Speakers of the evaluation-only pool (``iv_test``); empty if the file is absent."""
    return _speakers_of(EVAL_ONLY, path)


def dev_only_speakers(path: str = POOLS_PATH) -> set[str]:
    """Speakers of the development pool (``iv_dev``); empty if the file is absent."""
    return _speakers_of(DEV_ONLY, path)


def training_speakers(path: str = POOLS_PATH) -> set[str]:
    """Speakers of the training pool (``iv_train``); empty if the file is absent."""
    return _speakers_of(TRAIN_ONLY, path)


def load_clips(cache: str) -> pd.DataFrame:
    """The inventory's cached clip metadata, with shard and row number per clip."""
    from pathlib import Path

    frames = []
    for path in sorted(Path(cache).glob("*.parquet")):
        frame = pd.read_parquet(path, columns=["speaker_id", "scenario", "duration"])
        frames.append(frame.assign(shard=path.name, row=np.arange(len(frame))))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    """CLI: choose and freeze the row groups, then carve and freeze the pools."""
    import argparse
    import os
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Carve IndicVoices speaker pools")
    parser.add_argument("--cache", default="data/indicvoices/meta")
    parser.add_argument("--groups-out", default=GROUPS_PATH)
    parser.add_argument("--out", default=POOLS_PATH)
    parser.add_argument("--n-groups", type=int, default=N_GROUPS)
    parser.add_argument("--min-clips", type=int, default=MIN_CLIPS)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    for path in (args.out, args.groups_out):
        if Path(path).is_file():
            raise SystemExit(f"{path} is already frozen; delete it deliberately to re-carve")

    groups = row_group_table("hindi", os.environ.get("HF_TOKEN"))
    chosen = choose_groups(groups, args.n_groups, args.seed)
    clips = load_clips(args.cache)
    clips["group"] = assign_groups(clips, groups)
    counts = eligible_counts(clips, chosen)
    frame = carve(counts, min_clips=args.min_clips, seed=args.seed)

    Path(args.groups_out).parent.mkdir(parents=True, exist_ok=True)
    chosen.to_csv(args.groups_out, index=False)
    record = sp.freeze(frame, args.out)
    print(f"{len(chosen)} row groups ({int(chosen['rows'].sum())} clips) -> {args.groups_out}")
    print(f"eligible speakers: {int((counts >= args.min_clips).sum())}")
    print(record.describe())


if __name__ == "__main__":
    main()
