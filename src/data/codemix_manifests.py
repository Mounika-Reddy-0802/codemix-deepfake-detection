"""Assemble Stage-3 manifests: bonafide MUCS + matching XTTS clones, one pool.

Produces the same *raw* shape as ``gap_codemix_clean.csv`` (the manifest that
was materialised into the existing 4,434-clip gap-matrix bundle) -- bonafide
rows name the source recording, spoof rows name a standalone generated wav.
Run ``src.data.portable_bundle`` on the output before training or evaluating:
that step crops each bonafide row to its own utterance instead of the whole
8-10 minute recording, which is the bug ``portable_bundle.py`` documents (a
manifest built from ``wav_path`` alone collapsed 2,217 rows to 25 files).

The **adaptation** pool is split train/dev by speaker (dev held out from the
adaptation pool itself, purely for early stopping) -- this is *not* the
speaker-disjointness boundary the paper depends on, which is adaptation vs
eval, enforced separately by ``assert_pool_disjoint``. The **eval** pool is
never split; every row gets ``split="eval"``.

    python -m src.data.codemix_manifests --pool adaptation \\
        --jobs C:/dfdata/generated/xtts_v2_adaptation/generation_jobs.csv \\
        --pack-dir C:/dfdata/generated/xtts_v2_adaptation

    python -m src.data.portable_bundle \\
        --manifest data/manifests/codemix_adapt_train_clean.csv \\
        --out-dir C:/dfdata/colab_bundle_adapt \\
        --manifest-out data/manifests/codemix_adapt_train.csv
"""

from __future__ import annotations

import pandas as pd

from src.data.build_manifests import MANIFEST_COLUMNS
from src.data.pilot_jobs import pool_speakers

ALLOWED_POOLS = ("adaptation", "eval")


class ManifestError(AssertionError):
    """Raised when a code-mixed manifest would violate the pool firewall."""


def bonafide_rows(index: pd.DataFrame, pools: pd.DataFrame, pool: str) -> pd.DataFrame:
    """Real MUCS rows for every speaker in ``pool``.

    ``wav_path`` is carried as-is (the whole recording) and the crop happens later
    in ``portable_bundle.attach_spans``, which needs ``clip_index.csv`` for
    exactly this reason. The span columns are carried through so that step reads
    each row's own utterance instead of re-deriving it positionally.

    One row per *utterance*, not per recording. MUCS packs every utterance of a
    speaker into a single wav, so de-duplicating on ``filepath`` would collapse a
    speaker's whole contribution to one clip and leave the manifest ~99% spoof --
    the gap matrix's balanced 2,217/2,217 construction is what this must match.
    """
    allowed = pool_speakers(pools, pool)
    frame = index[index["speaker"].astype(str).isin(allowed)].copy()
    frame["speaker"] = frame["speaker"].astype(str)
    out = pd.DataFrame(
        {
            "filepath": frame["wav_path"],
            "label": "bonafide",
            "language": "hi-en",
            "speaker": frame["speaker"],
            "source": frame.get("source", "mucs2021"),
            "tool": "none",
            "condition": "clean",
            "utt_id": frame["utt_id"],
            "start_seconds": frame["start_seconds"],
            "end_seconds": frame["end_seconds"],
        }
    )
    return out.drop_duplicates("utt_id").reset_index(drop=True)


def spoof_rows(jobs: pd.DataFrame, pack_dir: str, pool: str, pools: pd.DataFrame) -> pd.DataFrame:
    """Generated clips for every job that actually produced a wav file.

    Cross-checked against disk rather than trusted from the job table: a
    generation run can die partway through, and a manifest naming files that
    do not exist would fail loudly at training time instead of here, where the
    fix is obvious (rerun generation, don't debug the training loop).

    Also cross-checked against ``pools`` -- the same pool firewall
    ``pool_jobs.py`` enforces at generation time, re-checked here in case
    ``jobs`` is a stale or mismatched ``generation_jobs.csv`` (the wrong pack
    handed to the wrong pool). A silent filter would hide exactly the mistake
    this check exists to catch, so it raises instead.
    """
    from pathlib import Path

    from src.data.pilot_jobs import pool_speakers

    allowed = pool_speakers(pools, pool)
    jobs = jobs.copy()
    jobs["speaker"] = jobs["speaker"].astype(str)
    intruders = set(jobs["speaker"]) - allowed
    if intruders:
        raise ManifestError(
            f"generation_jobs.csv contains speaker(s) outside the {pool} pool: "
            f"{sorted(intruders)} -- wrong pack for this pool?"
        )

    root = Path(pack_dir)
    jobs["full_path"] = jobs["output_path"].map(lambda p: str(root / p))
    exists = jobs["full_path"].map(lambda p: Path(p).is_file())
    missing = int((~exists).sum())
    if missing:
        print(f"  [{pool}] {missing}/{len(jobs)} generated clip(s) missing on disk, skipped")
    jobs = jobs[exists]
    if jobs.empty:
        return pd.DataFrame(columns=MANIFEST_COLUMNS[:-1])

    return pd.DataFrame(
        {
            "filepath": jobs["full_path"],
            "label": "spoof",
            "language": "hi-en",
            "speaker": jobs["speaker"],
            "source": "mucs2021",
            "tool": jobs["tool"],
            "condition": "clean",
        }
    ).reset_index(drop=True)


def split_speakers(
    speakers: list[str], dev_frac: float = 0.2, seed: int = 1234
) -> tuple[list[str], list[str]]:
    """Deterministic speaker-level train/dev split within one pool."""
    import numpy as np

    uniq = sorted(set(speakers))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    n_dev = max(1, int(round(len(uniq) * dev_frac))) if len(uniq) > 1 else 0
    dev = sorted(uniq[i] for i in order[:n_dev])
    train = sorted(uniq[i] for i in order[n_dev:])
    return train, dev


def assert_pool_disjoint(pools: pd.DataFrame) -> None:
    """Adaptation and eval speakers must never overlap -- the S2 firewall.

    Belt-and-braces: ``speaker_pools.csv`` is a frozen, hashed, single-source
    partition, so this can only fail if that file itself was hand-edited.
    """
    adapt = pool_speakers(pools, "adaptation")
    evalp = pool_speakers(pools, "eval")
    overlap = adapt & evalp
    if overlap:
        raise ManifestError(f"speakers in both adaptation and eval pools: {sorted(overlap)}")


def build_adaptation_manifests(
    index: pd.DataFrame,
    pools: pd.DataFrame,
    jobs: pd.DataFrame,
    pack_dir: str,
    dev_frac: float = 0.2,
    seed: int = 1234,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(train_df, dev_df)`` for the adaptation pool."""
    assert_pool_disjoint(pools)
    bona = bonafide_rows(index, pools, "adaptation")
    spoof = spoof_rows(jobs, pack_dir, "adaptation", pools)
    if bona.empty or spoof.empty:
        raise ManifestError("adaptation pool has no bonafide or no spoof rows -- generate first")

    train_speakers, dev_speakers = split_speakers(
        sorted(pool_speakers(pools, "adaptation")), dev_frac, seed
    )
    combined = pd.concat([bona, spoof], ignore_index=True)
    train_df = combined[combined["speaker"].isin(train_speakers)].copy()
    dev_df = combined[combined["speaker"].isin(dev_speakers)].copy()
    train_df["split"] = "train"
    dev_df["split"] = "dev"
    return train_df.reset_index(drop=True), dev_df.reset_index(drop=True)


def build_eval_manifest(
    index: pd.DataFrame, pools: pd.DataFrame, jobs: pd.DataFrame, pack_dir: str
) -> pd.DataFrame:
    """Every eval-pool row, bonafide and spoof, ``split="eval"``."""
    assert_pool_disjoint(pools)
    bona = bonafide_rows(index, pools, "eval")
    spoof = spoof_rows(jobs, pack_dir, "eval", pools)
    if bona.empty or spoof.empty:
        raise ManifestError("eval pool has no bonafide or no spoof rows -- generate first")
    combined = pd.concat([bona, spoof], ignore_index=True)
    combined["split"] = "eval"
    return combined.reset_index(drop=True)


def main() -> None:
    """CLI: builds the raw ('clean') manifests; portable_bundle materialises them."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Assemble Stage-3 code-mixed manifests")
    parser.add_argument("--pool", required=True, choices=ALLOWED_POOLS)
    parser.add_argument("--index", default="data/manifests/clip_index.csv")
    parser.add_argument("--pools", default="data/manifests/speaker_pools.csv")
    parser.add_argument("--jobs", required=True, help="generation_jobs.csv from the pack")
    parser.add_argument("--pack-dir", required=True, help="pack root holding outputs/")
    parser.add_argument("--dev-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out-dir", default="data/manifests")
    args = parser.parse_args()

    index = pd.read_csv(args.index)
    pools = pd.read_csv(args.pools)
    jobs = pd.read_csv(args.jobs)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.pool == "adaptation":
        train_df, dev_df = build_adaptation_manifests(
            index, pools, jobs, args.pack_dir, args.dev_frac, args.seed
        )
        train_path = out_dir / "codemix_adapt_train_clean.csv"
        dev_path = out_dir / "codemix_adapt_dev_clean.csv"
        train_df.to_csv(train_path, index=False)
        dev_df.to_csv(dev_path, index=False)
        print(f"wrote {train_path}: {len(train_df)} rows, {train_df['speaker'].nunique()} speakers")
        print(f"wrote {dev_path}: {len(dev_df)} rows, {dev_df['speaker'].nunique()} speakers")
    else:
        eval_df = build_eval_manifest(index, pools, jobs, args.pack_dir)
        eval_path = out_dir / "codemix_eval_clean.csv"
        eval_df.to_csv(eval_path, index=False)
        print(f"wrote {eval_path}: {len(eval_df)} rows, {eval_df['speaker'].nunique()} speakers")

    print("\nRun src.data.portable_bundle on each output before training/evaluating.")


if __name__ == "__main__":
    main()
