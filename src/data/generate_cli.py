"""One entry point for the week-13 generation runs: XTTS, Tortoise, IndicF5, Chatterbox.

``python -m src.data.generate_cli --tool xtts_v2 --pack <dir> --out <dir> --pool eval_ext
--shard 0 --num-shards 2`` generates every job in ``<pack>/generation_jobs.csv`` whose
position is ``shard`` modulo ``num-shards``, so two processes on two GPUs split one
table without overlap. Every driver skips clips that already exist, so a re-run resumes.

Each tool runs in its own Python environment on Kaggle (their pinned libraries
conflict), which is why this is a command and not a notebook function.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

TOOLS = ("xtts_v2", "tortoise", "indicf5", "chatterbox")


def shard(jobs: pd.DataFrame, index: int, count: int) -> pd.DataFrame:
    """Every ``count``-th row starting at ``index`` (by table order)."""
    if not 0 <= index < count:
        raise ValueError(f"shard {index} of {count}")
    return jobs.iloc[index::count].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate clones from a pack's job table")
    parser.add_argument("--tool", required=True, choices=TOOLS)
    parser.add_argument("--pack", required=True, help="folder with refs/ and generation_jobs.csv")
    parser.add_argument("--out", required=True)
    parser.add_argument("--pool", required=True, help="the one pool the table may contain")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    # Launched from a notebook, this process inherits Jupyter's inline plotting backend,
    # which does not exist in the tool environments; IndicF5 imports matplotlib.
    os.environ["MPLBACKEND"] = "Agg"

    table = pd.read_csv(os.path.join(args.pack, "generation_jobs.csv"))
    if "pool" in table and set(table["pool"].astype(str)) != {args.pool}:
        raise SystemExit(
            f"FIREWALL: table has pools {sorted(set(table['pool']))}, expected {args.pool}"
        )
    jobs = shard(table, args.shard, args.num_shards)
    print(
        f"{args.tool}: shard {args.shard}/{args.num_shards} -> {len(jobs)} of {len(table)} jobs",
        flush=True,
    )

    if args.tool == "indicf5":
        from src.data.indicf5_generation import generate_jobs, load_indicf5

        model = load_indicf5(token=os.environ.get("HF_TOKEN"))
        written = generate_jobs(jobs, args.pack, args.out, model)
    elif args.tool == "chatterbox":
        from src.data.chatterbox_generation import generate_jobs, load_chatterbox

        written = generate_jobs(jobs, args.pack, args.out, load_chatterbox())
    elif args.tool == "tortoise":
        from src.data.heldout_tts import build_heldout_jobs, generate_heldout_batch, load_tortoise

        refs = {s: os.path.join(args.pack, "refs", f"{s}.wav") for s in jobs["speaker"].astype(str)}
        pairs = list(zip(jobs["speaker"].astype(str), jobs["transcript"].astype(str), strict=True))
        clone_jobs = build_heldout_jobs(refs, pairs, args.out, pool=args.pool)
        for job, name in zip(clone_jobs, jobs["output_path"], strict=True):
            object.__setattr__(job, "output_path", os.path.join(args.out, os.path.basename(name)))
        todo = [j for j in clone_jobs if not os.path.exists(j.output_path)]
        written = generate_heldout_batch(
            todo,
            load_tortoise(),
            metadata_path=os.path.join(args.out, f"metadata_{args.shard}.jsonl"),
        )
    else:
        from src.data.spoof_generation import run_jobs_file

        part = os.path.join(args.out, f"jobs_shard{args.shard}.csv")
        os.makedirs(args.out, exist_ok=True)
        jobs.to_csv(part, index=False)
        written = run_jobs_file(
            part,
            args.pack,
            args.out,
            metadata_path=os.path.join(args.out, f"metadata_{args.shard}.jsonl"),
            expect_pool=args.pool,
        )
    print(f"{args.tool}: wrote {len(written)} clip(s) this run", flush=True)


if __name__ == "__main__":
    main()
