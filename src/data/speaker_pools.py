"""Carve and freeze the three disjoint speaker pools (W3-T4, owner SK).

The v2 plan needs **three** pools, not two:

- ``train``      -- S3 native code-mixed training; XTTS-v2 + RVC clones (CM01/CM02)
- ``adaptation`` -- S2 LoRA adaptation only (CM03), never seen by S3
- ``eval``       -- every evaluation column, plus the held-out Tortoise set (CM04)

``build_manifests.carve_pools`` carved two (eval / adaptation) under the old plan
and is kept for the Week-2 history; this module supersedes it.

The carve happens **once, before any audio is generated**, and is then frozen:
the CSV is committed and its SHA-256 recorded, so a later "why did this speaker
move pools?" question has an answer. If a pool assignment changed after
generation, every disjointness guarantee in the paper would be void — a clone of
a train-pool speaker sitting in the eval column is exactly the leak the whole
anti-leakage suite exists to prevent.

Pure stdlib + pandas: no audio, testable in CI.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pandas as pd

#: Pool names, in the order they are carved.
POOL_NAMES = ("train", "adaptation", "eval")

#: Default split. Train is largest (S3 needs the most data); eval must stay big
#: enough that per-cell EERs have usable confidence intervals.
DEFAULT_FRACTIONS = {"train": 0.50, "adaptation": 0.20, "eval": 0.30}

POOL_COLUMNS = ["speaker", "pool", "source"]


class PoolError(AssertionError):
    """Raised when a pool assignment violates disjointness or the frozen file."""


@dataclass(frozen=True)
class FreezeRecord:
    """What was frozen, and the hash that proves it has not moved."""

    path: str
    sha256: str
    n_speakers: int
    counts: dict

    def describe(self) -> str:
        parts = ", ".join(f"{k}={v}" for k, v in sorted(self.counts.items()))
        return f"{self.path} [{self.sha256[:12]}] {self.n_speakers} speakers ({parts})"


def carve_pools(
    speakers: list[str],
    fractions: dict[str, float] | None = None,
    seed: int = 1234,
) -> dict[str, list[str]]:
    """Split speakers into the three disjoint pools.

    Deterministic given ``seed``. Every speaker lands in exactly one pool and no
    speaker is dropped — the rounding remainder goes to the last pool rather than
    disappearing.
    """
    import numpy as np

    fractions = fractions or DEFAULT_FRACTIONS
    missing = set(POOL_NAMES) - set(fractions)
    if missing:
        raise ValueError(f"fractions missing pools: {sorted(missing)}")
    total = sum(fractions[name] for name in POOL_NAMES)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1.0, got {total}")

    unique = sorted(set(speakers))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))

    pools: dict[str, list[str]] = {name: [] for name in POOL_NAMES}
    start = 0
    cumulative = 0.0
    for name in POOL_NAMES[:-1]:
        cumulative += fractions[name]
        end = int(round(cumulative * len(unique)))
        pools[name] = sorted(unique[i] for i in order[start:end])
        start = end
    pools[POOL_NAMES[-1]] = sorted(unique[i] for i in order[start:])
    return pools


def check_disjoint(pools: dict[str, list[str]]) -> None:
    """Raise :class:`PoolError` if any speaker appears in more than one pool."""
    seen: dict[str, str] = {}
    for name, members in pools.items():
        for speaker in members:
            if speaker in seen:
                raise PoolError(f"speaker {speaker!r} in both {seen[speaker]!r} and {name!r}")
            seen[speaker] = name


def pools_to_frame(pools: dict[str, list[str]], source: str = "mucs2021") -> pd.DataFrame:
    """Flatten the pools into the frozen manifest shape."""
    rows = [
        {"speaker": speaker, "pool": name, "source": source}
        for name in POOL_NAMES
        for speaker in pools.get(name, [])
    ]
    return (
        pd.DataFrame(rows, columns=POOL_COLUMNS)
        .sort_values(["pool", "speaker"])
        .reset_index(drop=True)
    )


def frame_hash(frame: pd.DataFrame) -> str:
    """SHA-256 of the canonical CSV form, so the freeze is verifiable."""
    canonical = frame[POOL_COLUMNS].sort_values(["pool", "speaker"]).to_csv(index=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def pool_of(frame: pd.DataFrame, speaker: str) -> str | None:
    """Which pool a speaker belongs to, or ``None`` if unknown."""
    match = frame.loc[frame["speaker"].astype(str) == str(speaker), "pool"]
    return None if match.empty else str(match.iloc[0])


def freeze(frame: pd.DataFrame, path: str = "data/manifests/speaker_pools.csv") -> FreezeRecord:
    """Write the frozen pool manifest and return its hash record."""
    from pathlib import Path

    check_disjoint({name: list(g["speaker"]) for name, g in frame.groupby("pool")})
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ordered = frame[POOL_COLUMNS].sort_values(["pool", "speaker"]).reset_index(drop=True)
    ordered.to_csv(out, index=False)
    return FreezeRecord(
        path=str(path),
        sha256=frame_hash(ordered),
        n_speakers=int(ordered["speaker"].nunique()),
        counts={k: int(v) for k, v in ordered["pool"].value_counts().items()},
    )


def verify_frozen(path: str, expected_sha256: str) -> None:
    """Raise :class:`PoolError` if the frozen file no longer matches its hash."""
    frame = pd.read_csv(path)
    actual = frame_hash(frame)
    if actual != expected_sha256:
        raise PoolError(
            f"{path} has changed since it was frozen "
            f"(expected {expected_sha256[:12]}, got {actual[:12]}). "
            "Pools may not move after generation -- every disjointness guarantee "
            "in the paper depends on this file being stable."
        )


def main() -> None:
    """CLI: ``python -m src.data.speaker_pools --shortlist data/manifests/speaker_ranking.csv``."""
    import argparse

    parser = argparse.ArgumentParser(description="Carve and freeze the three speaker pools")
    parser.add_argument(
        "--shortlist",
        required=True,
        help="CSV with a 'speaker' column (from src.data.speaker_selection)",
    )
    parser.add_argument("--out", default="data/manifests/speaker_pools.csv")
    parser.add_argument("--source", default="mucs2021")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--verify", help="verify the frozen file against this SHA-256")
    args = parser.parse_args()

    if args.verify:
        verify_frozen(args.out, args.verify)
        print(f"{args.out} matches {args.verify[:12]} -- pools unchanged")
        return

    shortlist = pd.read_csv(args.shortlist)
    if "speaker" not in shortlist.columns:
        raise SystemExit(f"{args.shortlist} has no 'speaker' column")

    pools = carve_pools(shortlist["speaker"].astype(str).tolist(), seed=args.seed)
    check_disjoint(pools)
    record = freeze(pools_to_frame(pools, args.source), args.out)

    print(record.describe())
    print("\nrecord this hash in docs/progress.md and the datasheet; the pools are")
    print("frozen from here -- they may not move after generation starts.")


if __name__ == "__main__":
    main()
