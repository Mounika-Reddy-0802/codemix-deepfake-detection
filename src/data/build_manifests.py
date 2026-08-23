"""Manifest schema, speaker-disjoint splits, and the anti-leakage checklist.

A manifest is a CSV with one row per audio clip. Speaker leakage between training
and evaluation is the single bug that could invalidate the whole paper, so the
checks below are enforced by ``tests/test_splits.py`` and are meant to be run
before every training launch (see the golden rule in the team rules doc).

Key convention: a **cloned** voice of speaker X carries speaker id ``X`` (not the
tool's id), so speaker-disjointness holds across bonafide *and* spoof.
"""

from __future__ import annotations

import pandas as pd

MANIFEST_COLUMNS = [
    "filepath",  # path to the audio clip
    "label",  # "bonafide" | "spoof"
    "language",  # "en" | "hi" | "ta" | "hi-en"
    "speaker",  # underlying speaker id (clones use the source speaker's id)
    "source",  # corpus / generator: asvspoof2019_la, mucs2021, hiacc, indicsynth, ...
    "tool",  # "none" for real; "xtts_v2" | "tortoise" | ... for spoofs
    "condition",  # "clean" | "channel_matched"
    "split",  # "train" | "dev" | "eval"
]

TRAINING_SPLITS = {"train", "dev"}
HELD_OUT_TOOL = "tortoise"
EVAL_ONLY_SOURCES = {"indicsynth", "indictts_deepfake", "indicvoices"}


class LeakageError(AssertionError):
    """Raised when the anti-leakage checklist fails."""


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def read_manifest(path: str) -> pd.DataFrame:
    """Read a manifest CSV and validate its columns."""
    df = pd.read_csv(path)
    missing = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: manifest missing columns {missing}")
    return df


def write_manifest(df: pd.DataFrame, path: str) -> None:
    """Write a manifest CSV with the canonical column order."""
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df[MANIFEST_COLUMNS].to_csv(path, index=False)


# --------------------------------------------------------------------------- #
# Speaker-disjoint splitting
# --------------------------------------------------------------------------- #
def carve_pools(
    speakers: list[str], adaptation_frac: float = 0.3, seed: int = 1234
) -> tuple[list[str], list[str]]:
    """Split a speaker list into disjoint (eval_pool, adaptation_pool).

    Done once, up front, so speaker-disjointness is baked in before any spoof is
    generated (Week 2 requirement).
    """
    import numpy as np

    uniq = sorted(set(speakers))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    n_adapt = int(round(len(uniq) * adaptation_frac))
    adapt = sorted(uniq[i] for i in order[:n_adapt])
    evalp = sorted(uniq[i] for i in order[n_adapt:])
    return evalp, adapt


def assign_split_by_speaker(
    df: pd.DataFrame,
    ratios: dict[str, float] | None = None,
    seed: int = 1234,
) -> pd.DataFrame:
    """Assign each speaker wholly to one split so no speaker crosses splits."""
    import numpy as np

    ratios = ratios or {"train": 0.7, "dev": 0.1, "eval": 0.2}
    names = list(ratios)
    speakers = sorted(df["speaker"].unique())
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(speakers))

    bounds, acc = [], 0.0
    for name in names:
        acc += ratios[name]
        bounds.append((name, int(round(acc * len(speakers)))))

    assignment: dict[str, str] = {}
    start = 0
    for name, end in bounds:
        for i in order[start:end]:
            assignment[speakers[i]] = name
        start = end
    for i in order[start:]:  # any rounding remainder -> last split
        assignment[speakers[i]] = names[-1]

    out = df.copy()
    out["split"] = out["speaker"].map(assignment)
    return out


# --------------------------------------------------------------------------- #
# Anti-leakage checklist (enforced by tests/test_splits.py)
# --------------------------------------------------------------------------- #
def speaker_overlap(df: pd.DataFrame) -> set[str]:
    """Speakers appearing in BOTH a training split and the eval split."""
    train = set(df.loc[df["split"].isin(TRAINING_SPLITS), "speaker"])
    evalp = set(df.loc[df["split"] == "eval", "speaker"])
    return train & evalp


def check_speaker_disjoint(df: pd.DataFrame) -> None:
    overlap = speaker_overlap(df)
    if overlap:
        raise LeakageError(f"speakers in both train and eval: {sorted(overlap)}")


def check_no_heldout_tool_in_training(df: pd.DataFrame) -> None:
    bad = df[(df["split"].isin(TRAINING_SPLITS)) & (df["tool"].str.lower() == HELD_OUT_TOOL)]
    if len(bad):
        raise LeakageError(f"{HELD_OUT_TOOL} (held-out attack) found in training: {len(bad)} rows")


def check_no_eval_only_sources_in_training(df: pd.DataFrame) -> None:
    src = df["source"].str.lower()
    bad = df[(df["split"].isin(TRAINING_SPLITS)) & (src.isin(EVAL_ONLY_SOURCES))]
    if len(bad):
        srcs = sorted(bad["source"].str.lower().unique())
        raise LeakageError(f"eval-only source(s) in training: {srcs}")


def check_no_child_audio(df: pd.DataFrame) -> None:
    src = df["source"].astype(str).str.lower()
    spk = df["speaker"].astype(str).str.lower()
    bad = df[src.str.contains("child") | spk.str.contains("child")]
    if len(bad):
        raise LeakageError(f"child audio present in manifest: {len(bad)} rows")


def run_all_checks(df: pd.DataFrame) -> None:
    """Run the full anti-leakage checklist; raise ``LeakageError`` on any failure."""
    check_speaker_disjoint(df)
    check_no_heldout_tool_in_training(df)
    check_no_eval_only_sources_in_training(df)
    check_no_child_audio(df)
