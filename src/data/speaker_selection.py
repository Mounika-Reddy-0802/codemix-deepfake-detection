"""Rank candidate reference speakers by SNR and duration (W3-T4, owner L).

The team has to pick 30-50 adult speakers with enough clean audio to clone from.
Listening to every speaker is not feasible, so this module does the scripted half:
it indexes the preprocessed clips, estimates a per-clip SNR, aggregates per
speaker, and produces a ranked shortlist. The team then listens to the top of that
list (the human half of W3-T4) and confirms the final selection.

Two guards are baked in:

- the clip index refuses to walk quarantined directories, so HiACC child audio can
  never reach a shortlist (:mod:`src.data.preprocess` owns the exclusion list);
- the ranking is a pure pandas function over a clip table, so it is unit-tested in
  CI and the shortlist is reproducible from the committed index.

Pool carving (train / adaptation / eval) is the other half of W3-T4 and lives in
``src/data/speaker_pools.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.audio_utils import TARGET_SR, amp_to_db

#: Columns of the clip index this module consumes and produces.
CLIP_COLUMNS = ["filepath", "speaker", "source", "duration_seconds", "snr_db"]

#: Frame length for the SNR estimate. 25 ms is the usual speech-analysis window.
FRAME_SECONDS = 0.025
#: Percentiles used as the noise floor and the speech level.
NOISE_PERCENTILE = 10.0
SPEECH_PERCENTILE = 90.0


@dataclass
class SelectionConfig:
    """Thresholds for the shortlist. Defaults follow the plan (30-50 speakers)."""

    min_total_seconds: float = 30.0
    min_clip_seconds: float = 3.0
    min_snr_db: float = 10.0
    n_min: int = 30
    n_max: int = 50


# --------------------------------------------------------------------------- #
# Per-clip SNR (pure numpy, testable)
# --------------------------------------------------------------------------- #
def frame_energies_db(
    audio: np.ndarray, sr: int = TARGET_SR, frame_seconds: float = FRAME_SECONDS
) -> np.ndarray:
    """Per-frame RMS level in dB. Empty or all-silent input gives an empty array."""
    signal = np.asarray(audio, dtype=np.float64)
    frame = max(1, int(round(frame_seconds * sr)))
    if signal.size < frame:
        return np.array([], dtype=np.float64)
    n_frames = signal.size // frame
    frames = signal[: n_frames * frame].reshape(n_frames, frame)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    return np.array([amp_to_db(float(v)) for v in rms], dtype=np.float64)


def estimate_snr_db(audio: np.ndarray, sr: int = TARGET_SR) -> float:
    """Rough SNR: the gap between the speech level and the noise floor.

    Speech is taken as the 90th percentile of frame energy and the noise floor as
    the 10th; their difference is a standard cheap proxy that ranks recordings
    well even though it is not a calibrated SNR. Silent or too-short input returns
    ``0.0`` so it sorts to the bottom rather than blowing up the ranking.
    """
    energies = frame_energies_db(audio, sr)
    if energies.size == 0:
        return 0.0
    noise = float(np.percentile(energies, NOISE_PERCENTILE))
    speech = float(np.percentile(energies, SPEECH_PERCENTILE))
    return float(max(0.0, speech - noise))


# --------------------------------------------------------------------------- #
# Clip index (touches disk; the aggregation below does not)
# --------------------------------------------------------------------------- #
def speaker_from_path(path: str) -> str:
    """Speaker id from a preprocessed clip path.

    Convention from ``scripts/02_preprocess_all.sh``: clips land under
    ``<corpus>/<speaker>/<utterance>_segNNN.wav``, so the parent directory is the
    speaker. Falls back to the filename stem when the tree is flat.
    """
    from pathlib import Path

    parts = Path(str(path).replace("\\", "/")).parts
    return parts[-2] if len(parts) >= 2 else Path(path).stem


def index_clips(root: str, source: str, sr: int = TARGET_SR) -> pd.DataFrame:
    """Build the clip index for one preprocessed corpus.

    Reads audio, so this is the slow step; the result is written to CSV and every
    later step works from that file. Quarantined directories are never walked.
    """
    from src.data.preprocess import audio_files
    from src.utils.audio_utils import load_wav

    rows: list[dict[str, object]] = []
    for path in audio_files(root):
        try:
            audio, file_sr = load_wav(str(path), target_sr=sr)
        except Exception as exc:  # noqa: BLE001 - a bad clip must not stop the index
            print(f"  [skip] {path}: {type(exc).__name__}: {exc}")
            continue
        rows.append(
            {
                "filepath": str(path),
                "speaker": speaker_from_path(str(path)),
                "source": source,
                "duration_seconds": len(audio) / file_sr if file_sr else 0.0,
                "snr_db": estimate_snr_db(audio, file_sr),
            }
        )
    return pd.DataFrame(rows, columns=CLIP_COLUMNS)


# --------------------------------------------------------------------------- #
# Ranking + shortlist (pure pandas, testable)
# --------------------------------------------------------------------------- #
def rank_speakers(clips: pd.DataFrame, config: SelectionConfig | None = None) -> pd.DataFrame:
    """Aggregate clips per speaker and rank them best-first.

    Clips shorter than ``min_clip_seconds`` are dropped before aggregation: a
    two-second fragment is not a usable cloning reference and would inflate a
    speaker's clip count without adding usable audio.

    Ranked by median SNR then total duration, so a speaker with plenty of noisy
    audio does not outrank a speaker with enough clean audio.
    """
    cfg = config or SelectionConfig()
    if clips.empty:
        return pd.DataFrame(
            columns=["speaker", "source", "n_clips", "total_seconds", "median_snr_db", "eligible"]
        )

    usable = clips[clips["duration_seconds"] >= cfg.min_clip_seconds]
    if usable.empty:
        return pd.DataFrame(
            columns=["speaker", "source", "n_clips", "total_seconds", "median_snr_db", "eligible"]
        )

    grouped = (
        usable.groupby("speaker", dropna=False)
        .agg(
            source=("source", "first"),
            n_clips=("filepath", "count"),
            total_seconds=("duration_seconds", "sum"),
            median_snr_db=("snr_db", "median"),
        )
        .reset_index()
    )
    grouped["eligible"] = (grouped["total_seconds"] >= cfg.min_total_seconds) & (
        grouped["median_snr_db"] >= cfg.min_snr_db
    )
    return grouped.sort_values(
        ["eligible", "median_snr_db", "total_seconds"], ascending=[False, False, False]
    ).reset_index(drop=True)


def shortlist(ranked: pd.DataFrame, config: SelectionConfig | None = None) -> pd.DataFrame:
    """Top eligible speakers, capped at ``n_max``.

    Returns only eligible speakers -- never pads the list with speakers that fail
    the thresholds just to reach ``n_min``. :func:`enough_speakers` is how the
    caller finds out the corpus came up short.
    """
    cfg = config or SelectionConfig()
    if ranked.empty:
        return ranked
    return ranked[ranked["eligible"]].head(cfg.n_max).reset_index(drop=True)


def enough_speakers(selected: pd.DataFrame, config: SelectionConfig | None = None) -> bool:
    """Whether the shortlist reaches the plan's minimum of 30 speakers."""
    cfg = config or SelectionConfig()
    return len(selected) >= cfg.n_min


def selection_summary(ranked: pd.DataFrame, config: SelectionConfig | None = None) -> dict:
    """Numbers for the weekly doc and the listening session."""
    cfg = config or SelectionConfig()
    picked = shortlist(ranked, cfg)
    return {
        "speakers_indexed": int(len(ranked)),
        "speakers_eligible": int(ranked["eligible"].sum()) if len(ranked) else 0,
        "shortlisted": int(len(picked)),
        "enough": bool(enough_speakers(picked, cfg)),
        "total_hours": round(float(picked["total_seconds"].sum()) / 3600.0, 3)
        if len(picked)
        else 0.0,
        "median_snr_db": round(float(picked["median_snr_db"].median()), 2) if len(picked) else 0.0,
    }


def main() -> None:
    """CLI: ``python -m src.data.speaker_selection --root DIR --source mucs2021``."""
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Rank candidate reference speakers")
    parser.add_argument("--root", required=True, help="preprocessed corpus directory")
    parser.add_argument("--source", required=True, help="corpus name, e.g. mucs2021")
    parser.add_argument("--index-out", default="data/manifests/clip_index.csv")
    parser.add_argument("--ranked-out", default="data/manifests/speaker_ranking.csv")
    parser.add_argument("--min-total-seconds", type=float, default=30.0)
    parser.add_argument("--min-snr-db", type=float, default=10.0)
    args = parser.parse_args()

    cfg = SelectionConfig(min_total_seconds=args.min_total_seconds, min_snr_db=args.min_snr_db)
    clips = index_clips(args.root, args.source)
    ranked = rank_speakers(clips, cfg)

    for path, frame in ((args.index_out, clips), (args.ranked_out, ranked)):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)

    print(json.dumps(selection_summary(ranked, cfg), indent=2))
    print(f"\nwrote {args.index_out} and {args.ranked_out}")
    print("next: the team listens to the top of the ranking and confirms the selection")


if __name__ == "__main__":
    main()
