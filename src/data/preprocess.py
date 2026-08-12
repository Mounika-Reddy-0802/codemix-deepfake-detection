"""Preprocess raw audio: resample to 16 kHz mono, VAD-trim, loudness-normalise, segment.

Pipeline order:
    load -> mono -> resample 16 kHz -> silero VAD trim -> RMS loudness norm ->
    segment into 2-10 s chunks.

The segmentation and loudness steps are pure numpy (unit-testable). Resampling and
VAD import ``librosa`` / ``silero-vad`` lazily, so this module imports cheaply.

**Quarantine:** any path under a directory named in :data:`EXCLUDED_DIR_NAMES` is
skipped by :func:`preprocess_dir`. HiACC child recordings are quarantined into
``data/raw/hiacc/_EXCLUDED_children/`` on extraction, and walking a corpus root
with ``rglob`` would otherwise pull them straight back into the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.utils.audio_utils import TARGET_SR, resample, rms_normalize, to_mono

#: Directory names that are never read, at any depth. Ethics-critical: HiACC child
#: audio must not enter the pipeline as bonafide, as a cloning reference, or at all.
EXCLUDED_DIR_NAMES = ("_EXCLUDED_children",)


@dataclass
class PreprocessConfig:
    """Parameters for the preprocessing pipeline."""

    target_sr: int = TARGET_SR
    target_dbfs: float = -23.0
    min_seconds: float = 2.0
    max_seconds: float = 10.0
    vad: bool = True
    vad_threshold: float = 0.5


def segment(
    audio: np.ndarray,
    sr: int,
    min_seconds: float = 2.0,
    max_seconds: float = 10.0,
) -> list[np.ndarray]:
    """Split audio into chunks of at most ``max_seconds``.

    A trailing chunk shorter than ``min_seconds`` is dropped (too short to score).
    A signal already within the window is returned as a single segment.
    """
    a = np.asarray(audio, dtype=np.float32)
    min_len = int(round(min_seconds * sr))
    max_len = int(round(max_seconds * sr))
    if a.size == 0 or max_len <= 0:
        return []
    if a.size <= max_len:
        return [a] if a.size >= min_len else []
    segments: list[np.ndarray] = []
    for start in range(0, a.size, max_len):
        chunk = a[start : start + max_len]
        if chunk.size >= min_len:
            segments.append(chunk)
    return segments


def vad_trim(audio: np.ndarray, sr: int, threshold: float = 0.5) -> np.ndarray:
    """Trim leading/trailing non-speech with silero-vad (lazy import).

    Returns the concatenation of detected speech regions. If silero is unavailable
    or finds no speech, the original signal is returned unchanged.
    """
    try:
        import torch
        from silero_vad import get_speech_timestamps, load_silero_vad
    except ImportError:
        return np.asarray(audio, dtype=np.float32)

    model = load_silero_vad()
    wav = torch.from_numpy(np.asarray(audio, dtype=np.float32))
    stamps = get_speech_timestamps(wav, model, sampling_rate=sr, threshold=threshold)
    if not stamps:
        return np.asarray(audio, dtype=np.float32)
    parts = [np.asarray(audio, dtype=np.float32)[s["start"] : s["end"]] for s in stamps]
    return np.concatenate(parts).astype(np.float32)


def preprocess_signal(
    audio: np.ndarray, sr: int, config: PreprocessConfig | None = None
) -> list[np.ndarray]:
    """Run the full pipeline on an in-memory signal, returning 16 kHz segments."""
    cfg = config or PreprocessConfig()
    x = to_mono(audio)
    if sr != cfg.target_sr:
        x = resample(x, sr, cfg.target_sr)
    if cfg.vad:
        x = vad_trim(x, cfg.target_sr, cfg.vad_threshold)
    x = rms_normalize(x, cfg.target_dbfs)
    return segment(x, cfg.target_sr, cfg.min_seconds, cfg.max_seconds)


def preprocess_file(path: str, out_dir: str, config: PreprocessConfig | None = None) -> list[str]:
    """Preprocess one file and write its segments as 16 kHz WAVs. Returns paths."""
    from pathlib import Path

    from src.utils.audio_utils import load_wav, save_wav

    cfg = config or PreprocessConfig()
    audio, sr = load_wav(path, target_sr=cfg.target_sr)
    segments = preprocess_signal(audio, cfg.target_sr, cfg)
    stem = Path(path).stem
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for i, seg in enumerate(segments):
        dest = out / f"{stem}_seg{i:03d}.wav"
        save_wav(str(dest), seg, cfg.target_sr)
        written.append(str(dest))
    return written


def is_excluded(path, excluded: tuple[str, ...] = EXCLUDED_DIR_NAMES) -> bool:
    """Whether ``path`` sits under a quarantined directory, at any depth."""
    from pathlib import Path

    return any(part in excluded for part in Path(path).parts)


def audio_files(in_dir: str, excluded: tuple[str, ...] = EXCLUDED_DIR_NAMES) -> list:
    """Every ``.wav``/``.flac`` under ``in_dir``, minus anything quarantined."""
    from pathlib import Path

    found = (p for ext in ("*.wav", "*.flac") for p in Path(in_dir).rglob(ext))
    return sorted(p for p in found if not is_excluded(p, excluded))


def preprocess_dir(in_dir: str, out_dir: str, config: PreprocessConfig | None = None) -> int:
    """Preprocess every ``.wav``/``.flac`` under ``in_dir``. Returns segment count.

    Quarantined directories (:data:`EXCLUDED_DIR_NAMES`) are never read.
    """
    cfg = config or PreprocessConfig()
    total = 0
    files = audio_files(in_dir)
    for i, path in enumerate(files, 1):
        try:
            total += len(preprocess_file(str(path), out_dir, cfg))
        except Exception as exc:  # noqa: BLE001 - keep going, report at the end
            print(f"  [skip] {path}: {type(exc).__name__}: {exc}")
        if i % 200 == 0:
            print(f"  processed {i}/{len(files)} files, {total} segments")
    return total


def main() -> None:
    """CLI: ``python -m src.data.preprocess --in-dir RAW --out-dir OUT``."""
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess a corpus to 16 kHz segments")
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--no-vad", action="store_true", help="disable silero VAD")
    parser.add_argument("--target-dbfs", type=float, default=-23.0)
    args = parser.parse_args()

    cfg = PreprocessConfig(vad=not args.no_vad, target_dbfs=args.target_dbfs)
    n = preprocess_dir(args.in_dir, args.out_dir, cfg)
    print(f"done: {n} segments written to {args.out_dir}")


if __name__ == "__main__":
    main()
