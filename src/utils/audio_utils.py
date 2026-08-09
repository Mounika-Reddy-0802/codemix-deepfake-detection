"""Shared audio helpers: level math (pure numpy) and lazy-loaded I/O + resampling.

The numeric helpers (``rms``, dB conversions, ``to_mono``) are pure numpy so they
import cheaply and are unit-testable without the heavy audio stack. File I/O and
resampling import ``soundfile`` / ``librosa`` lazily, so importing this module does
not pull in those wheels (keeps CI light).
"""

from __future__ import annotations

import numpy as np

# The project's canonical working format.
TARGET_SR = 16_000
EPS = 1e-12


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Collapse audio to a 1-D mono signal.

    Accepts ``(n,)``, ``(channels, n)`` or ``(n, channels)`` and averages channels.
    """
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 1:
        return a
    if a.ndim == 2:
        # Put the shorter axis as channels, average it out.
        ch_axis = 0 if a.shape[0] < a.shape[1] else 1
        return a.mean(axis=ch_axis).astype(np.float32)
    raise ValueError(f"expected 1-D or 2-D audio, got shape {a.shape}")


def rms(signal: np.ndarray) -> float:
    """Root-mean-square level of a signal."""
    s = np.asarray(signal, dtype=np.float64)
    if s.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(s * s)))


def amp_to_db(amp: float) -> float:
    """Linear amplitude -> decibels."""
    return 20.0 * float(np.log10(max(amp, EPS)))


def db_to_amp(db: float) -> float:
    """Decibels -> linear amplitude."""
    return float(10.0 ** (db / 20.0))


def peak_normalize(signal: np.ndarray, peak: float = 0.99) -> np.ndarray:
    """Scale so the maximum absolute sample equals ``peak`` (no clipping)."""
    s = np.asarray(signal, dtype=np.float32)
    m = float(np.max(np.abs(s))) if s.size else 0.0
    if m < EPS:
        return s
    return (s * (peak / m)).astype(np.float32)


def rms_normalize(signal: np.ndarray, target_dbfs: float = -23.0) -> np.ndarray:
    """Scale the signal to a target RMS level in dBFS, then guard against clipping."""
    s = np.asarray(signal, dtype=np.float32)
    cur = rms(s)
    if cur < EPS:
        return s
    gain = db_to_amp(target_dbfs) / cur
    out = (s * gain).astype(np.float32)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:
        out = (out / peak).astype(np.float32)
    return out


def load_wav(path: str, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    """Load an audio file as float32 mono. Optionally resample to ``target_sr``.

    Uses ``soundfile`` (lazy import). Returns ``(audio, sample_rate)``.
    """
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    audio = to_mono(audio)
    if target_sr is not None and sr != target_sr:
        audio = resample(audio, sr, target_sr)
        sr = target_sr
    return audio, sr


def save_wav(path: str, audio: np.ndarray, sr: int) -> None:
    """Write a mono float32 signal to a WAV file (lazy ``soundfile`` import)."""
    import soundfile as sf

    sf.write(path, np.asarray(audio, dtype=np.float32), sr)


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a mono signal (lazy ``librosa`` import)."""
    if orig_sr == target_sr:
        return np.asarray(audio, dtype=np.float32)
    import librosa

    out = librosa.resample(
        np.asarray(audio, dtype=np.float32), orig_sr=orig_sr, target_sr=target_sr
    )
    return out.astype(np.float32)
