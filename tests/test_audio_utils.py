"""Unit tests for the pure-numpy audio helpers (Week 2, owner L)."""

import numpy as np

from src.utils import audio_utils as au


def test_to_mono_from_stereo() -> None:
    stereo = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)  # (2, n)
    mono = au.to_mono(stereo)
    assert mono.ndim == 1
    assert np.allclose(mono, [0.5, 0.5])


def test_rms_of_constant() -> None:
    assert abs(au.rms(np.full(100, 0.5, dtype=np.float32)) - 0.5) < 1e-6


def test_db_roundtrip() -> None:
    assert abs(au.db_to_amp(au.amp_to_db(0.5)) - 0.5) < 1e-6


def test_peak_normalize_hits_target_peak() -> None:
    out = au.peak_normalize(np.array([0.1, -0.2, 0.05], dtype=np.float32), peak=0.9)
    assert abs(np.max(np.abs(out)) - 0.9) < 1e-6


def test_rms_normalize_reaches_target_dbfs() -> None:
    x = np.random.default_rng(0).standard_normal(4000).astype(np.float32) * 0.01
    out = au.rms_normalize(x, target_dbfs=-20.0)
    assert abs(au.amp_to_db(au.rms(out)) - (-20.0)) < 0.5
    assert np.max(np.abs(out)) <= 1.0


def test_rms_normalize_silence_is_noop() -> None:
    silence = np.zeros(100, dtype=np.float32)
    assert np.array_equal(au.rms_normalize(silence), silence)
