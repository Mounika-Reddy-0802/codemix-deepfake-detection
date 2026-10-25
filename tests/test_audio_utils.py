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


def test_trim_silence_keeps_only_the_loud_middle() -> None:
    sr = 16_000
    pad = np.zeros(sr // 2, dtype=np.float32)
    tone = (0.1 * np.sin(np.arange(sr) * 0.1)).astype(np.float32)
    out = au.trim_silence(np.concatenate([pad, tone, pad]), sr)
    assert abs(out.size - sr) <= int(0.02 * sr)
    assert au.trim_silence(np.zeros(sr, dtype=np.float32), sr).size == 0


def test_load_any_reads_wav_and_decodes_phone_formats_with_ffmpeg(tmp_path) -> None:
    import shutil
    import subprocess

    import pytest

    pytest.importorskip("soundfile")  # audio wheel, not installed in CI
    sr = 16_000
    tone = (0.1 * np.sin(2 * np.pi * 440 * np.arange(2 * sr) / sr)).astype(np.float32)
    wav = tmp_path / "call.wav"
    au.save_wav(str(wav), tone, sr)
    assert au.load_any(str(wav), sr).size == 2 * sr
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is not installed")
    m4a = tmp_path / "call.m4a"
    subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(wav), str(m4a)], check=True)
    audio = au.load_any(str(m4a), sr)
    assert audio.dtype == np.float32 and abs(audio.size / sr - 2.0) < 0.1
