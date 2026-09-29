"""Carrier transforms: they must change the audio, and change only the audio."""

from __future__ import annotations

import numpy as np
import pytest

from src.inference import carrier_sim as cs


def _voice(seconds: float = 6.0, sr: int = cs.SAMPLE_RATE) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    tone = sum(np.sin(2 * np.pi * 130 * k * t) / k for k in range(1, 9))
    return (0.1 * tone * (0.5 * (1 + np.sin(2 * np.pi * 4 * t)))).astype(np.float32)


def test_the_control_carrier_is_the_identity():
    audio = _voice()
    assert np.array_equal(cs.CARRIERS["none"](audio), audio)


def test_opus_roundtrip_returns_comparable_audio():
    pytest.importorskip("soundfile")
    audio = _voice()
    carried = cs.opus_roundtrip(audio)
    assert carried.dtype == np.float32
    assert abs(carried.size - audio.size) < 0.25 * cs.SAMPLE_RATE, "length must survive"
    assert np.isfinite(carried).all()
    assert not np.array_equal(carried, audio), "a codec that changes nothing is not a codec"


def test_no_carrier_mutates_its_input():
    """A transform that edits the caller's array in place corrupts every later run."""
    pytest.importorskip("soundfile")
    for name in cs.CARRIERS:
        audio = _voice()
        before = audio.copy()
        cs.CARRIERS[name](audio)
        assert np.array_equal(audio, before), f"{name} modified its input"


def test_measure_counts_windows_below_the_threshold():
    audio = _voice(10.0)
    scores = iter([0.9, 0.1, 0.8, 0.05, 0.99])
    result = cs.measure(audio, "clip", "none", lambda w: next(scores), threshold=0.5)
    assert result.windows == 4
    assert result.low_windows == 2
    assert result.min_score == pytest.approx(0.05)


def test_a_silent_clip_reports_no_windows_rather_than_a_score():
    silence = np.zeros(8 * cs.SAMPLE_RATE, dtype=np.float32)
    result = cs.measure(silence, "silence", "none", lambda w: 0.9)
    assert result.windows == 0
    assert np.isnan(result.min_score)


def test_the_g711_carrier_is_the_one_the_model_was_trained_on():
    pytest.importorskip("librosa")
    audio = _voice()
    carried = cs.g711_roundtrip(audio)

    def high_band(x):
        spectrum = np.abs(np.fft.rfft(x)) ** 2
        freqs = np.fft.rfftfreq(x.size, 1 / cs.SAMPLE_RATE)
        return float(spectrum[freqs > 4200].sum() / spectrum.sum())

    assert high_band(carried) < 0.02, "a narrowband line cannot carry above 4 kHz"
