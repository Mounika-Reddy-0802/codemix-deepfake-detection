"""Tests for on-the-fly augmentation (src/data/augment.py) and the per-epoch crop."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from src.data import augment as aug
from src.data.dataset import AudioManifestDataset
from src.data.lowlevel_cue import FEATURE_NAMES, features

SR = 16_000


def _speech_like(seconds: float = 2.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    tone = 0.2 * np.sin(2 * np.pi * 220 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
    return (tone + 0.01 * rng.standard_normal(t.size)).astype(np.float32)


def _feature(x: np.ndarray, name: str) -> float:
    return float(features(x, SR)[FEATURE_NAMES.index(name)])


def test_every_operation_keeps_length_dtype_and_peak():
    x = _speech_like()
    rng = np.random.default_rng(1)
    for op in (aug.random_gain, aug.coloured_noise, aug.spectral_tilt):
        y = op(x, rng)
        assert y.dtype == np.float32
        assert y.shape == x.shape
        assert np.max(np.abs(y)) <= aug.PEAK_LIMIT + 1e-6


def test_operations_are_reproducible_for_a_seed():
    x = _speech_like()
    for op in (aug.random_gain, aug.coloured_noise, aug.spectral_tilt):
        a = op(x, np.random.default_rng(42))
        b = op(x, np.random.default_rng(42))
        assert np.array_equal(a, b)


def test_random_gain_stays_within_its_range():
    x = _speech_like()
    for seed in range(30):
        y = aug.random_gain(x, np.random.default_rng(seed), max_db=6.0)
        ratio_db = 20 * np.log10(np.sqrt(np.mean(y**2)) / np.sqrt(np.mean(x**2)))
        assert -6.01 <= ratio_db <= 6.01


def test_coloured_noise_moves_the_zero_crossing_cue():
    # The point of the operation: the zero-crossing rate, a residual shortcut cue,
    # must stop being a fixed property of the source.
    x = _speech_like()
    zcr = [
        _feature(
            aug.coloured_noise(x, np.random.default_rng(s), snr_db=(10, 10)), "zero_crossing_rate"
        )
        for s in range(12)
    ]
    assert max(zcr) - min(zcr) > 0.02


def test_spectral_tilt_moves_high_frequency_energy_but_not_loudness():
    x = _speech_like() + 0.05 * np.random.default_rng(3).standard_normal(2 * SR).astype(np.float32)
    rms_x = np.sqrt(np.mean(x**2))
    hf = []
    for seed in range(12):
        y = aug.spectral_tilt(x, np.random.default_rng(seed), max_db_per_octave=6.0)
        assert np.sqrt(np.mean(y**2)) == pytest.approx(rms_x, rel=0.02)
        hf.append(_feature(y, "hf_energy_ratio"))
    assert max(hf) - min(hf) > 0.05


def test_shaped_noise_has_the_right_spectral_slope():
    rng = np.random.default_rng(0)
    pink = aug._shaped_noise(2**16, rng, exponent=1.0)
    spectrum = np.abs(np.fft.rfft(pink)) ** 2
    low, high = spectrum[10:100].mean(), spectrum[10_000:20_000].mean()
    assert low > 10 * high  # pink: much more power at low frequency


def test_telephone_runs_with_a_stand_in_resampler(monkeypatch):
    from src.data import channel_sim

    def crude_resample(audio, orig_sr, target_sr):
        audio = np.asarray(audio, dtype=np.float32)
        if target_sr < orig_sr:
            return audio[:: orig_sr // target_sr]
        return np.repeat(audio, target_sr // orig_sr)

    monkeypatch.setattr(channel_sim, "resample", crude_resample)
    x = _speech_like()
    y = aug.telephone(x, np.random.default_rng(0))
    assert y.dtype == np.float32
    assert abs(len(y) - len(x)) <= 2
    assert np.max(np.abs(y)) <= aug.PEAK_LIMIT + 1e-6
    assert not np.array_equal(y, x)


def test_recipes_are_reproducible_and_vary_with_the_seed():
    augmenter = aug.build_augment("shortcut_v1")
    x = _speech_like()
    a = augmenter(x, np.random.default_rng(7))
    assert np.array_equal(a, augmenter(x, np.random.default_rng(7)))
    assert not np.array_equal(a, augmenter(x, np.random.default_rng(8)))


def test_unknown_recipe_is_refused():
    with pytest.raises(ValueError):
        aug.build_augment("nope")


def test_augmenter_survives_pickling_for_windows_workers():
    augmenter = pickle.loads(pickle.dumps(aug.build_augment("channel_v1")))
    assert augmenter.name == "channel_v1"


# --------------------------------------------------------------------------- #
# The per-epoch crop in AudioManifestDataset
# --------------------------------------------------------------------------- #


def _dataset(monkeypatch, **kwargs) -> AudioManifestDataset:
    from src.utils import audio_utils

    def load_wav(path, target_sr=SR):
        return np.arange(10 * SR, dtype=np.float32) / (10 * SR), target_sr  # 10 s ramp

    monkeypatch.setattr(audio_utils, "load_wav", load_wav)
    manifest = pd.DataFrame(
        {
            "filepath": [f"c{i}.wav" for i in range(4)],
            "label": ["bonafide", "spoof", "bonafide", "spoof"],
        }
    )
    return AudioManifestDataset(manifest, max_seconds=4.0, random_crop=True, seed=1234, **kwargs)


def test_published_recipe_crop_is_unchanged(monkeypatch):
    # Bit-for-bit the old seeding (seed + index): existing configs must train exactly
    # as they did, so the seed-repeat runs measure the published model.
    ds = _dataset(monkeypatch)
    rng = np.random.default_rng(1234 + 2)
    start = int(rng.integers(0, 10 * SR - 4 * SR + 1))
    expected = (np.arange(10 * SR, dtype=np.float32) / (10 * SR))[start : start + 4 * SR]
    assert np.array_equal(ds[2].waveform, expected)


def test_integer_index_gives_the_same_crop_every_time(monkeypatch):
    ds = _dataset(monkeypatch)
    assert np.array_equal(ds[1].waveform, ds[1].waveform)


def test_epoch_keys_give_a_new_crop_each_epoch_and_repeat_exactly(monkeypatch):
    ds = _dataset(monkeypatch)
    starts = {float(ds[(1, epoch)].waveform[0]) for epoch in range(6)}
    assert len(starts) > 1
    assert np.array_equal(ds[(1, 3)].waveform, ds[(1, 3)].waveform)


def test_augment_hook_is_applied_with_a_seeded_generator(monkeypatch):
    calls = []

    def marker(waveform, rng):
        calls.append(float(rng.random()))
        return waveform * 0.5

    ds = _dataset(monkeypatch, augment=marker)
    out = ds[(0, 2)].waveform
    first = calls[-1]
    ds[(0, 2)]
    assert calls[-1] == first  # same (seed, epoch, index) -> same generator
    assert np.max(out) <= 0.5
