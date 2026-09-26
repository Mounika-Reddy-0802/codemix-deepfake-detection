"""The interpretability probes change exactly what they claim to, and nothing else."""

from __future__ import annotations

import numpy as np
import pytest

from src.inference import interpretability as ip

SR = ip.SAMPLE_RATE


def _voice(seconds: float = 10.0, f0: float = 120.0) -> np.ndarray:
    """A harmonic tone with a syllable-rate envelope: a crude stand-in for speech."""
    t = np.arange(int(seconds * SR)) / SR
    tone = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 8))
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 4 * t))
    return (0.1 * tone * envelope).astype(np.float32)


def test_speech_windows_skips_silence_and_caps_count():
    audio = np.concatenate([np.zeros(8 * SR, np.float32), _voice(10.0)])
    windows = ip.speech_windows(audio, max_windows=3)
    assert len(windows) == 3
    assert all(w.size == 4 * SR for w in windows)
    assert ip.speech_windows(np.zeros(10 * SR, np.float32), max_windows=5) == []


def test_speech_windows_are_level_normalised():
    windows = ip.speech_windows(_voice() * 0.1, max_windows=1)  # about -45 dBFS, above the gate
    level = 20 * np.log10(np.sqrt(np.mean(windows[0] ** 2)))
    assert level == pytest.approx(-23.0, abs=0.5)


def test_shuffle_keeps_every_chunk_and_changes_the_order():
    audio = np.arange(SR, dtype=np.float32)
    out = ip.shuffle_chunks(audio, 100, np.random.default_rng(0))
    assert out.size == audio.size
    assert np.array_equal(np.sort(out), audio)
    assert not np.array_equal(out, audio)
    chunks = out.reshape(-1, SR // 10)
    assert all(np.all(np.diff(c) == 1) for c in chunks), "chunk interiors must stay intact"


def test_stft_roundtrip_is_near_identity():
    pytest.importorskip("librosa")  # heavy wheel, not installed in CI
    audio = _voice(2.0)
    out = ip.stft_roundtrip(audio)
    assert np.max(np.abs(out - audio)) < 1e-4


def test_phase_scramble_keeps_magnitude_and_changes_the_wave():
    pytest.importorskip("librosa")  # heavy wheel, not installed in CI
    import librosa

    audio = _voice(2.0)
    out = ip.phase_scramble(audio, np.random.default_rng(0))
    mag_in = np.abs(librosa.stft(audio, n_fft=512, hop_length=128))
    mag_out = np.abs(librosa.stft(out, n_fft=512, hop_length=128))
    assert np.corrcoef(mag_in.ravel(), mag_out.ravel())[0, 1] > 0.8
    assert np.corrcoef(audio, out)[0, 1] < 0.5


def test_white_noise_hits_the_target_snr():
    audio = _voice(4.0)
    noisy = ip.white_noise(audio, 10.0, np.random.default_rng(0))
    noise = noisy - audio
    snr = 10 * np.log10(np.mean(audio**2) / np.mean(noise**2))
    assert snr == pytest.approx(10.0, abs=0.1)


def test_band_filters_keep_and_remove_the_right_energy():
    pytest.importorskip("scipy")  # heavy wheel, not installed in CI
    t = np.arange(2 * SR) / SR
    low = np.sin(2 * np.pi * 500 * t).astype(np.float32)
    high = np.sin(2 * np.pi * 3000 * t).astype(np.float32)
    kept = ip.band_filter(low + high, 0, 1000, keep=True)
    assert np.corrcoef(kept, low)[0, 1] > 0.99
    removed = ip.band_filter(low + high, 2000, 3990, keep=False)
    assert np.corrcoef(removed, low)[0, 1] > 0.99


def test_every_registered_perturbation_returns_finite_float32():
    pytest.importorskip("librosa")  # heavy wheel, not installed in CI
    audio = _voice(5.0)
    for name, fn in ip.registry().items():
        out = fn(audio, np.random.default_rng(0))
        assert out.dtype == np.float32, name
        assert np.all(np.isfinite(out)), name
        assert out.size > 0, name


def test_perturbation_table_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown"):
        ip.perturbation_table({"a": _voice()}, ["no_such_probe"], lambda w: 0.5, 2, 0)


def test_perturbation_table_scores_every_clip_under_every_name():
    clips = {"loud": _voice(), "quiet": _voice() * 0.1}
    table = ip.perturbation_table(clips, ["original", "time_reverse"], lambda w: 0.25, 2, 0)
    assert set(table) == {"original", "time_reverse"}
    assert all(row == {"loud": 0.25, "quiet": 0.25} for row in table.values())


def test_layer_separation_finds_a_separable_layer_and_not_a_random_one():
    rng = np.random.default_rng(0)
    n, dims = 80, 16
    labels = np.array([1] * 40 + [0] * 40)
    noise_layer = rng.standard_normal((n, dims))
    signal_layer = rng.standard_normal((n, dims))
    signal_layer[:, 0] += 6.0 * (labels * 2 - 1)
    aucs = ip.layer_separation(np.stack([noise_layer, signal_layer], 1), labels, 0.6, 0)
    assert aucs[1] > 0.99
    assert 0.2 < aucs[0] < 0.8


def test_layer_separation_validates_shapes():
    with pytest.raises(ValueError):
        ip.layer_separation(np.zeros((10, 4)), np.zeros(10), 0.6, 0)


def test_shuffle_does_not_mutate_its_input():
    """A view-based shuffle would corrupt the caller's audio and every later probe."""
    audio = _voice(2.0)
    before = audio.copy()
    ip.shuffle_chunks(audio, 100, np.random.default_rng(0))
    assert np.array_equal(audio, before)


def test_no_perturbation_mutates_its_input():
    pytest.importorskip("librosa")  # heavy wheel, not installed in CI
    audio = _voice(3.0)
    before = audio.copy()
    for name, fn in ip.registry().items():
        fn(audio, np.random.default_rng(0))
        assert np.array_equal(audio, before), f"{name} modified the caller's audio"


def test_perturbation_table_gives_each_row_the_untouched_clip():
    clips = {"a": _voice(6.0)}
    seen = []
    table = ip.perturbation_table(
        clips, ["shuffle_5ms", "original"], lambda w: float(seen.append(w.copy()) or 0.5), 1, 0
    )
    assert table["original"]["a"] == 0.5
    original_window = ip.speech_windows(clips["a"], 1)[0]
    assert np.array_equal(seen[-1], original_window), "the last row must see unshuffled audio"
