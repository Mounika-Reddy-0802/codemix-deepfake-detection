"""Tests for objective channel-simulation verification (Week 3, W3-T2, owner L).

These pin the checks that catch a broken channel chain *before* three people
spend an hour listening. The important one is band-limiting: a chain that forgets
the 8 kHz downsample still sounds plausible on laptop speakers, and would quietly
put wideband audio in a column the paper calls telephony.

Synthetic signals with known spectra -- no corpus needed.
"""

import numpy as np
import pytest

from src.data import channel_qa as qa

SR = 16_000


def _tone(freq: float, seconds: float = 1.0, sr: int = SR, amp: float = 0.3) -> np.ndarray:
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _wideband(seconds: float = 1.0, sr: int = SR) -> np.ndarray:
    """Low tone plus a strong 6 kHz component -- unmistakably wideband."""
    return (_tone(400, seconds, sr) + _tone(6000, seconds, sr, amp=0.3)).astype(np.float32)


# --------------------------------------------------------------------------- #
# High-frequency energy -- the decisive test
# --------------------------------------------------------------------------- #
def test_narrowband_signal_has_no_high_frequency_energy() -> None:
    assert qa.high_frequency_energy_ratio(_tone(400), SR) < 1e-6


def test_wideband_signal_has_high_frequency_energy() -> None:
    assert qa.high_frequency_energy_ratio(_wideband(), SR) > 0.1


def test_empty_signal_is_zero() -> None:
    assert qa.high_frequency_energy_ratio(np.zeros(0, dtype=np.float32), SR) == 0.0


def test_silence_is_zero() -> None:
    assert qa.high_frequency_energy_ratio(np.zeros(SR, dtype=np.float32), SR) == 0.0


# --------------------------------------------------------------------------- #
# Bandwidth
# --------------------------------------------------------------------------- #
def test_bandwidth_tracks_the_tone() -> None:
    assert qa.spectral_bandwidth_hz(_tone(400), SR) == pytest.approx(400, abs=60)


def test_wideband_bandwidth_exceeds_the_telephony_cutoff() -> None:
    assert qa.spectral_bandwidth_hz(_wideband(), SR) > qa.NARROWBAND_NYQUIST_HZ


# --------------------------------------------------------------------------- #
# SNR + correlation
# --------------------------------------------------------------------------- #
def test_identical_signals_report_perfect_snr() -> None:
    clean = _tone(400)
    assert qa.measured_snr_db(clean, clean) > 100.0


def test_pure_gain_change_is_not_counted_as_noise() -> None:
    # The chain changes level; reporting that as noise would be wrong.
    clean = _tone(400)
    assert qa.measured_snr_db(clean, (clean * 0.5).astype(clean.dtype)) > 100.0


def test_added_noise_lowers_snr_monotonically() -> None:
    rng = np.random.default_rng(0)
    clean = _tone(400)
    scores = [
        qa.measured_snr_db(clean, clean + rng.normal(0, s, clean.shape).astype(np.float32))
        for s in (0.001, 0.01, 0.1)
    ]
    assert scores[0] > scores[1] > scores[2]


def test_correlation_is_one_for_identical_signals() -> None:
    clean = _tone(400)
    assert qa.correlation(clean, clean) == pytest.approx(1.0)


def test_correlation_is_near_zero_for_unrelated_signals() -> None:
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 1, SR).astype(np.float32)
    assert abs(qa.correlation(_tone(400), noise)) < 0.2


def test_clipping_is_detected() -> None:
    clipped = np.ones(1000, dtype=np.float32)
    assert qa.clipping_ratio(clipped) == pytest.approx(1.0)
    assert qa.clipping_ratio(_tone(400)) == 0.0


# --------------------------------------------------------------------------- #
# Whole-pair verdict
# --------------------------------------------------------------------------- #
def test_a_proper_narrowband_pair_passes() -> None:
    clean = _wideband()
    channel = _tone(400)  # wideband content removed
    result = qa.measure_pair(clean, channel, SR, pair_id=1)
    assert result.band_limited is True


def test_a_chain_that_skipped_the_downsample_fails() -> None:
    # The exact failure this module exists to catch: quieter, still wideband.
    clean = _wideband()
    channel = (clean * 0.6).astype(np.float32)
    result = qa.measure_pair(clean, channel, SR, pair_id=2)
    assert result.band_limited is False
    assert result.hf_energy_channel > qa.MAX_HF_ENERGY_RATIO


def test_a_truncated_channel_copy_is_not_intact() -> None:
    clean = _tone(400, 2.0)
    result = qa.measure_pair(clean, clean[: len(clean) // 2], SR, pair_id=3)
    assert result.intact is False


def test_a_clipped_channel_copy_is_not_intact() -> None:
    clean = _tone(400)
    result = qa.measure_pair(clean, np.ones_like(clean), SR, pair_id=4)
    assert result.intact is False


def test_summary_flags_the_failing_pairs() -> None:
    good = qa.measure_pair(_wideband(), _tone(400), SR, pair_id=1)
    bad = qa.measure_pair(_wideband(), _wideband(), SR, pair_id=2)
    summary = qa.summarise([good, bad])
    assert summary["pairs"] == 2
    assert summary["all_band_limited"] is False
    assert summary["failures"] == [2]


def test_summary_of_nothing_is_not_a_pass() -> None:
    summary = qa.summarise([])
    assert summary["all_band_limited"] is False
    assert summary["all_intact"] is False


# --------------------------------------------------------------------------- #
# Codec delay alignment
# --------------------------------------------------------------------------- #
# G.711 hands back sample-synchronous audio; AMR-NB does not -- it runs a 20 ms
# frame plus a lookahead, so on the real corpus its output arrives ~4.8 ms (76-80
# samples at 16 kHz) late. Compared at offset 0 that reads as correlation ~0 and a
# negative SNR, and 15 of 20 pairs "failed" a chain that was working correctly.
def _delayed(signal: np.ndarray, samples: int) -> np.ndarray:
    """Shift ``signal`` later by ``samples``, keeping its length."""
    if samples >= 0:
        return np.concatenate([np.zeros(samples, dtype=signal.dtype), signal])[: signal.size]
    return np.concatenate([signal[-samples:], np.zeros(-samples, dtype=signal.dtype)])


def test_lag_of_an_unshifted_pair_is_zero() -> None:
    signal = _tone(400)
    assert qa.estimate_lag(signal, signal, 1600) == 0


def test_lag_recovers_a_known_delay() -> None:
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(SR).astype(np.float64)
    for delay in (37, 80, 160):
        assert qa.estimate_lag(signal, _delayed(signal, delay), 1600) == delay


def test_lag_recovers_a_negative_delay() -> None:
    rng = np.random.default_rng(1)
    signal = rng.standard_normal(SR).astype(np.float64)
    assert qa.estimate_lag(signal, _delayed(signal, -80), 1600) == -80


def test_alignment_restores_correlation_destroyed_by_delay() -> None:
    rng = np.random.default_rng(2)
    signal = rng.standard_normal(SR).astype(np.float64)
    delayed = _delayed(signal, 80)
    assert abs(qa.correlation(signal, delayed)) < 0.1  # the failure being fixed
    assert qa.correlation(*qa.align(signal, delayed, 1600)) > 0.99


def _narrowband_noise(seconds: float = 1.0, sr: int = SR, seed: int = 4) -> np.ndarray:
    """Noise low-passed below 4 kHz: band-limited like the channel output, and
    aperiodic, so a delay is unambiguous.

    A pure tone cannot stand in here. At 400 Hz its period is 40 samples, so a
    delay of 80 is exactly two periods and is genuinely indistinguishable from no
    delay at all -- the estimator is right to call it 0. Speech is not periodic
    like that, and neither is this.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    spectrum[np.fft.rfftfreq(n, 1.0 / sr) > 3500.0] = 0.0
    signal = np.fft.irfft(spectrum, n)
    return (0.3 * signal / np.max(np.abs(signal))).astype(np.float32)


def test_a_delayed_band_limited_copy_still_measures_as_intact() -> None:
    clean = _narrowband_noise()
    result = qa.measure_pair(clean, _delayed(clean, 80), SR, pair_id=5)
    assert result.lag_samples == 80
    assert result.band_limited is True
    assert result.intact is True
    assert result.correlation > 0.9  # would be ~0 without alignment


def test_alignment_never_rescues_genuinely_unrelated_audio() -> None:
    # A shift is forgiven; a different signal is not. Otherwise the metric would
    # pass anything by sliding it until something lined up.
    rng = np.random.default_rng(3)
    clean = rng.standard_normal(SR).astype(np.float64)
    unrelated = rng.standard_normal(SR).astype(np.float64)
    assert qa.correlation(*qa.align(clean, unrelated, 1600)) < 0.3
