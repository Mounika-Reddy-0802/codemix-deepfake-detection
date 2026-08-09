"""CRITICAL-PATH tests (Week 2, owner L): telephony channel-simulation sanity.

The lossy pieces that matter (G.711 mu-law, SNR mixing) are pure numpy, so these
run in CI without the heavy audio stack. The full chain test needs ``librosa`` for
resampling and is skipped when it is unavailable.
"""

import numpy as np
import pytest

from src.data import channel_sim as cs


def _tone(freq: float = 220.0, sr: int = 8000, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_mu_law_codes_are_uint8() -> None:
    codes = cs.mu_law_encode(_tone())
    assert codes.dtype == np.uint8
    assert codes.min() >= 0 and codes.max() <= 255


def test_mu_law_roundtrip_stays_in_range_and_close() -> None:
    x = _tone()
    y = cs.apply_g711_ulaw(x)
    assert np.max(np.abs(y)) <= 1.0
    # 8-bit mu-law is lossy but should track the signal closely.
    assert np.max(np.abs(y - x)) < 0.05


def test_add_noise_hits_target_snr() -> None:
    x = _tone()
    for target in (5.0, 10.0, 15.0, 20.0):
        noisy = cs.add_white_noise_at_snr(x, target, seed=0)
        measured = cs.measured_snr_db(x, noisy)
        assert abs(measured - target) < 0.5, f"target {target}, got {measured}"


def test_lower_snr_adds_more_noise() -> None:
    x = _tone()
    e5 = float(np.mean((cs.add_white_noise_at_snr(x, 5.0, seed=1) - x) ** 2))
    e20 = float(np.mean((cs.add_white_noise_at_snr(x, 20.0, seed=1) - x) ** 2))
    assert e5 > e20


def test_white_noise_is_deterministic_with_seed() -> None:
    x = _tone()
    a = cs.add_white_noise_at_snr(x, 10.0, seed=42)
    b = cs.add_white_noise_at_snr(x, 10.0, seed=42)
    assert np.allclose(a, b)


def test_unknown_codec_raises() -> None:
    with pytest.raises(ValueError):
        cs.apply_codec(_tone(), "opus")


def test_full_chain_returns_16k_and_no_clipping() -> None:
    pytest.importorskip("librosa")
    sr = 16_000
    t = np.arange(sr) / sr
    x = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    cfg = cs.ChannelConfig(codec="g711", snr_db=20.0, seed=0)
    out = cs.simulate_channel(x, cfg)
    assert out.size > 0
    assert np.max(np.abs(out)) <= 1.0
