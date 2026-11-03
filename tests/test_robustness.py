"""Tests for src/data/robustness.py: the sweep's transforms on synthetic audio."""

import shutil

import numpy as np
import pandas as pd
import pytest

from src.data import robustness as rb


def _tone(seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(seconds * rb.SR)) / rb.SR
    return (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_the_list_is_fixed_and_starts_with_the_reference() -> None:
    assert rb.TRANSFORMS[0] == "none"
    assert len(rb.TRANSFORMS) == len(set(rb.TRANSFORMS)) == 15


def test_noise_lands_at_the_requested_snr() -> None:
    clean = _tone()
    noisy = rb.transform("white_10db", clean, np.random.default_rng(0))
    noise = noisy - clean
    snr = 10 * np.log10(
        np.mean(clean.astype(np.float64) ** 2) / np.mean(noise.astype(np.float64) ** 2)
    )
    assert snr == pytest.approx(10.0, abs=0.01)


def test_babble_mixes_four_pool_clips_and_needs_them() -> None:
    rng = np.random.default_rng(0)
    pool = [rng.standard_normal(8000).astype(np.float32) for _ in range(6)]
    out = rb.transform("babble_20db", _tone(), np.random.default_rng(1), pool)
    assert out.shape == _tone().shape
    with pytest.raises(ValueError):
        rb.transform("babble_20db", _tone(), np.random.default_rng(1), pool[:3])


def test_room_response_has_unit_energy_and_keeps_length() -> None:
    response = rb.room_response(np.random.default_rng(0))
    assert np.sum(response**2) == pytest.approx(1.0)
    assert len(response) == int(rb.RT60_S * rb.SR) + 1
    assert rb.transform("reverb_0.4s", _tone(), np.random.default_rng(0)).shape == _tone().shape


def test_speed_changes_length_and_pitch_does_not() -> None:
    pytest.importorskip("librosa")  # resampling wheel, not installed in CI
    clean = _tone(2.0)
    rng = np.random.default_rng(0)
    assert len(rb.transform("speed_1.05", clean, rng)) == pytest.approx(len(clean) / 1.05, rel=0.01)
    assert len(rb.transform("speed_0.95", clean, rng)) == pytest.approx(len(clean) / 0.95, rel=0.01)
    assert len(rb.transform("pitch_+1", clean, rng)) == len(clean)


def test_render_is_reproducible_and_double_line_differs() -> None:
    pytest.importorskip("librosa")  # the line resamples; not installed in CI
    clean = _tone()
    a = rb.render("white_20db", clean, seed=7)
    b = rb.render("white_20db", clean, seed=7)
    assert np.array_equal(a, b)
    assert not np.array_equal(rb.render("none", clean, 7), rb.render("double_g711", clean, 7))


def test_unknown_transform_is_refused() -> None:
    with pytest.raises(ValueError):
        rb.transform("chorus", _tone(), np.random.default_rng(0))


def test_fit_length_trims_and_pads() -> None:
    assert len(rb.fit_length(np.ones(10), 6)) == 6
    padded = rb.fit_length(np.ones(4), 6)
    assert len(padded) == 6 and padded[-1] == 0.0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
@pytest.mark.parametrize("name", ["mp3_32k", "opus_12k", "aac_64k"])
def test_codec_round_trips_keep_length(name) -> None:
    pytest.importorskip("soundfile")  # audio wheel, not installed in CI
    out = rb.transform(name, _tone(), np.random.default_rng(0))
    assert out.shape == _tone().shape
    assert np.corrcoef(out[2000:-2000], _tone()[2000:-2000])[0, 1] > 0.5


def test_sweep_sample_is_fixed_and_tagged() -> None:
    groups = {
        "genuine": pd.DataFrame({"filepath": [f"g{i}" for i in range(50)]}),
        "tortoise": pd.DataFrame({"filepath": [f"t{i}" for i in range(5)]}),
    }
    a = rb.sweep_manifest(groups, per_group=10)
    assert a["group"].value_counts().to_dict() == {"genuine": 10, "tortoise": 5}
    assert a["filepath"].tolist() == rb.sweep_manifest(groups, per_group=10)["filepath"].tolist()
