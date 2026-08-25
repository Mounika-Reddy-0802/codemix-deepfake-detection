"""Unit tests for the W5-T4 low-level-cue shortcut check.

Numpy only -- no audio, no torch -- so CI exercises the exact code that produces the
gate number rather than a dependency that CI never installs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import lowlevel_cue as lc

SR = 16_000


def _noise(n: int = SR, scale: float = 0.1, seed: int = 0) -> np.ndarray:
    return (np.random.default_rng(seed).standard_normal(n) * scale).astype(np.float32)


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def test_features_returns_all_eight_in_order():
    out = lc.features(_noise(), SR)
    assert out.shape == (len(lc.FEATURE_NAMES),)
    assert np.isfinite(out).all()


def test_features_of_an_empty_clip_are_zero_not_nan():
    """A zero-length clip must not poison the feature matrix with NaNs."""
    out = lc.features(np.array([], dtype=np.float32), SR)
    assert out.shape == (len(lc.FEATURE_NAMES),)
    assert np.isfinite(out).all()
    assert (out == 0).all()


def test_features_of_silence_are_finite():
    out = lc.features(np.zeros(SR, dtype=np.float32), SR)
    assert np.isfinite(out).all()


def test_loudness_and_clipping_are_picked_up():
    idx = {name: i for i, name in enumerate(lc.FEATURE_NAMES)}
    quiet = lc.features(_noise(scale=0.01), SR)
    loud = lc.features(np.clip(_noise(scale=5.0), -1.0, 1.0), SR)

    assert loud[idx["rms_mean"]] > quiet[idx["rms_mean"]]
    assert loud[idx["clipping_ratio"]] > quiet[idx["clipping_ratio"]]
    assert quiet[idx["clipping_ratio"]] == pytest.approx(0.0)


def test_dc_offset_is_measured():
    idx = lc.FEATURE_NAMES.index("dc_offset")
    centred = lc.features(_noise(), SR)
    shifted = lc.features(_noise() + 0.3, SR)
    assert abs(centred[idx]) < 0.01
    assert shifted[idx] == pytest.approx(0.3, abs=0.02)


def test_rolloff_and_hf_ratio_track_bandwidth():
    """A band-limited clip must show a lower rolloff and less energy above 4 kHz."""
    idx = {name: i for i, name in enumerate(lc.FEATURE_NAMES)}
    t = np.arange(SR, dtype=np.float64) / SR
    low = lc.features((0.4 * np.sin(2 * np.pi * 300 * t)).astype(np.float32), SR)
    high = lc.features((0.4 * np.sin(2 * np.pi * 6_000 * t)).astype(np.float32), SR)

    assert low[idx["spectral_rolloff_hz"]] < high[idx["spectral_rolloff_hz"]]
    assert low[idx["hf_energy_ratio"]] < 0.01
    assert high[idx["hf_energy_ratio"]] > 0.9


def test_zero_crossing_rate_rises_with_frequency():
    idx = lc.FEATURE_NAMES.index("zero_crossing_rate")
    t = np.arange(SR, dtype=np.float64) / SR
    slow = lc.features((0.4 * np.sin(2 * np.pi * 100 * t)).astype(np.float32), SR)
    fast = lc.features((0.4 * np.sin(2 * np.pi * 4_000 * t)).astype(np.float32), SR)
    assert fast[idx] > slow[idx]


# --------------------------------------------------------------------------- #
# Standardiser
# --------------------------------------------------------------------------- #
def test_standardiser_centres_and_scales():
    x = np.random.default_rng(0).normal(loc=5.0, scale=2.0, size=(200, 3))
    z = lc.Standardiser().fit(x).transform(x)
    assert np.allclose(z.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(z.std(axis=0), 1.0, atol=1e-9)


def test_standardiser_survives_a_constant_column():
    """A feature that never varies must not divide by zero into NaNs."""
    x = np.column_stack([np.random.default_rng(0).normal(size=50), np.full(50, 3.0)])
    z = lc.Standardiser().fit(x).transform(x)
    assert np.isfinite(z).all()


def test_standardiser_refuses_to_transform_before_fit():
    with pytest.raises(RuntimeError):
        lc.Standardiser().transform(np.zeros((2, 2)))


def test_standardiser_uses_train_statistics_on_test():
    """Re-fitting on test would leak its distribution into the check."""
    train = np.array([[0.0], [2.0]])
    scaler = lc.Standardiser().fit(train)
    assert scaler.transform(np.array([[1.0]]))[0, 0] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Logistic regression
# --------------------------------------------------------------------------- #
def test_sigmoid_is_stable_at_extremes():
    out = lc._sigmoid(np.array([-2000.0, 0.0, 2000.0]))
    assert np.isfinite(out).all()
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(0.5)
    assert out[2] == pytest.approx(1.0)


def test_logistic_learns_a_separable_problem():
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(-2.0, 0.5, (200, 2)), rng.normal(2.0, 0.5, (200, 2))])
    y = np.array([0] * 200 + [1] * 200)
    w, b = lc.fit_logistic(x, y)
    assert (lc.predict(x, w, b) > 0.5).astype(int).tolist() == y.tolist()


def test_logistic_stays_near_chance_on_pure_noise():
    """The outcome the gate wants: features carrying no class information.

    This is the property the whole module is built to detect, so it is asserted
    rather than assumed.
    """
    from src.training.metrics import eer

    rng = np.random.default_rng(0)
    x = rng.normal(size=(600, 8))
    y = rng.integers(0, 2, size=600)
    w, b = lc.fit_logistic(x, y)
    assert abs(eer(lc.predict(x, w, b), y) - 0.5) < 0.12


def test_class_weighting_does_not_collapse_on_an_imbalanced_fit():
    """With 90% one class, an unweighted fit can win by ignoring the features."""
    rng = np.random.default_rng(1)
    x = np.vstack([rng.normal(-1.5, 0.5, (450, 2)), rng.normal(1.5, 0.5, (50, 2))])
    y = np.array([0] * 450 + [1] * 50)
    w, b = lc.fit_logistic(x, y, class_weighted=True)
    predicted = (lc.predict(x, w, b) > 0.5).astype(int)
    assert predicted.sum() > 25  # the rare class is still being predicted


# --------------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------------- #
def test_verdict_passes_near_chance():
    assert lc.verdict(0.5).startswith("PASS")
    assert lc.verdict(0.5316).startswith("PASS")  # AffectDF's own Appendix-G number


def test_verdict_fails_a_separable_result():
    assert lc.verdict(0.05).startswith("FAIL")


def test_verdict_fails_an_inverted_classifier_too():
    """A reliably wrong classifier is just as separable -- flip its sign."""
    assert lc.verdict(0.95).startswith("FAIL")


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #
def test_run_check_flags_a_loudness_shortcut(tmp_path):
    """A corpus whose classes differ only in level must FAIL the gate."""
    soundfile = pytest.importorskip("soundfile")
    clips = tmp_path / "clips"
    clips.mkdir()
    rows = []
    for i in range(40):
        bona = i % 2 == 0
        # The only difference between the classes is amplitude.
        audio = _noise(scale=0.30 if bona else 0.03, seed=i)
        soundfile.write(str(clips / f"c{i}.wav"), audio, SR)
        rows.append(
            {
                "filepath": f"${{DATA_ROOT}}/clips/c{i}.wav",
                "label": "bonafide" if bona else "spoof",
            }
        )
    manifest = pd.DataFrame(rows)

    result = lc.run_check(manifest, manifest, str(tmp_path), str(tmp_path))
    assert result["test_clips"] == 40
    assert set(result["coefficients"]) == set(lc.FEATURE_NAMES)
    assert result["eer"] < 0.2
    assert lc.verdict(result["eer"]).startswith("FAIL")
