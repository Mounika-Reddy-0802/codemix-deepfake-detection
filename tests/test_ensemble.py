"""Tests for src/inference/ensemble.py: averaging and re-smoothing two window files."""

import numpy as np
import pandas as pd
import pytest

from src.inference import ensemble


def _windows(scores, path="c1.wav") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "filepath": path,
            "label": "bonafide",
            "tool": "none",
            "window": range(len(scores)),
            "end_seconds": [4.0 + 2 * i for i in range(len(scores))],
            "level_dbfs": -23.0,
            "score": scores,
            "smoothed": scores,  # not used: the ensemble re-smooths from the scores
        }
    )


def test_scores_are_averaged_and_resmoothed_like_the_live_path() -> None:
    out = ensemble.average_windows(_windows([1.0, 0.0, 1.0]), _windows([0.0, 0.0, 1.0]))
    assert out["score"].tolist() == [0.5, 0.0, 1.0]
    # live moving average, alpha 0.5: 0.5, then 0.25, then 0.625
    assert out["smoothed"].tolist() == pytest.approx([0.5, 0.25, 0.625])
    assert list(out.columns) == list(_windows([1.0]).columns)


def test_skipped_windows_stay_skipped() -> None:
    out = ensemble.average_windows(_windows([0.8, np.nan, 0.4]), _windows([0.6, np.nan, 0.2]))
    assert np.isnan(out["score"].iloc[1]) and np.isnan(out["smoothed"].iloc[1])
    assert out["smoothed"].iloc[2] == pytest.approx(0.5 * 0.3 + 0.5 * 0.7)


def test_files_from_different_audio_are_refused() -> None:
    with pytest.raises(ValueError, match="skip different"):
        ensemble.average_windows(_windows([0.8, np.nan]), _windows([0.6, 0.5]))
    with pytest.raises(ValueError, match="same windows"):
        ensemble.average_windows(_windows([0.8, 0.5]), _windows([0.6, 0.5], path="c2.wav"))
    with pytest.raises(ValueError, match="length"):
        ensemble.average_windows(_windows([0.8, 0.5]), _windows([0.6]))


def _clips(scores, paths=("a.wav", "b.wav")) -> pd.DataFrame:
    return pd.DataFrame({"filepath": list(paths), "label": "bonafide", "score": scores})


def test_clip_log_odds_are_averaged_as_probabilities() -> None:
    out = ensemble.average_clips(_clips([0.0, 2.0]), _clips([0.0, -2.0], ("a.wav", "b.wav")))
    assert out["score"].tolist() == pytest.approx([0.5, 0.5])
    assert list(out.columns) == ["filepath", "label", "score"]
    # the order of b does not matter: clips are joined on their path
    out = ensemble.average_clips(_clips([10.0, 0.0]), _clips([0.0, 10.0], ("b.wav", "a.wav")))
    assert out["filepath"].tolist() == ["a.wav", "b.wav"]
    assert out["score"].tolist() == pytest.approx([1 / (1 + np.exp(-10.0)), 0.5])


def test_clip_files_with_different_clips_are_refused() -> None:
    with pytest.raises(ValueError, match="same clips"):
        ensemble.average_clips(_clips([0.0, 1.0]), _clips([0.0, 1.0], ("a.wav", "c.wav")))
    with pytest.raises(ValueError, match="same clips"):
        ensemble.average_clips(_clips([0.0, 1.0]), _clips([0.0], ("a.wav",)))
