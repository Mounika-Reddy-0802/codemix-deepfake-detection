"""File prediction (owner SK): threshold precedence, verdicts, window aggregation."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.inference import predict as pr

SR = pr.SAMPLE_RATE


def _speech(seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.1 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)


T = pr.Threshold(0.9, "test", provisional=True)


def test_explicit_threshold_wins(tmp_path) -> None:
    (tmp_path / "t.yaml").write_text("threshold: 0.7\n")
    th = pr.resolve_threshold(0.8, str(tmp_path / "t.yaml"))
    assert th.value == 0.8 and th.provisional


def test_frozen_file_beats_results_json(tmp_path) -> None:
    (tmp_path / "t.yaml").write_text("threshold: 0.7\n")
    (tmp_path / "r.json").write_text(json.dumps({"pooled": {"threshold": 0.99}}))
    th = pr.resolve_threshold(None, str(tmp_path / "t.yaml"), str(tmp_path / "r.json"))
    assert th.value == 0.7 and not th.provisional


def test_results_json_is_provisional(tmp_path) -> None:
    (tmp_path / "r.json").write_text(json.dumps({"eer": 0.04, "threshold": 0.96}))
    th = pr.resolve_threshold(None, str(tmp_path / "absent.yaml"), str(tmp_path / "r.json"))
    assert th.value == 0.96 and th.provisional


def test_no_threshold_refuses_rather_than_using_half(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="results freeze"):
        pr.resolve_threshold(None, str(tmp_path / "absent.yaml"))


def test_verdict_convention_matches_metrics() -> None:
    assert pr.verdict_for(0.9, T) == "bonafide"  # at the threshold: accepted
    assert pr.verdict_for(0.89, T) == "spoof"
    assert pr.verdict_for(None, T) == "no speech"


def test_file_score_is_the_worst_window() -> None:
    scores = iter([0.99, 0.2, 0.99, 0.99])
    pred = pr.predict_audio(_speech(10.0), lambda w: next(scores), T)
    assert pred.windows == 4 and pred.score == pytest.approx(0.2)
    assert pred.verdict == "spoof"


def test_short_clip_still_gets_one_window() -> None:
    pred = pr.predict_audio(_speech(1.5), lambda w: 0.95, T)
    assert pred.windows == 1 and pred.verdict == "bonafide"


def test_silent_file_is_no_speech() -> None:
    pred = pr.predict_audio(np.zeros(6 * SR, dtype=np.float32), lambda w: 0.0, T)
    assert pred.verdict == "no speech" and pred.skipped == pred.windows


def test_prediction_dict_flags_provisional() -> None:
    out = pr.predict_audio(_speech(4.0), lambda w: 0.95, T).as_dict()
    assert out["provisional"] is True and out["threshold"] == 0.9
