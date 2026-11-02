"""Tests for src/reporting/det_curves.py on small synthetic score files (no D: data)."""

import numpy as np
import pandas as pd
import pytest

from src.reporting import det_curves as dc
from src.training.metrics import eer


def test_log_odds_become_probabilities_and_probabilities_stay() -> None:
    assert dc.probability(np.array([0.1, 0.9])).tolist() == [0.1, 0.9]
    p = dc.probability(np.array([-2.0, 0.0, 3.0]))
    assert p[1] == pytest.approx(0.5)
    assert p[0] < 0.5 < p[2]


def test_rule_counts_false_alarms_and_misses() -> None:
    labels = np.array([1, 1, 1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.4, 0.7, 0.2, 0.6])  # one genuine low, one fake high
    m = dc.clip_metrics(scores, labels)
    assert m["false_alarm"] == pytest.approx(25.0)
    assert m["miss"] == pytest.approx(50.0)
    assert m["accuracy"] == pytest.approx(100 * 4 / 6)
    assert m["eer"] == pytest.approx(eer(scores, labels) * 100)


def test_log_odds_and_probabilities_give_the_same_decisions() -> None:
    rng = np.random.default_rng(0)
    logits = rng.normal(size=200)
    labels = (rng.random(200) < 0.5).astype(int)
    a = dc.clip_metrics(logits, labels)
    b = dc.clip_metrics(1 / (1 + np.exp(-logits)), labels)
    for key in ("eer", "accuracy", "false_alarm", "miss", "brier"):
        assert a[key] == pytest.approx(b[key])


def test_perfectly_calibrated_scores_have_no_calibration_error() -> None:
    p = np.repeat([0.05, 0.95], 100)
    labels = np.concatenate([np.r_[np.ones(5), np.zeros(95)], np.r_[np.ones(95), np.zeros(5)]])
    assert dc.calibration_error(p, labels) == pytest.approx(0.0)
    assert dc.calibration_error(np.full(200, 0.95), labels) == pytest.approx(0.45)


def test_genuine_only_set_has_no_eer_or_miss() -> None:
    m = dc.clip_metrics(np.array([0.9, 0.3, 0.8]), np.array([1, 1, 1]))
    assert m["eer"] is None and m["miss"] is None
    assert m["false_alarm"] == pytest.approx(100 / 3)


def _write(path, labels, scores) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"label": ["bonafide" if y else "spoof" for y in labels], "score": scores}).to_csv(
        path, index=False
    )


def test_files_are_found_measured_and_summarised(tmp_path) -> None:
    labels = [1, 1, 0, 0]
    folder = tmp_path / "final_x" / "final_results"
    _write(folder / "final_s1__cm07_scores.csv", labels, [0.9, 0.6, 0.4, 0.1])
    _write(folder / "final_s2__cm07_scores.csv", labels, [0.9, 0.3, 0.6, 0.1])
    _write(folder / "deployed_s1__cm07_scores.csv", labels, [0.9, 0.8, 0.2, 0.1])
    _write(folder / "final_s1__unknown_set_scores.csv", labels, [0.9, 0.8, 0.2, 0.1])
    files = dc.score_files(tmp_path, sources=dc.SOURCES[:2])
    assert sorted(zip(files["system"], files["seed"], strict=True)) == [
        ("Deployed adapter", 1),
        ("Final adapter", 1),
        ("Final adapter", 2),
    ]
    summary = dc.summarise(dc.measure(files)).set_index("system")
    assert summary.loc["Final adapter", "seeds"] == 2
    assert summary.loc["Final adapter", "eer_mean"] == pytest.approx(25.0)
    assert summary.loc["Deployed adapter", "eer_mean"] == pytest.approx(0.0)
    assert "IndicF5, G.711" in dc.markdown(dc.summarise(dc.measure(files)))
