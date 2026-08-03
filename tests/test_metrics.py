"""Tests for detection metrics against hand-computable cases (Week 2, owner M)."""

import numpy as np

from src.training import metrics as m


def test_eer_perfect_separation_is_zero() -> None:
    scores = np.array([0.9, 0.8, 0.85, 0.1, 0.2, 0.15])
    labels = np.array([1, 1, 1, 0, 0, 0])
    eer, _ = m.compute_eer(scores, labels)
    assert eer == 0.0


def test_eer_random_is_near_half() -> None:
    rng = np.random.default_rng(0)
    scores = rng.random(2000)
    labels = rng.integers(0, 2, 2000)
    eer, _ = m.compute_eer(scores, labels)
    assert 0.4 < eer < 0.6


def test_auc_perfect_and_inverse() -> None:
    scores = np.array([0.9, 0.8, 0.1, 0.2])
    labels = np.array([1, 1, 0, 0])
    assert m.roc_auc(scores, labels) == 1.0
    assert m.roc_auc(-scores, labels) == 0.0


def test_auc_half_on_constant_scores() -> None:
    scores = np.full(10, 0.5)
    labels = np.array([1, 0] * 5)
    assert abs(m.roc_auc(scores, labels) - 0.5) < 1e-9


def test_f1_known_case() -> None:
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0])  # tp=1, fp=0, fn=1 -> P=1, R=0.5, F1=2/3
    assert abs(m.f1_score(y_true, y_pred) - (2 / 3)) < 1e-9


def test_missing_class_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        m.compute_eer(np.array([0.1, 0.2]), np.array([1, 1]))


def test_bootstrap_ci_brackets_point_estimate() -> None:
    rng = np.random.default_rng(1)
    scores = np.concatenate([rng.normal(1.0, 1.0, 300), rng.normal(-1.0, 1.0, 300)])
    labels = np.concatenate([np.ones(300, int), np.zeros(300, int)])
    point = m.roc_auc(scores, labels)
    lo, hi = m.bootstrap_ci(m.roc_auc, scores, labels, n_boot=200, seed=0)
    assert lo <= point <= hi
    assert lo < hi


def test_evaluate_summary_keys() -> None:
    rng = np.random.default_rng(2)
    scores = np.concatenate([rng.normal(1, 1, 200), rng.normal(-1, 1, 200)])
    labels = np.concatenate([np.ones(200, int), np.zeros(200, int)])
    out = m.evaluate(scores, labels, n_boot=100)
    for key in ("eer", "auc", "f1", "eer_ci_low", "auc_ci_high", "threshold"):
        assert key in out
