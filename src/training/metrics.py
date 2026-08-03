"""Detection metrics: EER, ROC-AUC, F1, and bootstrap confidence intervals.

ASVspoof-standard reporting. Everything here is pure numpy (no sklearn), so the
module imports cheaply and runs in CI. Convention throughout:

- ``scores`` is a real-valued detector output where **higher = more bonafide**,
- ``labels`` is 0/1 with **1 = bonafide (positive)**, **0 = spoof (negative)**.

(If your detector emits P(spoof), pass ``-scores`` or ``1 - scores``.)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Equal Error Rate and its decision threshold.

    Returns ``(eer, threshold)`` where the false-accept and false-reject rates are
    closest. Vectorised (O(n log n)).
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pos = np.sort(scores[labels == 1])
    neg = np.sort(scores[labels == 0])
    if pos.size == 0 or neg.size == 0:
        raise ValueError("need both positive and negative examples for EER")

    thresholds = np.unique(np.concatenate([pos, neg]))
    # Decision: accept (predict bonafide) if score >= threshold.
    far = (neg.size - np.searchsorted(neg, thresholds, side="left")) / neg.size
    frr = np.searchsorted(pos, thresholds, side="left") / pos.size
    idx = int(np.argmin(np.abs(far - frr)))
    eer = float((far[idx] + frr[idx]) / 2.0)
    return eer, float(thresholds[idx])


def eer(scores: np.ndarray, labels: np.ndarray) -> float:
    """EER only (convenience for bootstrap)."""
    return compute_eer(scores, labels)[0]


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks with ties resolved by their average (like scipy rankdata)."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    i = 0
    n = values.size
    while i < n:
        j = i
        while j + 1 < n and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve via the rank statistic (exact, tie-aware)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("need both classes for AUC")
    ranks = _average_ranks(scores)
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Binary F1 for the positive (bonafide) class."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def f1_at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    """F1 when predicting bonafide for ``score >= threshold``."""
    preds = (np.asarray(scores, dtype=np.float64) >= threshold).astype(np.int64)
    return f1_score(labels, preds)


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    scores: np.ndarray,
    labels: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a ``metric_fn(scores, labels)``.

    Resamples clips with replacement ``n_boot`` times. Degenerate resamples
    (missing a class) are skipped.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n = scores.size
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            vals.append(metric_fn(scores[idx], labels[idx]))
        except ValueError:
            continue
    if not vals:
        raise ValueError("all bootstrap resamples were degenerate")
    lo = float(np.percentile(vals, 100 * alpha / 2))
    hi = float(np.percentile(vals, 100 * (1 - alpha / 2)))
    return lo, hi


def evaluate(
    scores: np.ndarray,
    labels: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """One-shot summary: EER, AUC, F1 (at the EER threshold), each with a 95% CI."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    eer_value, threshold = compute_eer(scores, labels)
    auc_value = roc_auc(scores, labels)
    f1_value = f1_at_threshold(scores, labels, threshold)

    eer_lo, eer_hi = bootstrap_ci(eer, scores, labels, n_boot, seed=seed)
    auc_lo, auc_hi = bootstrap_ci(roc_auc, scores, labels, n_boot, seed=seed)
    return {
        "eer": eer_value,
        "eer_ci_low": eer_lo,
        "eer_ci_high": eer_hi,
        "auc": auc_value,
        "auc_ci_low": auc_lo,
        "auc_ci_high": auc_hi,
        "f1": f1_value,
        "threshold": threshold,
    }
