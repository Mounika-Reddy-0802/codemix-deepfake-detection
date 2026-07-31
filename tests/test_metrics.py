"""Test stub (Week 2, owner M): metrics against hand-computed toy cases."""

import pytest


@pytest.mark.skip(reason="TODO(week-2, M): implement once metrics.py exists")
def test_eer_on_toy_scores() -> None:
    """EER on a tiny hand-checkable score/label set matches the known value."""


@pytest.mark.skip(reason="TODO(week-2, M): implement once metrics.py exists")
def test_bootstrap_ci_covers_point_estimate() -> None:
    """The bootstrap CI brackets the point estimate."""


def test_metrics_module_is_importable() -> None:
    """Smoke: the module exists and imports (no heavy deps at import time)."""
    import importlib

    mod = importlib.import_module("src.training.metrics")
    assert mod.__doc__ is not None
