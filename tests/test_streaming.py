"""Test stub (Week 5, owner SK): rolling-window streaming math."""

import pytest


@pytest.mark.skip(reason="TODO(week-5, SK): implement once streaming.py exists")
def test_window_count_on_synthetic_stream() -> None:
    """A 4 s window / 2 s hop over a fixed-length stream yields the expected
    number of windows."""


@pytest.mark.skip(reason="TODO(week-5, SK): implement once streaming.py exists")
def test_exponential_smoothing_is_stable() -> None:
    """Per-window verdicts are smoothed without oscillation."""


def test_streaming_module_is_importable() -> None:
    """Smoke: the module exists and imports (no heavy deps at import time)."""
    import importlib

    mod = importlib.import_module("src.inference.streaming")
    assert mod.__doc__ is not None
