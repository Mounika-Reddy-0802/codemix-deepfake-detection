"""Tests for reproducible seeding (Week 3, owner M)."""

import numpy as np

from src.utils.seed import set_seed


def test_set_seed_returns_seed() -> None:
    assert set_seed(7) == 7


def test_numpy_is_reproducible_after_seed() -> None:
    set_seed(123)
    a = np.random.rand(5)
    set_seed(123)
    b = np.random.rand(5)
    assert np.allclose(a, b)


def test_python_random_is_reproducible_after_seed() -> None:
    import random

    set_seed(99)
    a = [random.random() for _ in range(5)]
    set_seed(99)
    b = [random.random() for _ in range(5)]
    assert a == b
