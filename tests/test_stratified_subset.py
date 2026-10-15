"""Tests for the fixed attack-balanced sample (src/data/stratified_subset.py)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.stratified_subset import stratified_subset


def _manifest() -> pd.DataFrame:
    rows = [
        {"filepath": f"b{i}", "label": "bonafide", "tool": "none", "speaker": f"s{i % 5}"}
        for i in range(50)
    ]
    for attack, n in (("A07", 40), ("A08", 40), ("A09", 3)):
        rows += [
            {"filepath": f"{attack}_{i}", "label": "spoof", "tool": attack, "speaker": f"s{i % 5}"}
            for i in range(n)
        ]
    return pd.DataFrame(rows)


def test_counts_per_class_and_attack():
    sub = stratified_subset(_manifest(), n_bonafide=10, n_per_attack=5)
    assert (sub["label"] == "bonafide").sum() == 10
    counts = sub[sub["label"] == "spoof"].groupby("tool").size().to_dict()
    assert counts == {"A07": 5, "A08": 5, "A09": 3}  # a small attack gives all it has


def test_same_seed_same_sample_and_different_seed_differs():
    a = stratified_subset(_manifest(), 10, 5, seed=1)
    b = stratified_subset(_manifest(), 10, 5, seed=1)
    c = stratified_subset(_manifest(), 10, 5, seed=2)
    assert a["filepath"].tolist() == b["filepath"].tolist()
    assert set(a["filepath"]) != set(c["filepath"])


def test_no_row_is_drawn_twice():
    sub = stratified_subset(_manifest(), 50, 40)
    assert sub["filepath"].is_unique
    assert len(sub) == 50 + 40 + 40 + 3


def test_needs_both_classes():
    only_genuine = _manifest()[lambda d: d["label"] == "bonafide"]
    with pytest.raises(ValueError):
        stratified_subset(only_genuine)
