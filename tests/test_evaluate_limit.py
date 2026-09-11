"""``--limit`` must keep both classes, or a smoke run cannot compute EER."""

from __future__ import annotations

import pandas as pd

from src.training.evaluate import limit_rows


def _manifest(n_bona: int, n_spoof: int, bona_first: bool = True) -> pd.DataFrame:
    rows = [{"filepath": f"b{i}.wav", "label": "bonafide"} for i in range(n_bona)]
    spoof = [{"filepath": f"s{i}.wav", "label": "spoof"} for i in range(n_spoof)]
    return pd.DataFrame(rows + spoof if bona_first else spoof + rows)


def test_limit_keeps_both_classes_when_manifest_is_grouped_by_label():
    df = _manifest(1566, 2400)
    out = limit_rows(df, 200)
    assert len(out) == 200
    assert set(out["label"]) == {"bonafide", "spoof"}


def test_limit_keeps_class_shares_and_is_deterministic():
    df = _manifest(302, 317, bona_first=False)
    a, b = limit_rows(df, 200), limit_rows(df, 200)
    assert a.equals(b)
    counts = a["label"].value_counts()
    assert abs(counts["bonafide"] - 98) <= 1 and abs(counts["spoof"] - 102) <= 1


def test_limit_keeps_a_minority_class_of_one():
    df = _manifest(999, 1)
    out = limit_rows(df, 10)
    assert (out["label"] == "spoof").sum() == 1
    assert len(out) == 10


def test_limit_larger_than_manifest_returns_everything():
    df = _manifest(3, 2)
    assert len(limit_rows(df, 50)) == 5
