"""Tests for false alarms on genuine-only speech (src/training/false_alarm.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.training import false_alarm as fa


def _scores(rates: dict[str, float], per_speaker: int = 50, seed: int = 0) -> pd.DataFrame:
    """Genuine clips where speaker s is called fake with probability rates[s]."""
    rng = np.random.default_rng(seed)
    rows = []
    for speaker, rate in rates.items():
        flagged = rng.random(per_speaker) < rate
        for f in flagged:
            rows.append({"speaker": speaker, "label": "bonafide", "score": 0.1 if f else 0.9})
    return pd.DataFrame(rows)


def test_rate_counts_scores_below_the_threshold():
    assert fa.false_alarm_rate(np.array([0.1, 0.4, 0.6, 0.9])) == 0.5
    assert fa.false_alarm_rate(np.array([0.5]), threshold=0.5) == 0.0  # at threshold = genuine


def test_spoof_rows_are_ignored():
    df = pd.DataFrame(
        {
            "speaker": ["a", "a", "a"],
            "label": ["bonafide", "spoof", "spoof"],
            "score": [0.9, 0.1, 0.1],
        }
    )
    assert fa.summarise(df)["false_alarm_rate"] == 0.0
    assert fa.summarise(df)["clips"] == 1


def test_summary_interval_contains_the_rate_and_is_speaker_level():
    df = _scores({f"s{i}": r for i, r in enumerate([0.0, 0.0, 0.05, 0.1, 0.4, 0.0])})
    s = fa.summarise(df, n_boot=500)
    assert s["false_alarm_ci"][0] <= s["false_alarm_rate"] <= s["false_alarm_ci"][1]
    assert s["speakers"] == 6 and s["clips"] == 300
    # one bad speaker drives the spread: the interval must be wide, not clip-level narrow
    assert s["false_alarm_ci"][1] - s["false_alarm_ci"][0] > 0.05
    assert s["speakers_above_10pct"] == 1


def test_per_speaker_sorts_worst_first():
    df = _scores({"good": 0.0, "bad": 1.0, "mid": 0.5}, per_speaker=40, seed=3)
    table = fa.per_speaker(df)
    assert table.iloc[0]["speaker"] == "bad"
    assert table.iloc[0]["false_alarm_rate"] == 1.0
    assert set(table.columns) == {"speaker", "clips", "false_alarms", "false_alarm_rate"}


def test_compare_detects_a_real_difference_and_not_a_fake_one():
    worse = _scores({f"t{i}": 0.3 for i in range(10)}, seed=1)
    base = _scores({f"r{i}": 0.02 for i in range(10)}, seed=2)
    same = _scores({f"q{i}": 0.02 for i in range(10)}, seed=4)
    d = fa.compare(worse, base, n_boot=400)
    assert d["difference_ci"][0] > 0
    d0 = fa.compare(same, base, n_boot=400)
    assert d0["difference_ci"][0] < 0 < d0["difference_ci"][1] or d0["difference_ci"] == [0.0, 0.0]


def test_inputs_without_genuine_rows_or_columns_are_refused():
    with pytest.raises(ValueError, match="no genuine"):
        fa.summarise(pd.DataFrame({"speaker": ["a"], "label": ["spoof"], "score": [0.2]}))
    with pytest.raises(ValueError, match="missing columns"):
        fa.summarise(pd.DataFrame({"score": [0.2]}))


def test_paired_comparison_finds_a_drop_on_the_same_clips():
    worse = _scores({"a": 0.6, "b": 0.6, "c": 0.6}, seed=1)
    worse["filepath"] = [f"clip{i}" for i in range(len(worse))]
    better = worse.copy()
    better["score"] = np.where(np.arange(len(better)) % 2 == 0, 0.9, better["score"])
    out = fa.compare_paired(better, worse, n_boot=300)
    assert out["difference"] < 0 and out["difference_ci"][1] < 0
    assert out["speakers_lower"] == 3 and out["clips"] == len(worse)
    same = fa.compare_paired(worse, worse, n_boot=200)
    assert same["difference"] == 0.0 and same["speakers_lower"] == 0


def test_paired_comparison_refuses_different_clips():
    a = _scores({"a": 0.5})
    a["filepath"] = [f"clip{i}" for i in range(len(a))]
    with pytest.raises(ValueError, match="different clips"):
        fa.compare_paired(a, a.iloc[:-1])
