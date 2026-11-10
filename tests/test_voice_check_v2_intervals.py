"""Voice check v2 with 95% intervals: the points match the rule's, the intervals behave."""

from __future__ import annotations

import pandas as pd
import pytest

from src.reporting import any_audio_accuracy as acc
from src.reporting import voice_check_v2_intervals as iv


def _scores(p, speakers, label="spoof", tool="indicf5"):
    return pd.DataFrame({"p": p, "speaker": speakers, "label": label, "tool": tool})


@pytest.fixture
def fake_sources(monkeypatch):
    sets = {
        "cm07": _scores([0.01, 0.02, 0.9, 0.01, 0.03, 0.01], ["a", "a", "b", "b", "t", "c"]),
        "cm09": _scores([0.01] * 4, ["a", "b", "c", "d"], tool="chatterbox"),
        "hiacc": pd.DataFrame({"p": [0.9, 0.8, 0.01], "label": "bonafide", "tool": "none"}),
    }
    monkeypatch.setattr(acc, "get", lambda model, key: sets.get(key))
    monkeypatch.setattr(acc, "threshold", lambda model: 0.5)
    return sets


def test_points_match_the_rule_and_training_speakers_are_left_out(fake_sources):
    r = iv.measure("m", "cm07", ("spoof", None), "spoof", frozenset({"t"}))
    assert r["n"] == 5 and r["speakers"] == 3 and r["method"] == "speakers"
    assert r["point"] == pytest.approx(80.0)
    low, high = r["interval"]
    assert 0 <= low <= 80.0 <= high <= 100


def test_all_correct_gets_an_exact_interval_and_no_speakers_gets_wilson(fake_sources):
    r = iv.measure("m", "cm09", ("spoof", None), "spoof", frozenset())
    assert r["method"] == "clips (exact)" and r["point"] == 100.0
    assert r["interval"][1] == 100.0 and r["interval"][0] < 100.0
    h = iv.measure("m", "hiacc", ("bonafide", None), "bonafide", frozenset())
    assert h["method"] == "clips (Wilson)"
    assert iv.measure("m", "cm04", ("spoof", None), "spoof", frozenset()) is None


def test_interval_helpers():
    assert iv.exact(12, 12) == (pytest.approx(73.535, abs=0.01), 100.0)
    low, high = iv.wilson(50, 100)
    assert low == pytest.approx(40.38, abs=0.01) and high == pytest.approx(59.62, abs=0.01)
    import numpy as np

    correct = np.array([1, 1, 0, 1, 0, 0], dtype=bool)
    speakers = np.array(["a", "a", "b", "c", "c", "d"])
    a = iv.speaker_interval(correct, speakers, n=2000)
    assert a == iv.speaker_interval(correct, speakers, n=2000)  # seeded
    assert 0 <= a[0] < 50 < a[1] <= 100


def test_cells_mark_the_interval_kind():
    r = {"point": 89.34, "interval": [83.2, 94.2], "method": "speakers"}
    assert iv._cell(r) == "89.3 [83.2, 94.2]"
    assert iv._cell({**r, "method": "calls (exact)"}).endswith(" c")
    assert iv._cell(None) == "-"
