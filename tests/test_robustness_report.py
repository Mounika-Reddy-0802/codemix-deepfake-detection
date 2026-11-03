"""Tests for src/reporting/robustness.py on synthetic score files."""

import numpy as np
import pandas as pd
import pytest

from src.reporting import robustness as rr


def _scores(genuine, xtts, tortoise) -> pd.DataFrame:
    rows = [("genuine", s) for s in genuine] + [("xtts_v2", s) for s in xtts]
    rows += [("tortoise", s) for s in tortoise]
    return pd.DataFrame(rows, columns=["group", "score"])


def test_measure_reports_eer_and_rule_errors_per_tool() -> None:
    m = rr.measure(_scores([0.9, 0.8, 0.4, 0.95], [0.1, 0.2, 0.6, 0.05], [0.85, 0.9, 0.92, 0.2]))
    assert m["false_alarm"] == pytest.approx(25.0)
    assert m["miss_xtts_v2"] == pytest.approx(25.0)
    assert m["miss_tortoise"] == pytest.approx(75.0)
    assert 0.0 <= m["eer_xtts_v2"] < m["eer_tortoise"]
    assert "eer_indicf5" not in m  # a tool with no clips is left out


def test_table_gives_changes_from_the_reference(tmp_path) -> None:
    rng = np.random.default_rng(0)
    clean = _scores(rng.uniform(0.7, 1, 50), rng.uniform(0, 0.3, 50), rng.uniform(0, 0.3, 50))
    worse = _scores(rng.uniform(0.3, 1, 50), rng.uniform(0, 0.8, 50), rng.uniform(0, 0.8, 50))
    clean.to_csv(tmp_path / "sweep_none_scores.csv", index=False)
    worse.to_csv(tmp_path / "sweep_mp3_32k_scores.csv", index=False)
    frame = rr.table(tmp_path).set_index("transform")
    assert list(frame.index) == ["none", "mp3_32k"]
    assert frame.loc["none", "delta_eer_xtts_v2"] == 0.0
    assert frame.loc["mp3_32k", "delta_eer_xtts_v2"] > 0.0
    text = rr.markdown(frame.reset_index(), "Model X")
    assert "| mp3_32k |" in text and "(+" in text


def test_missing_reference_is_an_error(tmp_path) -> None:
    _scores([0.9], [0.1], [0.2]).to_csv(tmp_path / "sweep_mp3_32k_scores.csv", index=False)
    with pytest.raises(ValueError, match="reference"):
        rr.table(tmp_path)
