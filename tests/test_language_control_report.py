"""Smoke tests for src.reporting.language_control."""

import pandas as pd

from src.reporting import language_control as lc


def _scores():
    rows = []
    for speaker in ("a", "b", "c"):
        rows += [
            {"speaker": speaker, "language_condition": "all English", "score": 0.1},
            {"speaker": speaker, "language_condition": "all Hindi", "score": 0.9},
            {"speaker": speaker, "language_condition": "mixed", "score": 0.9},
        ]
    df = pd.DataFrame(rows)
    df["false_alarm"] = df["score"] < lc.THRESHOLD
    return df


def test_condition_rates():
    rates = lc.condition_rates(_scores()).set_index("condition")
    assert rates.loc["all English", "false_alarm_rate"] == 1.0
    assert rates.loc["mixed", "false_alarm_rate"] == 0.0
    assert rates.loc["mixed", "speakers"] == 3


def test_paired_differences_are_within_speaker():
    paired = lc.paired_differences(_scores()).set_index("comparison")
    assert paired.loc["mixed - all English", "mean_difference"] == -1.0
    assert paired.loc["mixed - all English", "ci_high"] == -1.0


def test_markdown_lists_each_model(tmp_path):
    df = _scores()
    df["label"] = "bonafide"
    df.drop(columns="false_alarm").to_csv(tmp_path / "s1__clean_scores.csv", index=False)
    text = lc.markdown(lc.analyse(str(tmp_path)))
    assert "S1 (English only), clean audio" in text and "100.0%" in text
