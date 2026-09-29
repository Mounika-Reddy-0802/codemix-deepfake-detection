"""Confusion matrices and threshold metrics, with spoof as the positive class."""

from __future__ import annotations

import numpy as np
import pytest

from src.reporting import classification as cl


def test_a_clip_is_called_spoof_when_it_scores_below_the_threshold():
    labels = np.array([1, 1, 0, 0])  # bonafide, bonafide, spoof, spoof
    scores = np.array([0.9, 0.4, 0.1, 0.8])  # P(bonafide)
    counts = cl.confusion_at(labels, scores, threshold=0.5)
    assert counts.tp == 1, "the spoof at 0.1 is caught"
    assert counts.fn == 1, "the spoof at 0.8 is missed"
    assert counts.tn == 1, "the real clip at 0.9 is left alone"
    assert counts.fp == 1, "the real clip at 0.4 is a false alarm"
    assert counts.total == 4 and counts.spoof == 2 and counts.bonafide == 2


def test_a_score_exactly_on_the_threshold_counts_as_real():
    labels = np.array([0, 1])
    counts = cl.confusion_at(labels, np.array([0.5, 0.5]), threshold=0.5)
    assert (counts.tp, counts.fn, counts.tn, counts.fp) == (0, 1, 1, 0)


def test_metrics_match_the_counts():
    counts = cl.Confusion(tp=90, fn=10, tn=80, fp=20)
    m = cl.metrics_from(counts)
    assert m["recall"] == pytest.approx(0.90)
    assert m["precision"] == pytest.approx(90 / 110)
    assert m["specificity"] == pytest.approx(0.80)
    assert m["accuracy"] == pytest.approx(170 / 200)
    assert m["f1"] == pytest.approx(2 * (90 / 110) * 0.9 / ((90 / 110) + 0.9))
    assert m["balanced_accuracy"] == pytest.approx(0.85)
    assert m["false_alarm_rate"] == pytest.approx(0.20)
    assert m["miss_rate"] == pytest.approx(0.10)


def test_an_empty_denominator_is_reported_as_unknown_not_as_zero():
    """Precision with no positive predictions is undefined; 0.0 would be a lie."""
    m = cl.metrics_from(cl.Confusion(tp=0, fn=5, tn=5, fp=0))
    assert m["precision"] is None
    assert m["recall"] == pytest.approx(0.0)
    assert m["f1"] is None


def test_perfect_and_useless_detectors_sit_at_the_extremes():
    labels = np.array([1, 1, 0, 0])
    perfect = cl.metrics_from(cl.confusion_at(labels, np.array([0.9, 0.8, 0.1, 0.2]), 0.5))
    assert perfect["accuracy"] == 1.0 and perfect["recall"] == 1.0
    inverted = cl.metrics_from(cl.confusion_at(labels, np.array([0.1, 0.2, 0.9, 0.8]), 0.5))
    assert inverted["accuracy"] == 0.0


def test_labels_and_scores_must_line_up():
    with pytest.raises(ValueError, match="same length"):
        cl.confusion_at(np.array([1, 0]), np.array([0.5]), 0.5)


def test_read_scores_maps_labels_and_skips_clips_that_never_scored(tmp_path):
    csv = tmp_path / "run__set_scores.csv"
    csv.write_text(
        "filepath,label,score\na.wav,bonafide,0.91\nb.wav,spoof,0.02\nc.wav,spoof,\n",
        encoding="utf-8",
    )
    labels, scores = cl.read_scores(csv)
    assert labels.tolist() == [1, 0], "the clip with no score is not a prediction"
    assert scores.tolist() == pytest.approx([0.91, 0.02])


def test_run_and_set_are_read_from_the_filename():
    assert cl.split_name("lora_norm_channel__cm04_scores.csv") == ("lora_norm_channel", "cm04")
    assert cl.split_name("results/w10/stage1__eval_pool_channel_scores.csv") == (
        "stage1",
        "eval_pool_channel",
    )


def test_a_single_class_set_is_refused_rather_than_scored(tmp_path):
    csv = tmp_path / "run__set_scores.csv"
    csv.write_text("filepath,label,score\na.wav,spoof,0.1\nb.wav,spoof,0.2\n", encoding="utf-8")
    entry = cl.report_for(csv, threshold=0.5)
    assert "error" in entry
    assert "deployed" not in entry


def test_a_report_carries_both_operating_points_and_the_class_balance(tmp_path):
    csv = tmp_path / "lora_norm_rvc_channel__eval_pool_scores.csv"
    rows = ["filepath,label,score"]
    rows += [f"r{i}.wav,bonafide,{0.95 - i * 0.01:.3f}" for i in range(10)]
    rows += [f"s{i}.wav,spoof,{0.05 + i * 0.01:.3f}" for i in range(10)]
    csv.write_text("\n".join(rows), encoding="utf-8")

    entry = cl.report_for(csv, threshold=0.5)
    assert entry["run_label"] == "S2 LoRA, XTTS+RVC, channel (deployed)"
    assert entry["set_label"] == "seen tool (XTTS)"
    assert entry["bonafide"] == 10 and entry["spoof"] == 10
    assert entry["deployed"]["threshold"] == 0.5
    assert entry["eer_point"]["threshold"] == pytest.approx(entry["eer_threshold"])
    assert entry["eer"] == pytest.approx(0.0, abs=1e-9), "these classes separate perfectly"


def test_the_markdown_table_has_a_row_per_run():
    report = {
        "runs": [
            {
                "run_label": "S1",
                "set_label": "English",
                "bonafide": 2,
                "spoof": 2,
                "eer": 0.1,
                "deployed": {
                    "confusion": {"tp": 2, "fn": 0, "tn": 2, "fp": 0},
                    "accuracy": 1.0,
                    "precision": 1.0,
                    "recall": 1.0,
                    "specificity": 1.0,
                    "f1": 1.0,
                },
            },
            {"run": "broken", "error": "one class only"},
        ]
    }
    table = cl.markdown(report)
    assert "| S1 | English |" in table
    assert "broken" not in table, "a refused run must not appear as a row of zeros"
