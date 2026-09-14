"""Demo operating point from streaming-path window scores (W8-T5, P-031)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from live_call.verdict_engine import EngineConfig
from src.inference import calibrate as cb


def _windows(label: str, scores: list[list[float]], prefix: str) -> list[dict]:
    rows = []
    for c, clip in enumerate(scores):
        ema = None
        for w, s in enumerate(clip):
            ema = s if ema is None else 0.5 * s + 0.5 * ema
            rows.append(
                {
                    "filepath": f"{prefix}{c}",
                    "label": label,
                    "tool": "x",
                    "window": w,
                    "end_seconds": 4.0 + 2 * w,
                    "level_dbfs": -23.0,
                    "score": s,
                    "smoothed": ema,
                }
            )
    return rows


def _dev(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    real = [list(np.clip(0.999 - rng.exponential(0.002, 3), 0, 1)) for _ in range(40)]
    real[0][1] = 0.3  # one genuine window that looks fake
    fake = [list(np.clip(rng.exponential(0.05, 3), 0, 1)) for _ in range(40)]
    return pd.DataFrame(_windows("bonafide", real, "r") + _windows("spoof", fake, "f"))


def test_stratified_subset_keeps_both_labels_and_is_deterministic():
    frame = pd.DataFrame(
        {"filepath": [f"c{i}" for i in range(100)], "label": ["bonafide"] * 30 + ["spoof"] * 70}
    )
    a, b = cb.stratified_subset(frame, 20), cb.stratified_subset(frame, 20)
    assert a.equals(b) and set(a["label"]) == {"bonafide", "spoof"}
    assert len(cb.stratified_subset(frame, None)) == 100


def test_window_rows_pad_short_clips_to_one_window():
    rows = cb.window_rows("a.wav", "spoof", "xtts", np.full(8000, 0.05, np.float32), lambda w: 0.2)
    assert len(rows) == 1 and rows[0]["score"] == 0.2


def test_eer_threshold_separates_well_ordered_scores():
    dev = _dev()
    choice = cb.choose_threshold(dev)
    assert choice["dev_window_eer"] < 5.0
    assert 0.0 < choice["threshold"] < 1.0


def test_stitched_calls_are_long_and_resmoothed():
    dev = _dev()
    calls = cb.stitch_calls(dev, 30)
    per_call = calls.groupby("filepath")["window"].count()
    assert (per_call == 14).all()
    assert set(calls["label"]) == {"bonafide", "spoof"}


def test_call_metrics_count_false_alarms_and_detections():
    dev = _dev()
    metrics = cb.call_metrics(
        cb.replay_calls(cb.stitch_calls(dev, 60), EngineConfig(threshold=0.5))
    )
    assert metrics["fake_calls_alerted_pct"] == 100.0
    assert metrics["real_calls_alerted_pct"] == 0.0


def test_sweep_prefers_a_point_without_false_alarms():
    chosen, candidates = cb.sweep(_dev())
    assert chosen["rule_satisfied"]
    assert chosen["dev_calls_60s"]["real_calls_alerted_pct"] <= cb.MAX_REAL_CALL_ALERT_PCT
    assert len(candidates) >= 2


def test_choose_needs_both_classes():
    only_real = _dev()[lambda d: d["label"] == "bonafide"]
    with pytest.raises(ValueError, match="both real and fake"):
        cb.choose_threshold(only_real)


def test_committed_threshold_matches_its_report():
    import yaml

    config = yaml.safe_load(open("configs/threshold.yaml"))
    report = json.load(open("experiments/results/threshold_calibration.json"))
    assert config["threshold"] == pytest.approx(report["threshold"])
    assert report["rule_satisfied"] is True
    for held_out in report["validation"].values():
        assert held_out["calls_60s"]["real_calls_alerted_pct"] <= cb.MAX_REAL_CALL_ALERT_PCT
