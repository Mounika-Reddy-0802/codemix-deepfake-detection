"""Spontaneous genuine speech against each tool's clones, threshold-free."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.reporting import spontaneous_separation as ss


def test_genuine_files_with_and_without_labels_and_one_tool_of_a_clone_set(tmp_path):
    hiacc = tmp_path / "hiacc.csv"
    pd.DataFrame({"filepath": ["a", "b"], "score": [0.9, 0.8]}).to_csv(hiacc, index=False)
    mixed = tmp_path / "mixed.csv"
    pd.DataFrame(
        {
            "label": ["bonafide", "spoof", "spoof", "spoof"],
            "tool": ["none", "tortoise", "tortoise", "xtts_v2"],
            "score": [0.7, 0.1, 0.2, 0.95],
        }
    ).to_csv(mixed, index=False)

    genuine = ss.genuine_scores([str(hiacc), str(mixed)])
    assert sorted(genuine) == [0.7, 0.8, 0.9]
    report = ss.separation(
        genuine,
        {
            "tortoise": ss.clone_scores(str(mixed), "tortoise"),
            "xtts_v2": ss.clone_scores(str(mixed), "xtts_v2"),
        },
    )
    assert report["genuine_clips"] == 3
    assert report["tools"]["tortoise"] == {"clones": 2, "eer_pct": 0.0}
    assert report["tools"]["xtts_v2"]["eer_pct"] == 100.0


def test_files_on_different_score_scales_are_refused():
    probabilities = np.array([0.9, 0.8, 0.7])
    log_odds = np.array([-4.2, 1.3, 6.0])
    with pytest.raises(ValueError, match="mix P"):
        ss.separation(probabilities, {"tortoise": log_odds})
    caught = ss.separation(log_odds, {"tortoise": np.array([-9.0, -8.0])})
    assert caught["tools"]["tortoise"]["eer_pct"] == 0.0
