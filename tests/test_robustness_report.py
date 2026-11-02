"""The robustness report applies the reading rules fixed before scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.reporting import robustness as rr


def _scored(transforms: dict[str, tuple[float, float]], per_speaker: int = 4) -> pd.DataFrame:
    """Ten genuine and ten cloned speakers; each transform sets (genuine, clone) scores."""
    rng = np.random.default_rng(0)
    rows = []
    for transform, (genuine, clone) in {"none": (0.9, 0.1), **transforms}.items():
        for label, centre, prefix in (("bonafide", genuine, "g"), ("spoof", clone, "c")):
            for s in range(10):
                for k in range(per_speaker):
                    name = f"{prefix}{s}_{k}.wav"
                    rows.append(
                        {
                            "filepath": f"${{DATA_ROOT}}/{transform}/{name}",
                            "label": label,
                            "speaker": f"{prefix}{s}",
                            "set": "new_speakers",
                            "transform": transform,
                            "score": float(np.clip(centre + rng.normal(0, 0.02), 0, 1)),
                        }
                    )
    return pd.DataFrame(rows)


def test_each_reading_follows_its_rule() -> None:
    rows = rr.compare(
        _scored({"helps": (0.9, 0.7), "hurts": (0.3, 0.1), "same": (0.9, 0.1)}), "adapter", 200
    )
    by = {r["transform"]: r for r in rows}
    assert by["helps"]["reading"] == "helps an attacker"
    assert by["helps"]["clones_passed"]["change"] == pytest.approx(1.0)
    assert by["hurts"]["reading"] == "hurts a genuine caller"
    assert by["hurts"]["false_alarms"]["with"] == pytest.approx(1.0)
    assert by["same"]["reading"] == "no measurable effect"
    assert by["same"]["eer_change"] == pytest.approx(0.0, abs=0.05)


def test_ssl_scores_are_log_odds_cut_at_zero() -> None:
    df = _scored({"same": (0.9, 0.1)})
    df["score"] = np.where(df["label"] == "bonafide", 2.0, -2.0)  # log-odds, both sides of 0
    (row,) = rr.compare(df, "ssl", 100)
    assert row["false_alarms"]["with"] == 0.0 and row["clones_passed"]["with"] == 0.0


def test_a_transform_missing_clips_is_refused() -> None:
    df = _scored({"same": (0.9, 0.1)})
    df = df.drop(df[(df["transform"] == "same")].index[:3])
    with pytest.raises(ValueError, match="not scored"):
        rr.compare(df, "adapter", 50)


def test_the_table_has_one_row_per_transform() -> None:
    rows = rr.compare(_scored({"helps": (0.9, 0.7)}), "adapter", 50)
    table = rr.markdown(rows)
    assert table.count("\n") == 3 and "`helps`" in table and "helps an attacker" in table
