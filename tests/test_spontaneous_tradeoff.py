"""Smoke tests for the spontaneous-speech trade-off (src.reporting.spontaneous_tradeoff)."""

from __future__ import annotations

import pandas as pd

from src.reporting import spontaneous_tradeoff as st


def _windows(prefix: str, label: str, tool: str, score: float, clips: int = 6) -> pd.DataFrame:
    rows = [
        {
            "filepath": f"{prefix}_{c}.wav",
            "label": label,
            "tool": tool,
            "window": w,
            "end_seconds": 4.0 + 2.0 * w,
            "level_dbfs": -22.0,
            "score": score,
            "smoothed": score,
        }
        for c in range(clips)
        for w in range(6)
    ]
    return pd.DataFrame(rows)


def _write_sets(root) -> None:
    pd.concat(
        [
            _windows("cm04_real", "bonafide", "none", 0.95),
            _windows("cm04_fake", "spoof", "tortoise", 0.3),
        ]
    ).to_csv(root / "cm04_windows.csv", index=False)
    pd.concat(
        [
            _windows("cm07_real", "bonafide", "none", 0.95),
            _windows("cm07_fake", "spoof", "indicf5", 0.6),
        ]
    ).to_csv(root / "cm07_windows.csv", index=False)
    _windows("hiacc", "bonafide", "none", 0.4).to_csv(root / "hiacc_windows.csv", index=False)


def test_sweep_reads_every_set_and_rates_rise_with_the_threshold(tmp_path):
    _write_sets(tmp_path)
    calls = st.stitched_sets(tmp_path)
    assert set(calls) == {"cm04", "cm07", "hiacc"}
    table = st.summary(st.sweep(calls, [0.2, 0.5, 0.8]))
    assert list(table["threshold"]) == [0.2, 0.5, 0.8]
    # every call scores a constant, so each is warned exactly when its score is below the threshold
    assert list(table["hiacc_genuine_warned_pct"]) == [0.0, 100.0, 100.0]
    assert list(table["cm04_caught_pct"]) == [0.0, 100.0, 100.0]
    assert list(table["cm07_caught_pct"]) == [0.0, 0.0, 100.0]
    assert list(table["read_genuine_warned_worst_set_pct"]) == [0.0, 0.0, 0.0]
    assert list(table["unseen_clones_caught_pct"]) == [0.0, 50.0, 100.0]


def test_figure_is_written(tmp_path):
    _write_sets(tmp_path)
    table = st.summary(st.sweep(st.stitched_sets(tmp_path), st.GRID))
    out = st.figure(table, 0.687, out_dir=str(tmp_path))
    assert (tmp_path / "spontaneous_tradeoff.png").is_file()
    assert out.endswith("spontaneous_tradeoff.png")
