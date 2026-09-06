"""f0-statistics tests: the arithmetic behind P-019's claim, pinned.

P-019's 25-29 Hz vs 41.1 Hz was measured ad hoc, so nothing pinned *how* the
number was computed. These tests pin the part that would otherwise drift
silently under a library upgrade -- voiced-frame filtering, the IQR itself, the
median-of-IQRs summary, and the direction of the verdict. The pitch tracker is
not exercised here: CI installs neither Praat nor librosa (P-004), and a
synthetic contour tests the statistics more exactly than synthetic audio would.
"""

from __future__ import annotations

import numpy as np

from src.data import f0_stats


def test_iqr_is_the_quartile_spread() -> None:
    assert f0_stats.iqr([1, 2, 3, 4, 5]) == 2.0


def test_iqr_of_a_flat_contour_is_zero() -> None:
    assert f0_stats.iqr([120.0] * 50) == 0.0


def test_iqr_needs_two_values() -> None:
    assert f0_stats.iqr([]) == 0.0
    assert f0_stats.iqr([110.0]) == 0.0


def test_unvoiced_frames_never_reach_the_statistics() -> None:
    """Zeros mean 'no pitch here', not 'pitch of 0 Hz'.

    Folding them in would drag every IQR toward the floor and manufacture the
    very compression P-019 is looking for.
    """
    voiced = list(np.linspace(100.0, 140.0, 40))
    stats = f0_stats.contour_stats(voiced + [0.0] * 40)
    assert stats["voiced_frames"] == 40
    assert stats["f0_iqr_hz"] == f0_stats.contour_stats(voiced)["f0_iqr_hz"]


def test_a_contour_too_short_to_measure_is_marked_unusable() -> None:
    stats = f0_stats.contour_stats([110.0, 120.0, 130.0])
    assert stats["usable"] is False
    assert stats["f0_iqr_hz"] == 0.0


def _rows(iqr_values: list[float]) -> list[dict]:
    """Per-clip stat rows with the given IQRs, all usable."""
    return [
        {"voiced_frames": 50, "f0_median_hz": 110.0, "f0_iqr_hz": value, "usable": True}
        for value in iqr_values
    ]


def test_summary_headline_is_the_median_of_per_clip_iqrs() -> None:
    summary = f0_stats.summarise(_rows([10.0, 40.0, 42.0, 44.0, 300.0]))
    assert summary["median_f0_iqr_hz"] == 42.0
    assert summary["usable"] == 5


def test_unusable_clips_are_counted_but_not_measured() -> None:
    rows = _rows([40.0, 42.0]) + [{"voiced_frames": 2, "f0_iqr_hz": 0.0, "usable": False}]
    summary = f0_stats.summarise(rows)
    assert summary["clips"] == 3
    assert summary["usable"] == 2


def test_summary_of_nothing_usable_reports_no_measurement() -> None:
    assert f0_stats.summarise([])["usable"] == 0


def test_retention_is_converted_over_real_as_a_percentage() -> None:
    converted = {"median_f0_iqr_hz": 41.0}
    real = {"median_f0_iqr_hz": 41.0}
    assert f0_stats.retention(converted, real) == 100.0
    assert f0_stats.retention({"median_f0_iqr_hz": 27.0}, real) == 65.9


def test_retention_without_a_reference_is_zero_not_a_crash() -> None:
    assert f0_stats.retention({"median_f0_iqr_hz": 41.0}, {}) == 0.0


def test_verdict_holds_when_the_contour_survives_conversion() -> None:
    text = f0_stats.verdict({"median_f0_iqr_hz": 40.0}, {"median_f0_iqr_hz": 41.0})
    assert "HOLDS" in text


def test_verdict_fails_at_xtts_level_compression() -> None:
    """The finding that matters most is the one that contradicts the plan."""
    text = f0_stats.verdict({"median_f0_iqr_hz": 27.0}, {"median_f0_iqr_hz": 41.0})
    assert "FAILS" in text


def test_compare_reports_both_sides_and_a_verdict() -> None:
    result = f0_stats.compare(_rows([41.0, 41.0]), _rows([41.0, 41.0]))
    assert result["retention_pct"] == 100.0
    assert result["converted"]["usable"] == 2
    assert result["real_source"]["usable"] == 2
    assert "HOLDS" in result["verdict"]
