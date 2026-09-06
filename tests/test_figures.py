"""Tests for the publication figures (W9-T2).

Two things are worth testing here and one is not. The DET maths and the
figure/measurement agreement **are** worth testing: a figure that disagrees with
the results JSON it claims to plot is worse than no figure, because it looks
authoritative. Whether matplotlib draws a nice-looking axis is not worth testing,
so the plotting functions get a smoke test against a temp dir and nothing more.

The heatmap values are hard-coded in ``SYSTEM_MATRIX`` so the figure carries its
own provenance; ``test_english_column_matches_the_retention_json`` is what stops
that convenience from drifting away from the measured run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.reporting import figures as fg

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# DET maths
# --------------------------------------------------------------------------- #
def test_det_points_are_monotone_and_bounded():
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.normal(0.2, 0.1, 200), rng.normal(0.8, 0.1, 200)])
    labels = np.concatenate([np.zeros(200, int), np.ones(200, int)])
    fa, miss = fg.det_points(scores, labels)

    assert len(fa) == len(miss) == 400
    assert fa.min() >= 0 and fa.max() <= 100
    assert miss.min() >= 0 and miss.max() <= 100
    # sweeping the threshold up: more bonafide rejected, fewer spoofs accepted
    assert np.all(np.diff(fa) >= -1e-9)
    assert np.all(np.diff(miss) <= 1e-9)


def test_det_points_on_a_perfect_separator_touch_the_origin():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    fa, miss = fg.det_points(scores, labels)
    assert np.isclose(min(fa + miss), 0.0)  # a threshold exists with neither error


def test_det_points_need_both_classes():
    with pytest.raises(fg.FigureDataError, match="both classes"):
        fg.det_points(np.array([0.1, 0.2]), np.array([1, 1]))


def test_curve_from_scores_rejects_a_missing_file(tmp_path):
    with pytest.raises(fg.FigureDataError, match="not found"):
        fg.curve_from_scores("x", str(tmp_path / "nope.csv"))


def test_curve_from_scores_rejects_a_csv_without_labels(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"score": [0.1, 0.9]}).to_csv(path, index=False)
    with pytest.raises(fg.FigureDataError, match="'label'"):
        fg.curve_from_scores("x", str(path))


def test_curve_eer_matches_the_metrics_module(tmp_path):
    """The EER printed in the legend is the project's EER, not a second opinion."""
    from src.training.metrics import eer

    rng = np.random.default_rng(7)
    scores = np.concatenate([rng.normal(0.3, 0.15, 300), rng.normal(0.7, 0.15, 300)])
    labels = np.concatenate([np.zeros(300, int), np.ones(300, int)])
    path = tmp_path / "scores.csv"
    pd.DataFrame({"label": np.where(labels == 1, "bonafide", "spoof"), "score": scores}).to_csv(
        path, index=False
    )

    curve = fg.curve_from_scores("t", str(path))
    assert curve.clips == 600
    assert curve.eer == pytest.approx(float(eer(scores, labels)) * 100.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# The figure must agree with the measurement it claims to plot
# --------------------------------------------------------------------------- #
def test_english_column_matches_the_retention_json():
    """Every English cell is the number in the committed retention summary."""
    summary = REPO / "experiments" / "asvspoof_retention_summary.json"
    if not summary.is_file():
        pytest.skip("retention summary not present in this checkout")
    measured = json.loads(summary.read_text(encoding="utf-8"))["results"]

    expected = {
        "S1 baseline\n(English-trained)": "stage1_baseline",
        "S2 LoRA\n(clean-trained)": "lora_clean",
        "S2 LoRA\n(channel-matched)": "lora_channel",
    }
    column = "English\n(ASVspoof eval)"
    for system, key in expected.items():
        plotted = fg.SYSTEM_MATRIX[system][column][0]
        assert plotted == pytest.approx(measured[key]["eer"] * 100, abs=0.01), (
            f"{system} English cell says {plotted}% but "
            f"{summary.name} measured {measured[key]['eer'] * 100:.2f}%"
        )


def test_every_matrix_cell_cites_a_source():
    for system, cells in fg.SYSTEM_MATRIX.items():
        for condition, (value, source) in cells.items():
            assert source, f"{system}/{condition} has no provenance"
            assert 0.0 < value < 100.0, f"{system}/{condition} EER out of range"


def test_matrix_frame_is_systems_by_conditions():
    frame = fg.matrix_frame()
    assert frame.shape == (3, 3)
    assert frame.loc["S2 LoRA\n(channel-matched)", "Code-mixed\nchannel (G.711)"] == 3.89
    # the finding the figure exists to show: the clean adapter dies on a phone line
    assert (
        frame.loc["S2 LoRA\n(clean-trained)", "Code-mixed\nchannel (G.711)"]
        > frame.loc["S2 LoRA\n(channel-matched)", "Code-mixed\nchannel (G.711)"]
    )


def test_shortcut_gate_entries_are_sane():
    values = [v for _, v, _ in fg.SHORTCUT_GATE]
    assert all(0 < v < 100 for v in values)
    assert values[-1] > 50 - 10  # AffectDF's reference sits near chance
    assert values[0] < 10  # CM01 clean is the failing one


def test_missing_inputs_names_the_unmeasured_artefacts():
    absent = dict(fg.missing_inputs())
    for path in absent.values():
        assert not Path(path).is_file()


# --------------------------------------------------------------------------- #
# Plotting smoke tests
# --------------------------------------------------------------------------- #
def test_system_matrix_figure_writes_a_png(tmp_path):
    pytest.importorskip("matplotlib")
    out = fg.system_matrix_figure(str(tmp_path))
    assert Path(out).is_file() and Path(out).stat().st_size > 5_000


def test_shortcut_gate_figure_writes_a_png(tmp_path):
    pytest.importorskip("matplotlib")
    out = fg.shortcut_gate_figure(str(tmp_path))
    assert Path(out).is_file() and Path(out).stat().st_size > 5_000


def test_det_figure_writes_a_png(tmp_path):
    pytest.importorskip("matplotlib")
    rng = np.random.default_rng(3)
    sources = []
    for name in ("a", "b"):
        scores = np.concatenate([rng.normal(0.3, 0.2, 150), rng.normal(0.7, 0.2, 150)])
        labels = np.concatenate([np.zeros(150, int), np.ones(150, int)])
        path = tmp_path / f"{name}.csv"
        pd.DataFrame({"label": np.where(labels == 1, "bonafide", "spoof"), "score": scores}).to_csv(
            path, index=False
        )
        sources.append((name, str(path)))

    out = fg.det_curves_figure(str(tmp_path), sources=tuple(sources))
    assert Path(out).is_file() and Path(out).stat().st_size > 5_000
