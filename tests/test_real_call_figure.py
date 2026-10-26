"""Smoke tests for src.reporting.real_call_figure, on a small made-up result folder."""

from pathlib import Path

import pandas as pd
import pytest

from src.reporting import real_call_figure as rf


def _results(root: Path) -> Path:
    """Two speakers through the chain, and one genuine live call and one clone call per model."""
    rows = [
        {
            "recording": rec,
            "version": stage,
            "model": model,
            "mean_p_genuine": p,
            "fake_leaning_pct": 0.0,
            "windows": 10,
        }
        for rec in ("P2_read_A", "P3_read_A")
        for stage, _ in rf.STAGES[:-1]
        for model, p in (("deployed", 0.4), ("final", 0.9))
    ]
    pd.DataFrame(rows).to_csv(root / "codec_chain.csv", index=False)
    for model, live, clone in (("deployed", 0.1, 0.05), ("final", 0.95, 0.2)):
        pd.DataFrame(
            {
                "filepath": [
                    "${DATA_ROOT}/g711/C05.wav",
                    "${DATA_ROOT}/g711/C05.wav",
                    "${DATA_ROOT}/g711/C21.wav",
                ],
                "score": [live, live + 0.02, clone],
            }
        ).to_csv(root / f"windows_{model}_g711.csv", index=False)
    return root


def test_table_has_every_stage_in_order_and_only_genuine_live_calls(tmp_path):
    t = rf.table(_results(tmp_path))
    for model in rf.MODELS:
        part = t[t["model"] == model]
        assert list(part["stage"]) == [key for key, _ in rf.STAGES]
    real = t[t["stage"] == "real_calls"].set_index("model")["mean_p_genuine"]
    # the clone call (C21) is left out of the genuine live mean
    assert real["deployed"] == pytest.approx(0.11)
    assert real["final"] == pytest.approx(0.96)


def test_every_genuine_call_counts_once_however_long(tmp_path):
    root = _results(tmp_path)
    pd.DataFrame(
        {
            "filepath": ["${DATA_ROOT}/g711/C05.wav"] * 3 + ["${DATA_ROOT}/g711/C09.wav"],
            "score": [0.0, 0.0, 0.0, 1.0],
        }
    ).to_csv(root / "windows_final_g711.csv", index=False)
    real = rf.table(root).set_index(["model", "stage"])["mean_p_genuine"]
    assert real[("final", "real_calls")] == pytest.approx(0.5)  # not 0.25


def test_table_without_window_files_has_the_chain_only(tmp_path):
    root = _results(tmp_path)
    for model in rf.MODELS:
        (root / f"windows_{model}_g711.csv").unlink()
    assert "real_calls" not in set(rf.table(root)["stage"])


def test_labels_avoid_threshold_lines_and_the_axis():
    above = (0, 6)
    assert rf._placement(0.08, [0.5, 0.687])[0] == above  # would fall below the axis
    assert rf._placement(0.77, [0.5, 0.687])[0] == above  # would sit on the 0.69 line
    assert rf._placement(0.29, [0.5, 0.687])[0] != above


def test_figure_is_written(tmp_path):
    pytest.importorskip("matplotlib")
    out = Path(rf.figure(str(tmp_path / "fig"), root=_results(tmp_path)))
    assert out.is_file() and out.with_suffix(".pdf").is_file()
