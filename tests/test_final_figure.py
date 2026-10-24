"""Smoke tests for src.reporting.final_figure (reads the committed result files)."""

from pathlib import Path

import pytest

from src.reporting import final_figure as ff

NEEDED = (
    "final_v1/runs.json",
    "final_ssl_v1/runs.json",
    "final_ssl_seeds2_v1/runs.json",
    "floor_final_v1.json",
    "cm09_v1/final_s1234__score_cm09_channel.json",
)


def _have_results() -> bool:
    return all((ff.R / p).is_file() for p in NEEDED)


@pytest.mark.skipif(not _have_results(), reason="week-13 result files not present")
def test_table_has_every_cell_and_sane_numbers():
    t = ff.table()
    assert set(t["system"]) == set(ff.SYSTEMS)
    assert t["eer"].between(0, 100).all() and t["floor"].between(0, 100).all()
    # five paired seeds for both final systems on every multi-seed cell
    final = t[(t["system"] == "Final-recipe adapter") & (t["tool"] != "Chatterbox")]
    assert (final["n"] == 5).all()
    assert (t[t["system"] == "SSL-AASIST, final data"]["n"] == 5).all()
    # S1 has no AMR-NB scores, so it never appears in that panel
    assert not ((t["system"] == "S1, English only") & t["panel"].str.startswith("AMR")).any()


@pytest.mark.skipif(not _have_results(), reason="week-13 result files not present")
def test_figure_is_written(tmp_path):
    pytest.importorskip("matplotlib")
    out = Path(ff.figure(str(tmp_path)))
    assert out.is_file() and out.with_suffix(".pdf").is_file()
