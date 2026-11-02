"""Smoke test for src.reporting.paper_diagrams: both figures are written."""

from pathlib import Path

import pytest

from src.reporting import paper_diagrams


def test_draw_writes_both_figures(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    written = paper_diagrams.draw(str(tmp_path))
    names = sorted(Path(p).name for p in written)
    assert names == [
        "algorithm_flowchart.pdf",
        "algorithm_flowchart.png",
        "system_architecture_detailed.pdf",
        "system_architecture_detailed.png",
    ]
    assert all(Path(p).stat().st_size > 1000 for p in written)


def test_ladders_match_threshold_configs() -> None:
    """The numbers printed on the figure are the ones in the live configs."""
    import yaml

    deployed = yaml.safe_load(Path("configs/threshold.yaml").read_text())
    final = yaml.safe_load(Path("configs/threshold_final.yaml").read_text())
    assert paper_diagrams.LADDER["deployed"][:2] == (
        deployed["suspicious_after"],
        deployed["fake_after"],
    )
    assert paper_diagrams.LADDER["final"][:2] == (final["suspicious_after"], final["fake_after"])
    assert round(final["threshold"], 3) == paper_diagrams.LADDER["final"][2]
