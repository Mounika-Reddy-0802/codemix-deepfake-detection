"""The two-detector generator study figure draws from both rules' reports, never half of one."""

from __future__ import annotations

import pytest

pytest.importorskip("matplotlib")
from src.reporting import genstudy_both_figure as gb  # noqa: E402


def _report(seeds: int) -> dict:
    fold = {"control": [0.20] * seeds, "fold": [0.10] * seeds, "decision": "confirmed"}
    return {
        "answer": "carries over",
        "folds": {
            "fold_a": {**fold, "unseen_tool": "chatterbox"},
            "fold_b": {**fold, "unseen_tool": "indicf5", "decision": "not confirmed"},
        },
    }


def test_draws_png_and_pdf_from_both_reports(tmp_path):
    paths = gb.draw({"adapter": _report(5), "ssl": _report(2)}, str(tmp_path / "both"))
    assert [p.suffix for p in paths] == [".png", ".pdf"]
    assert all(p.stat().st_size > 0 for p in paths)


def test_refuses_an_unfinished_study(tmp_path):
    with pytest.raises(ValueError):
        gb.draw({"adapter": _report(5), "ssl": {"answer": "incomplete"}}, str(tmp_path / "x"))
