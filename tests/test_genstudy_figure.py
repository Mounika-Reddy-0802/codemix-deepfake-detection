"""The generator study figure draws from a report and refuses an unfinished one."""

from __future__ import annotations

import pytest

from src.reporting import generator_study as rep
from tests.test_generator_study_report import _table

pytest.importorskip("matplotlib")
from src.reporting import genstudy_figure as gf  # noqa: E402


def test_draws_png_and_pdf_from_a_finished_report(tmp_path):
    table = _table(fold_a=0.10, fold_b=0.25)
    report = rep.judge(table)
    report["means"] = rep.means(table)
    paths = gf.draw(report, str(tmp_path / "fig"))
    assert [p.suffix for p in paths] == [".png", ".pdf"]
    assert all(p.stat().st_size > 0 for p in paths)


def test_refuses_an_incomplete_study():
    with pytest.raises(ValueError):
        gf.draw({"answer": "incomplete"})


def test_every_condition_of_the_rule_has_a_reason():
    report = rep.judge(_table(fold_a=0.10, fold_b=0.25))
    for f in report["folds"].values():
        assert set(f["conditions"]) == set(gf.FAILED)
        assert all(gf.FAILED[k](f) for k in f["conditions"])
