"""The study's tables are written from the rule's output, numbers unchanged."""

from __future__ import annotations

from src.reporting import generator_study as rep
from src.reporting import genstudy_tables as gt
from tests.test_generator_study_report import _table


def test_tables_quote_the_rule_output_exactly():
    table = _table(fold_a=0.10, fold_b=0.25)
    report = rep.judge(table)
    report["means"] = rep.means(table)
    text = gt.render(report)
    assert "**Answer: partly.**" in text
    assert "| 1234 | 30.00 | 10.00 |" in text
    assert "not met" in text and "met" in text
    assert "Chatterbox, G.711" in text


def test_an_unfinished_study_says_so():
    assert "incomplete" in gt.decision_tables({"answer": "incomplete", "seeds": [7]})
