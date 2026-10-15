"""Tests for the English-over-the-phone-line report."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.reporting import english_channel as ec


def _write(tmp_path, model: str, condition: str, genuine: list, spoof: list) -> None:
    rows = [
        {"label": "bonafide", "speaker": f"S{i % 3}", "tool": "-", "score": s}
        for i, s in enumerate(genuine)
    ] + [
        {"label": "spoof", "speaker": f"S{i % 3}", "tool": "A07", "score": s}
        for i, s in enumerate(spoof)
    ]
    stem = tmp_path / f"{model}__english_{condition}"
    pd.DataFrame(rows).to_csv(f"{stem}_scores.csv", index=False)
    pooled = {"eer": 0.25, "eer_ci_low": 0.2, "eer_ci_high": 0.3, "auc": 0.8, "clips": len(rows)}
    per_attack = [{"attack": "A07", "eer": 0.1}, {"attack": "A19", "eer": 0.4}]
    stem.with_suffix(".json").write_text(json.dumps({"pooled": pooled, "per_attack": per_attack}))


def test_fakes_caught_counts_spoofs_below_threshold() -> None:
    scores = pd.DataFrame(
        {"label": ["spoof"] * 4 + ["bonafide"], "score": [0.1, 0.2, 0.7, 0.9, 0.1]}
    )
    assert ec.fakes_caught(scores) == pytest.approx(0.5)


def test_fakes_caught_needs_spoofs() -> None:
    with pytest.raises(ValueError):
        ec.fakes_caught(pd.DataFrame({"label": ["bonafide"], "score": [0.3]}))


def test_cell_reads_rates_and_attacks(tmp_path) -> None:
    _write(tmp_path, "s1", "channel", genuine=[0.9, 0.9, 0.1, 0.9], spoof=[0.1, 0.1, 0.1, 0.9])
    c = ec.cell(tmp_path, "s1", "channel")
    assert c["eer"] == 25.0 and c["eer_ci"] == [20.0, 30.0]
    assert c["genuine_called_fake"] == 25.0
    assert c["fakes_caught"] == 75.0
    assert c["per_attack_eer"] == {"A07": 10.0, "A19": 40.0}
    assert c["speakers"] == 3


def test_missing_cells_are_not_measured(tmp_path) -> None:
    _write(tmp_path, "deployed_s2", "clean", genuine=[0.9, 0.8, 0.7], spoof=[0.1, 0.2, 0.3])
    result = ec.build(tmp_path)
    assert result["cells"]["s1"]["clean"] is None
    assert result["cells"]["deployed_s2"]["clean"]["fakes_caught"] == 100.0
    text = ec.markdown(result)
    assert "not measured" in text
    assert "| A19 |" in text
