"""Tests for the Hinglish-vs-English trade-off table (src/reporting/tradeoff.py)."""

from __future__ import annotations

import json

from src.reporting import tradeoff as tr


def _write(path, eer, lo=None, hi=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"pooled": {"eer": eer, "eer_ci_low": lo or eer, "eer_ci_high": hi or eer}})
    )


def _inputs(tmp_path, with_new_results: bool):
    cells = []
    for run in tr.S2_RUNS:
        for ev in (*tr.SETS, "english"):
            cells.append({"run": run, "set": ev, "eer": 1.0, "eer_ci": [0.5, 1.5]})
    systems = tmp_path / "systems.json"
    systems.write_text(json.dumps({"cells": cells}))
    runs = [
        {"run": r, "set": "english", "deployed": {"recall": 0.774, "specificity": 1.0}}
        for r in tr.S2_RUNS
    ]
    runs.append(
        {
            "run": "stage1",
            "set": "asvspoof_eval",
            "deployed": {"recall": 0.935, "specificity": 0.996},
        }
    )
    classification = tmp_path / "classification.json"
    classification.write_text(json.dumps({"runs": runs}))
    english, cross = tmp_path / "english", tmp_path / "cross"
    if with_new_results:
        for run, (_, trained) in tr.S2_RUNS.items():
            other = "clean" if trained == "channel" else "channel"
            for ev in tr.SETS:
                _write(cross / f"{run}__{ev}_{other}.json", 0.2, 0.18, 0.22)
        for key in tr.ENGLISH_MODELS:
            _write(english / f"{key}__english_clean.json", 0.02)
            _write(english / f"{key}__english_channel.json", 0.05)
    return str(systems), str(classification), str(english), str(cross)


def test_table_reads_every_piece_from_its_file(tmp_path):
    result = tr.build(*_inputs(tmp_path, with_new_results=True))
    deployed = next(a for a in result["adapters"] if a["run"] == "lora_norm_rvc_channel")
    assert deployed["other_condition"] == "clean"
    assert deployed["hinglish_other_condition"]["cm04"] == {"eer": 20.0, "ci": [18.0, 22.0]}
    assert deployed["english_at_deployed_threshold"]["fakes_caught"] == 77.4
    assert result["english_sample"]["deployed_s2"]["channel"]["eer"] == 5.0
    assert result["s1_english_at_deployed_threshold"]["fakes_caught"] == 93.5


def test_missing_results_are_not_measured_not_invented(tmp_path):
    result = tr.build(*_inputs(tmp_path, with_new_results=False))
    deployed = next(a for a in result["adapters"] if a["run"] == "lora_norm_rvc_channel")
    assert deployed["hinglish_other_condition"]["eval_pool"] is None
    assert result["english_sample"]["s1"]["clean"] is None
    assert "not measured" in tr.markdown(result)


def test_markdown_has_one_row_per_adapter(tmp_path):
    text = tr.markdown(tr.build(*_inputs(tmp_path, with_new_results=True)))
    assert text.count("| XTTS + RVC |") == 2 and text.count("| XTTS |") == 2
    assert "| S1 (English only) | 2.00 | 5.00 |" in text
