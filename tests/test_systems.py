"""The three-system comparison reads every run it needs and answers W7-T2 from numbers."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.reporting import systems as sy
from src.reporting.ablation import FLOORS, SETS


def _run(eer: float) -> dict:
    return {
        "eer": eer,
        "eer_ci_low": eer - 0.01,
        "eer_ci_high": eer + 0.01,
        "auc": 0.9,
        "clips": 100,
    }


def _summaries(s3_english: float = 0.55, s2_english: float = 0.03):
    s2, s3 = {}, {}
    for s2_run, s3_run in sy.PAIRS.values():
        for ev in SETS:
            s2[f"{s2_run}__{ev}"] = _run(0.05)
            s3[f"{s3_run}__{ev}"] = _run(0.04)
        s3[f"{s2_run}__english"] = _run(s2_english)
        s3[f"{s3_run}__english"] = _run(s3_english)
    return s2, s3


FLOOR_VALUES = {key: 20.0 for key in FLOORS}


def test_table_covers_both_systems_every_set_and_s1_english():
    s2, s3 = _summaries()
    frame = sy.build_table(s2, s3, 0.85, FLOOR_VALUES)
    assert set(frame["system"]) == {"S1 English-only", "S2 LoRA", "S3 native"}
    for system in ("S2 LoRA", "S3 native"):
        part = frame[frame["system"] == system]
        assert len(part) == len(sy.PAIRS) * (len(SETS) + 1)


def test_english_has_no_floor_and_codemix_cells_do():
    s2, s3 = _summaries()
    frame = sy.build_table(s2, s3, 0.85, FLOOR_VALUES)
    assert frame.loc[frame["set"] == "english", "floor"].isna().all()
    assert frame.loc[frame["set"] != "english", "floor"].notna().all()


def test_reverse_degradation_says_yes_when_s3_english_is_at_chance():
    s2, s3 = _summaries(s3_english=0.52, s2_english=0.03)
    frame = sy.build_table(s2, s3, 0.85, FLOOR_VALUES)
    answer = sy.reverse_degradation(frame, 0.85)
    assert answer["answer"].startswith("Yes.")
    assert answer["s3_english_range"] == [52.0, 52.0]


def test_reverse_degradation_does_not_overclaim():
    s2, s3 = _summaries(s3_english=0.10)
    frame = sy.build_table(s2, s3, 0.85, FLOOR_VALUES)
    assert sy.reverse_degradation(frame, 0.85)["answer"].startswith("Partly")


def test_missing_run_is_an_error():
    s2, s3 = _summaries()
    s3.pop("s3_native_clean__english")
    with pytest.raises(sy.SystemsDataError, match="s3_native_clean__english"):
        sy.build_table(s2, s3, 0.85, FLOOR_VALUES)


def test_committed_artefacts_match_the_committed_summaries():
    committed = pd.DataFrame(json.load(open(sy.OUT))["cells"])
    s3 = json.load(open(sy.S3_SUMMARY))
    row = committed[(committed["run"] == "s3_native_rvc_clean") & (committed["set"] == "eval_pool")]
    assert row["eer"].iloc[0] == round(100 * s3["s3_native_rvc_clean__eval_pool"]["eer"], 2)
    reverse = json.load(open(sy.REVERSE_OUT))
    assert len(reverse["models"]) == 2 * len(sy.PAIRS)


def _s1_codemix(eer: float = 0.48) -> dict:
    return {
        f"stage1_{condition}__{ev}": _run(eer) for ev in SETS for condition in ("clean", "channel")
    }


def test_s1_codemix_cells_are_read_against_the_same_floors():
    s2, s3 = _summaries()
    frame = sy.build_table(s2, s3, 0.85, FLOOR_VALUES, _s1_codemix(0.48))
    s1 = frame[(frame["system"] == "S1 English-only") & (frame["set"] != "english")]
    assert len(s1) == 2 * len(SETS)
    assert (s1["floor"] == 20.0).all()
    assert not s1["clears_floor"].any()


def test_s1_codemix_is_omitted_when_not_measured():
    s2, s3 = _summaries()
    frame = sy.build_table(s2, s3, 0.85, FLOOR_VALUES)
    s1 = frame[frame["system"] == "S1 English-only"]
    assert set(s1["set"]) == {"english"}


def test_load_s1_codemix_needs_every_set_and_condition(tmp_path):
    assert sy.load_s1_codemix(str(tmp_path)) is None
    for ev in SETS:
        for suffix in ("", "_channel"):
            (tmp_path / f"stage1__{ev}{suffix}.json").write_text(json.dumps({"pooled": _run(0.4)}))
    loaded = sy.load_s1_codemix(str(tmp_path))
    assert loaded is not None and len(loaded) == 2 * len(SETS)
    (tmp_path / "stage1__cm04_channel.json").unlink()
    assert sy.load_s1_codemix(str(tmp_path)) is None
