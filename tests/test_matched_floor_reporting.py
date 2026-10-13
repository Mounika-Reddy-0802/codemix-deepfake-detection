"""The report tables and figures must judge S2/S3 cells by the matched, paired floor test.

See docs/results/shortcut_floor_v2.md: each detector's floor is fitted on its own
training data and scored on its own clips, and a cell counts as evidence only when the
speaker-resampled interval of detector minus floor lies below zero.
"""

from __future__ import annotations

import json

import pytest

from src.reporting import ablation as ab
from src.reporting import systems as sy
from src.training.floor_compare import BEATS_FLOOR, INCONCLUSIVE, SYSTEMS


def test_load_matched_covers_every_run_and_set():
    matched = ab.load_matched()
    assert len(matched) == len(SYSTEMS) * len(ab.SETS)
    deployed = matched[("lora_norm_rvc_channel", "cm04")]
    assert deployed["floor"] == pytest.approx(37.79)
    assert deployed["verdict"] == INCONCLUSIVE
    assert deployed["diff_ci"][0] < 0 < deployed["diff_ci"][1]
    assert matched[("s3_native_channel", "cm04")]["verdict"] == BEATS_FLOOR


def test_load_matched_needs_its_file(tmp_path):
    with pytest.raises(ab.AblationDataError, match="not computed"):
        ab.load_matched(str(tmp_path / "absent.json"))


def test_without_a_matched_entry_the_old_rule_applies():
    fields = ab.floor_fields(eer=4.0, ci_high=5.0, floor_v1=10.0, matched=None)
    assert fields == {"floor": 10.0, "margin": 6.0, "clears_floor": True}


def test_the_paired_verdict_overrides_the_clip_interval():
    # The clip interval (upper 5.0) sits under the floor, but with few speakers the
    # paired interval reaches zero: the cell is not evidence.
    entry = {"floor": 10.0, "diff_ci": [-12.1, 4.1], "verdict": INCONCLUSIVE}
    fields = ab.floor_fields(eer=4.0, ci_high=5.0, floor_v1=9.0, matched=entry)
    assert fields["clears_floor"] is False
    assert fields["floor"] == 10.0 and fields["floor_v1"] == 9.0
    assert fields["floor_verdict"] == INCONCLUSIVE and fields["diff_ci"] == [-12.1, 4.1]


def test_table_reads_each_adapter_against_its_own_matched_floor():
    summary = json.load(open(ab.SUMMARY))
    frame = ab.table(summary, ab.load_floors(), ab.load_matched())
    cell = frame[(frame.adapter == "lora_norm_rvc_channel") & (frame.set == "cm04")].iloc[0]
    assert cell["floor"] == pytest.approx(37.79)
    assert cell["floor_v1"] == pytest.approx(25.79)
    assert not cell["clears_floor"]
    # the two training mixes of one condition get different floors
    xtts = frame[(frame.adapter == "lora_norm_channel") & (frame.set == "cm04")].iloc[0]
    assert xtts["floor"] == pytest.approx(33.64)


def test_table_refuses_a_run_with_no_matched_floor():
    summary = json.load(open(ab.SUMMARY))
    matched = ab.load_matched()
    matched.pop(("lora_norm_clean", "rvc_test"))
    with pytest.raises(ab.AblationDataError, match="no matched floor"):
        ab.table(summary, ab.load_floors(), matched)


def test_systems_table_gives_s1_the_lowest_floor_of_its_set():
    s2 = json.load(open(sy.S2_SUMMARY))
    s3 = json.load(open(sy.S3_SUMMARY))
    s1_codemix = sy.load_s1_codemix()
    if s1_codemix is None:
        pytest.skip("S1 code-mixed results not in this checkout")
    matched = ab.load_matched()
    frame = sy.build_table(s2, s3, 0.85, ab.load_floors(), s1_codemix, matched)
    s1 = frame[(frame.system == "S1 English-only") & (frame.set == "cm04")]
    channel = s1[s1.condition == "channel"].iloc[0]
    assert channel["floor"] == pytest.approx(25.79)  # the oracle floor is the lowest
    s3_row = frame[(frame.run == "s3_native_channel") & (frame.set == "cm04")].iloc[0]
    assert s3_row["floor"] == pytest.approx(33.64) and s3_row["clears_floor"]


def test_committed_reports_agree_with_the_matched_floors():
    matched = ab.load_matched()
    for cell in json.load(open(ab.OUT))["cells"]:
        entry = matched[(cell["adapter"], cell["set"])]
        assert cell["floor"] == entry["floor"]
        assert cell["clears_floor"] == (entry["verdict"] == BEATS_FLOOR)
    for cell in json.load(open(sy.OUT))["cells"]:
        if cell["system"] in ("S2 LoRA", "S3 native") and cell["set"] != "english":
            entry = matched[(cell["run"], cell["set"])]
            assert cell["floor"] == entry["floor"]
            assert cell["clears_floor"] == (entry["verdict"] == BEATS_FLOOR)
