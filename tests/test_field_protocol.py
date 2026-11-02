"""The voice check's field-recording plan agrees with its protocol and the real-call files."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.data import real_call_pack as rcp

ROOT = Path(__file__).resolve().parents[1]
PLAN = pd.read_csv(ROOT / "docs/field_validation/take_plan.csv", dtype=str, keep_default_na=False)
PROTOCOL = (ROOT / "docs/field_validation/PROTOCOL.md").read_text(encoding="utf-8")


def test_plan_has_35_takes_per_person_in_two_batches() -> None:
    assert len(PLAN) == 105 and PLAN["take_id"].is_unique
    assert (PLAN["take_id"] == PLAN["speaker"] + "_" + PLAN["take"]).all()
    assert PLAN.groupby("speaker").size().to_dict() == {"P1": 35, "P2": 35, "P3": 35}
    # the protocol's count table: batch 1 is 60 genuine and 15 cloned, batch 2 15 and 15
    counts = PLAN.groupby(["batch", "label"]).size().to_dict()
    assert counts == {
        ("1", "bonafide"): 60,
        ("1", "spoof"): 15,
        ("2", "bonafide"): 15,
        ("2", "spoof"): 15,
    }


def test_only_the_clone_takes_are_labelled_spoof() -> None:
    clone = PLAN["kind"].isin({"clone_replay", "call_clone"})
    assert set(PLAN.loc[clone, "label"]) == {"spoof"}
    assert set(PLAN.loc[~clone, "label"]) == {"bonafide"}
    assert set(PLAN.loc[clone, "tool"]) == set(rcp.TOOLS)
    assert set(PLAN.loc[~clone, "tool"]) == {"none"}


def test_clone_takes_play_the_files_the_real_call_pack_wrote() -> None:
    clone = PLAN[PLAN["label"] == "spoof"]
    expected = clone.apply(
        lambda r: rcp.playback_name(r["tool"], r["speaker"], r["script"]), axis=1
    )
    assert (clone["play"] == expected).all()


def test_every_take_in_the_plan_is_in_the_protocol() -> None:
    for take in sorted(set(PLAN["take"])):
        number = int(take[1:])
        listed = re.search(rf"\| {take} \|", PROTOCOL)
        in_range = take.startswith("R") and 1 <= number <= 10 and "| R01-R10 |" in PROTOCOL
        assert listed or in_range, f"{take} is not in PROTOCOL.md"
