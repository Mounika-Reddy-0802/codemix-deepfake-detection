"""The generator study's speaker split, its manifests, and the guard's narrow exception."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data import generator_study as gs
from src.training import config_guard as cg

POOLS = Path("data/manifests") / gs.POOLS_OUT


def _pools() -> pd.DataFrame:
    return pd.read_csv(POOLS, dtype={"speaker": str})


def test_the_split_on_disk_is_the_frozen_one():
    assert gs.frame_hash(_pools()) == gs.FROZEN_SHA256


def test_the_split_is_rebuilt_identically_from_the_manifests():
    assert gs.frame_hash(gs.split_speakers()) == gs.FROZEN_SHA256


def test_halves_are_disjoint_and_balanced_within_each_pool():
    pools = _pools()
    assert not pools["speaker"].duplicated().any()
    counts = pools.groupby(["pool", "half"]).size().unstack()
    assert (counts["gen_train"] - counts["gen_test"]).abs().max() <= 1


@pytest.mark.parametrize("system", sorted(gs.SYSTEMS))
def test_each_manifest_adds_only_its_tools_from_the_training_half(system):
    out = pd.read_csv(f"data/manifests/genstudy_train_{system}.csv", dtype={"speaker": str})
    train = gs.train_speakers(str(POOLS))
    tool = out["tool"].str.lower()
    assert not tool.eq("tortoise").any()
    clones = out[tool.isin(cg.STUDY_TOOLS)]
    assert set(clones["speaker"]) <= train
    assert set(clones["tool"].str.lower()) == set(gs.SYSTEMS[system])
    extra = out[out["speaker"].isin(set(_pools()["speaker"]))]
    assert set(extra["speaker"]) <= train
    assert out.equals(gs.build(system))


def test_test_speakers_masks_out_the_training_half():
    train = {"a", "b"}
    mask = gs.test_speakers(pd.Series(["a", "c", "b", "d"]), train)
    assert mask.tolist() == [False, True, False, True]


def _row(**overrides) -> pd.DataFrame:
    row = {
        "filepath": "${DATA_ROOT}/clips/a.wav",
        "label": "spoof",
        "language": "hi",
        "speaker": "s1",
        "source": "cm07",
        "tool": "indicf5",
        "condition": "channel_g711_20db",
        "split": "train",
    }
    row.update(overrides)
    return pd.DataFrame([row])


STUDY = {"system": cg.GENERATOR_STUDY, "lora": True}


def test_the_study_may_train_on_its_half_of_the_clones():
    rows = _row(speaker="s1")
    assert cg.check_rows(rows, STUDY, {"s1"}, set(), set(), {"s1"}) == []


def test_clones_of_the_held_out_half_are_refused():
    problems = cg.check_rows(_row(speaker="s2"), STUDY, {"s2"}, set(), set(), {"s1"})
    assert any("held-out tool" in p for p in problems)
    assert any("eval-pool speakers" in p for p in problems)


def test_tortoise_is_refused_even_in_the_study():
    rows = _row(speaker="s1", tool="tortoise", source="cm04")
    problems = cg.check_rows(rows, STUDY, set(), set(), set(), {"s1"})
    assert any("held-out tool" in p for p in problems)


def test_clones_are_refused_outside_the_study():
    problems = cg.check_rows(_row(), {"lora": True}, {"s1"}, set(), set(), {"s1"})
    assert any("held-out tool" in p for p in problems)


def test_the_study_must_be_a_lora_run():
    problems = cg.check_rows(_row(), {"system": cg.GENERATOR_STUDY}, set(), set(), set(), {"s1"})
    assert any("LoRA" in p for p in problems)
