"""Tests for the variant selection replay and the exploratory seed comparison."""

from __future__ import annotations

import pytest

from src.reporting import variants as v


def _row(name, dev, options=None, status="ok", block="variants"):
    return {
        "name": name,
        "dev_eer": dev,
        "options": options or {},
        "status": status,
        "block": block,
        "eval_pool": 0.03,
        "cm04": 0.3,
        "rvc_test": 0.05,
    }


def test_lowest_dev_wins_outright():
    pool = [_row("published", 0.055), _row("crop", 0.040, {"crop_per_epoch": True})]
    assert v.pick(pool)["name"] == "crop"


def test_a_tie_goes_to_fewer_changes():
    pool = [
        _row("published", 0.0550),
        _row("crop_aug", 0.0534, {"crop_per_epoch": True, "augment": "channel_v1"}),
    ]
    assert v.pick(pool)["name"] == "published"  # 0.16 points apart: a tie


def test_failed_runs_are_not_eligible():
    pool = [_row("published", 0.055), _row("layers", 0.01, {"x": 1}, status="failed")]
    assert v.pick(pool)["name"] == "published"
    assert v.pick([]) is None


def test_mann_whitney_exact_on_complete_separation():
    # five against five with no overlap: 2 of the 252 splits are as extreme
    assert v.mann_whitney_exact([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]) == pytest.approx(2 / 252)
    # four against five: 2 of 126
    assert v.mann_whitney_exact([1, 2, 3, 4], [6, 7, 8, 9, 10]) == pytest.approx(2 / 126)


def test_mann_whitney_exact_on_identical_groups_is_one():
    assert v.mann_whitney_exact([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        v.mann_whitney_exact([], [1.0])


def test_candidates_take_the_published_run_and_every_finished_variant():
    run1 = [
        _row("deployed_s1234", 0.055, block="deployed_seeds"),
        _row("deployed_s7", 0.10, block="deployed_seeds"),
        _row("deployed_crop_s1234", 0.091, {"crop_per_epoch": True}),
        _row("deployed_crop_layers_s1234", None, {"layer_weighting": True}, status="failed"),
    ]
    layers = [_row("deployed_crop_layers_s1234", 0.091, {"layer_weighting": True})]
    names = [r["name"] for r in v.candidates(run1, layers)]
    assert names == ["deployed_s1234", "deployed_crop_s1234", "deployed_crop_layers_s1234"]
