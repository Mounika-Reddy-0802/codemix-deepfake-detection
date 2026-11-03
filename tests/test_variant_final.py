"""The A5 decision rule on hand-made runs (no score file is read)."""

from __future__ import annotations

from src.reporting import variant_final as vf

FINAL = {  # the final recipe's real IndicF5 and Tortoise EERs (final_v1, run 4)
    7: {"cm07": 0.2319, "new_tortoise": 0.0679},
    42: {"cm07": 0.2413, "new_tortoise": 0.0892},
    99: {"cm07": 0.1733, "new_tortoise": 0.1080},
    1234: {"cm07": 0.2566, "new_tortoise": 0.0900},
    2025: {"cm07": 0.1809, "new_tortoise": 0.0855},
}


def _variant(cm07: list[float], tortoise_shift: float = 0.0) -> dict[int, dict]:
    return {
        s: {"cm07": c, "new_tortoise": FINAL[s]["new_tortoise"] + tortoise_shift}
        for s, c in zip(vf.SEEDS, cm07, strict=True)
    }


def test_clearly_lower_on_every_seed_is_confirmed():
    out = vf.judge(_variant([0.10, 0.11, 0.09, 0.12, 0.10]), FINAL)
    assert out["lower_in_seeds"] == 5 and out["mann_whitney_p"] < 0.01
    assert out["decision"] == "confirmed"


def test_lower_on_four_seeds_without_a_significant_shift_is_not_confirmed():
    # the variant's values sit among the final recipe's, so the test cannot separate them
    out = vf.judge(_variant([0.2500, 0.1700, 0.2300, 0.2400, 0.1800]), FINAL)
    assert out["conditions"]["1_lower_in_4_of_5"] is True
    assert out["conditions"]["2_mann_whitney"] is False
    assert out["decision"] == "not confirmed"


def test_a_tortoise_cost_above_one_point_blocks_confirmation():
    out = vf.judge(_variant([0.10, 0.11, 0.09, 0.12, 0.10], tortoise_shift=0.011), FINAL)
    assert out["conditions"]["3_tortoise_within_1_point"] is False
    assert out["decision"] == "not confirmed"


def test_fewer_than_five_seeds_is_incomplete():
    variant = _variant([0.10, 0.11, 0.09, 0.12, 0.10])
    del variant[99]
    assert vf.judge(variant, FINAL)["decision"] == "incomplete"


def test_only_finished_runs_of_the_named_system_count():
    rows = [
        {"system": "variant", "seed": 7, "status": "ok", "cm07": 0.1},
        {"system": "variant", "seed": 42, "status": "failed", "cm07": None},
        {"system": "final", "seed": 7, "status": "ok", "cm07": 0.2},
    ]
    assert set(vf.by_seed(rows, "variant")) == {7}
