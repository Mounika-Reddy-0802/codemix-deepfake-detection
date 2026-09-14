"""The ablation table must read every EER against its own floor, and state the answer."""

from __future__ import annotations

import pytest

from src.reporting import ablation as ab


def _summary(eers: dict[str, float]) -> dict:
    out = {}
    for run in ab.ADAPTERS:
        for ev in ab.SETS:
            e = eers.get(f"{run}__{ev}", 0.02)
            out[f"{run}__{ev}"] = {
                "eer": e,
                "eer_ci_low": e - 0.01,
                "eer_ci_high": e + 0.01,
                "auc": 0.99,
                "clips": 100,
            }
    return out


def _floors(value: float = 20.0) -> dict[tuple[str, str], float]:
    return {key: value for key in ab.FLOORS}


def test_table_has_one_row_per_adapter_and_set():
    frame = ab.table(_summary({}), _floors())
    assert len(frame) == len(ab.ADAPTERS) * len(ab.SETS)
    assert set(frame["condition"]) == {"clean", "channel"}


def test_a_cell_above_its_floor_does_not_clear_it():
    frame = ab.table(_summary({"lora_norm_channel__cm04": 0.28}), _floors(25.0))
    cell = frame[(frame.adapter == "lora_norm_channel") & (frame.set == "cm04")].iloc[0]
    assert cell["margin"] == pytest.approx(-3.0)
    assert not cell["clears_floor"]


def test_the_ci_upper_bound_decides_clearing():
    # EER 19% sits below a 19.5% floor, but its CI reaches 20%: not cleared.
    frame = ab.table(_summary({"lora_norm_clean__eval_pool": 0.19}), _floors(19.5))
    cell = frame[(frame.adapter == "lora_norm_clean") & (frame.set == "eval_pool")].iloc[0]
    assert not cell["clears_floor"]


def test_verdict_names_the_trade_off():
    eers = {
        "lora_norm_clean__cm04": 0.05,
        "lora_norm_rvc_clean__cm04": 0.12,
        "lora_norm_channel__cm04": 0.28,
        "lora_norm_rvc_channel__cm04": 0.31,
        "lora_norm_clean__rvc_test": 0.24,
        "lora_norm_rvc_clean__rvc_test": 0.05,
        "lora_norm_channel__rvc_test": 0.30,
        "lora_norm_rvc_channel__rvc_test": 0.05,
    }
    effect = ab.rvc_effect(ab.table(_summary(eers), _floors()))
    assert (effect[effect.set == "cm04"]["delta"] > 0).all()
    assert ab.verdict(effect).startswith("No.")


def test_verdict_yes_when_rvc_helps_without_cost():
    eers = {"lora_norm_clean__rvc_test": 0.24, "lora_norm_channel__rvc_test": 0.30}
    effect = ab.rvc_effect(ab.table(_summary(eers), _floors()))
    assert ab.verdict(effect).startswith("Yes.")


def test_missing_run_is_an_error():
    summary = _summary({})
    summary.pop("lora_norm_clean__cm04")
    with pytest.raises(ab.AblationDataError, match="lora_norm_clean__cm04"):
        ab.table(summary, _floors())


def test_missing_floor_is_an_error(tmp_path):
    with pytest.raises(ab.AblationDataError, match="not measured"):
        ab.load_floors({("cm04", "clean"): str(tmp_path / "absent.json")})


def test_committed_ablation_matches_the_committed_summary():
    frame = ab.table(__import__("json").load(open(ab.SUMMARY)), ab.load_floors())
    committed = __import__("json").load(open(ab.OUT))
    assert [c["eer"] for c in committed["cells"]] == frame["eer"].tolist()
