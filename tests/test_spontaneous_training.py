"""The pre-registered hypotheses of the spontaneous recipe, coded and checked on made-up runs."""

from __future__ import annotations

import json

from src.reporting import spontaneous_training as stv


def _row(seed: int, hiacc: float, cm04: float, cm07: float, cm09: float | None = None) -> dict:
    row = {"seed": seed, "hiacc_fa": hiacc, "cm04": cm04, "cm07": cm07}
    if cm09 is not None:
        row["cm09"] = cm09
    return row


FROZEN_ADAPTER = {
    1234: _row(1234, 0.439, 0.175, 0.257),
    2025: {"seed": 2025, "cm04": 0.166, "cm07": 0.181},  # no HiACC value kept
    7: {"seed": 7, "cm04": 0.157, "cm07": 0.232},
}


def test_the_adapter_helps_when_hiacc_falls_and_unseen_tools_hold():
    new = {
        s: _row(s, 0.20, FROZEN_ADAPTER[s]["cm04"] + 0.01, FROZEN_ADAPTER[s]["cm07"])
        for s in (1234, 2025, 7)
    }
    r = stv.judge("adapter", new, FROZEN_ADAPTER)
    assert r["h1"] and r["h2"] and r["verdict"] == "helps"
    assert abs(r["sets"]["cm04"]["paired_mean_change"] - 0.01) < 1e-9
    assert r["sets"]["hiacc_fa"]["paired_seeds"] == 1  # only seed 1234 has a frozen value


def test_a_cost_on_unseen_tools_makes_it_a_trade_off():
    new = {
        s: _row(s, 0.20, FROZEN_ADAPTER[s]["cm04"], FROZEN_ADAPTER[s]["cm07"] + 0.05)
        for s in (1234, 2025, 7)
    }
    assert stv.judge("adapter", new, FROZEN_ADAPTER)["verdict"] == "trade-off"


def test_a_small_hiacc_drop_does_not_help():
    new = {
        s: _row(s, 0.36, FROZEN_ADAPTER[s]["cm04"], FROZEN_ADAPTER[s]["cm07"])
        for s in (1234, 2025, 7)
    }  # 7.9 points lower: short of the 10 asked
    r = stv.judge("adapter", new, FROZEN_ADAPTER)
    assert not r["h1"] and r["verdict"] == "does not help"


def test_a_missing_seed_leaves_the_verdict_incomplete():
    new = {1234: _row(1234, 0.20, 0.175, 0.257)}
    r = stv.judge("adapter", new, FROZEN_ADAPTER)
    assert r["verdict"] == "incomplete" and r["h1"] is None
    assert r["sets"]["hiacc_fa"]["per_seed"][1234]["frozen"] == 0.439


def test_ssl_needs_every_seed_within_two_points_of_its_frozen_seed():
    frozen = {
        1234: _row(1234, 0.058, 0.017, 0.258, 0.126),
        2025: _row(2025, 0.467, 0.007, 0.149, 0.124),
    }
    good = {
        1234: _row(1234, 0.07, 0.017, 0.258, 0.126),
        2025: _row(2025, 0.20, 0.007, 0.149, 0.124),
    }
    assert stv.judge("ssl", good, frozen)["verdict"] == "helps"
    worse = {
        1234: _row(1234, 0.09, 0.017, 0.258, 0.126),
        2025: _row(2025, 0.10, 0.007, 0.149, 0.124),
    }
    assert stv.judge("ssl", worse, frozen)["h1"] is False  # seed 1234 up 3.2 points


def test_runs_files_load_by_seed_and_the_frozen_runs_are_the_committed_ones(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps([_row(1234, 0.3, 0.1, 0.2)]))
    b.write_text(json.dumps([_row(2025, 0.3, 0.1, 0.2), _row(7, 0.3, 0.1, 0.2)]))
    assert sorted(stv.load_runs([str(a), str(b)])) == [7, 1234, 2025]
    frozen = stv.frozen_runs("adapter")
    assert abs(frozen[1234]["hiacc_fa"] - 0.4385) < 1e-3
    assert abs(frozen[2025]["cm04"] - 0.1660) < 1e-3
    ssl = stv.frozen_runs("ssl")
    assert abs(ssl[2025]["hiacc_fa"] - 0.4668) < 1e-3
