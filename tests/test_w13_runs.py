"""Smoke tests for src.reporting.w13_runs."""

import json
import zipfile

import pytest

from src.reporting import w13_runs as w


def _row(system, seed, eer, key="system"):
    return {
        "name": f"{system}_s{seed}",
        key: system,
        "seed": seed,
        "status": "ok",
        "eval_pool": eer,
        "cm04": eer + 0.1,
    }


def test_sign_flip_p_is_exact():
    assert w.sign_flip_p([-1.0, -1.0, -1.0]) == pytest.approx(2 / 8)
    assert w.sign_flip_p([1.0, -1.0]) == 1.0


def test_compare_pairs_by_seed():
    a = [_row("ext", s, 0.1) for s in (1, 2, 3)]
    b = [_row("deployed", s, 0.2) for s in (1, 2, 3, 4)]
    out = w.compare(a, b)["eval_pool"]
    assert out["paired_seeds"] == [1, 2, 3]
    assert out["diff_mean"] == pytest.approx(-0.1) and out["a_better"] == 3
    assert out["b"]["n"] == 4


def test_analyse_reads_zips_and_writes_tables(tmp_path):
    run1 = [_row("ext", s, 0.1) for s in (1, 2)] + [_row("deployed", s, 0.2) for s in (1, 2)]
    run2 = [{"name": "zero_shot", "data": None, "seed": None, "status": "ok", "eval_pool": 0.43}]
    run2 += [_row("s2data", s, 0.3, key="data") for s in (1, 2)]
    for name, rows in (("run1.zip", run1), ("run2.zip", run2)):
        with zipfile.ZipFile(tmp_path / name, "w") as zf:
            zf.writestr("results/runs.json", json.dumps(rows))
    report = w.analyse(str(tmp_path / "run1.zip"), str(tmp_path / "run2.zip"))
    assert report["run2_zero_shot"]["eval_pool"] == 0.43
    assert report["run2_aasist_s2data_vs_deployed"]["eval_pool"]["diff_mean"] == pytest.approx(0.1)
    text = w.markdown(report)
    assert "AASIST zero-shot" in text and "Run 1: extended recipe" in text


def test_analyse_ssl_pools_the_two_ssl_runs_and_pairs_by_seed(tmp_path):
    adapter = [_row("final", seed, 0.2) for seed in (1, 2, 3, 4, 5)]
    run7 = [_row("final", seed, 0.1, key="data") for seed in (1, 2, 3)]
    run8 = [_row("final", seed, 0.1, key="data") for seed in (4, 5)]
    for name, rows in (("a.zip", adapter), ("b.zip", run7), ("c.zip", run8)):
        with zipfile.ZipFile(tmp_path / name, "w") as zf:
            zf.writestr("x/runs.json", json.dumps(rows))
    report = w.analyse_ssl(
        str(tmp_path / "a.zip"), str(tmp_path / "b.zip"), str(tmp_path / "c.zip")
    )
    entry = report["run7_ssl_vs_final"]["eval_pool"]
    assert entry["paired_seeds"] == [1, 2, 3, 4, 5]
    assert entry["diff_mean"] == pytest.approx(-0.1) and entry["a_better"] == 5
    assert "SSL-AASIST vs the adapter" in w.markdown(report)
