"""Tests for seed stability reporting (src/reporting/seeds.py)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.reporting import seeds as sd


def test_mean_interval_matches_hand_computation():
    out = sd.mean_interval([2.0, 4.0, 6.0])
    assert out["mean"] == 4.0 and out["sd"] == pytest.approx(2.0)
    half = 4.303 * 2.0 / np.sqrt(3)
    assert out["mean_ci"] == pytest.approx([4.0 - half, 4.0 + half])
    assert (out["min"], out["max"], out["n"]) == (2.0, 6.0, 3)


def test_one_seed_has_no_spread():
    out = sd.mean_interval([3.0])
    assert out["sd"] is None and out["mean_ci"] is None


def _clips(n_speakers=8, per=20, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_speakers):
        for k in range(per):
            for label in ("bonafide", "spoof"):
                rows.append(
                    {
                        "filepath": f"{s}_{k}_{label}",
                        "label": label,
                        "speaker": f"spk{s}",
                        "floor": rng.random(),
                    }
                )
    return pd.DataFrame(rows)


def test_floor_test_on_a_perfect_detector_beats_a_random_floor():
    floor = _clips()
    scores = floor.assign(score=(floor["label"] == "bonafide").astype(float))[["filepath", "score"]]
    r = sd.floor_test(scores, floor, n_boot=200)
    assert r["eer"] == 0.0 and r["verdict"] == "beats floor"


def test_floor_test_refuses_missing_clips():
    floor = _clips()
    scores = floor.assign(score=0.5)[["filepath", "score"]].iloc[:-3]
    with pytest.raises(ValueError, match="cover"):
        sd.floor_test(scores, floor)


def test_analyse_end_to_end(tmp_path, monkeypatch):
    # a fake repo layout: published seed-1234 run, floor files, and a review results dir
    system = sd.SeededSystem("xtts_channel", "toy", str(tmp_path / "pub/toy"), "toy", False)
    monkeypatch.setattr(sd, "SYSTEMS", (system,))
    floor_dir = tmp_path / "floor"
    floor_dir.mkdir()
    (tmp_path / "pub").mkdir()
    results = tmp_path / "review"
    results.mkdir()
    floor = _clips(seed=1)
    good = (floor["label"] == "bonafide").astype(float)
    runs = []
    for ev in sd.SETS:
        floor.to_csv(floor_dir / f"toy__{ev}_paired.csv", index=False)
        floor.assign(score=good)[["filepath", "score"]].to_csv(
            tmp_path / f"pub/toy__{ev}_scores.csv", index=False
        )
        (tmp_path / f"pub/toy__{ev}.json").write_text(json.dumps({"pooled": {"eer": 0.0}}))
    for seed in (7, 42):
        name = f"xttsch_s{seed}"
        runs.append(
            {
                "name": name,
                "block": "xtts_channel_seeds",
                "system": "xtts_channel",
                "seed": seed,
                "options": {},
                "status": "ok",
            }
        )
        for ev in sd.SETS:
            floor.assign(score=good)[["filepath", "score"]].to_csv(
                results / f"{name}__{ev}_scores.csv", index=False
            )
    runs.append(
        {
            "name": "variant",
            "block": "variants",
            "system": "xtts_channel",
            "seed": 1234,
            "options": {"crop_per_epoch": True},
            "status": "ok",
        }
    )  # must be ignored
    (results / "runs.json").write_text(json.dumps(runs))

    report = sd.analyse(str(results), str(floor_dir), n_boot=100)
    toy = report["xtts_channel"]
    assert toy["seeds"] == [7, 42, 1234]
    cm04 = toy["sets"]["cm04"]
    assert cm04["eer"]["n"] == 3 and cm04["eer"]["mean"] == 0.0
    assert cm04["seeds_beating_floor"] == 3
    assert "| toy | cm04 | 3 |" in sd.markdown(report)
