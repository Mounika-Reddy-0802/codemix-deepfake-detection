"""Smoke tests for src.reporting.naturalness (no UTMOS download: the model is injected)."""

import numpy as np
import pandas as pd

from src.reporting import naturalness as nat


def test_sample_is_seeded_and_capped():
    frame = pd.DataFrame({"x": range(50)})
    a, b = nat.sample(frame, 10, seed=3), nat.sample(frame, 10, seed=3)
    assert len(a) == 10 and a["x"].tolist() == b["x"].tolist()
    assert len(nat.sample(frame, 100)) == 50


def test_summarise_interval_brackets_mean():
    r = nat.summarise(np.array([3.0, 3.5, 4.0, 4.5]))
    assert r["mos_mean"] == 3.75 and r["ci_low"] <= 3.75 <= r["ci_high"]


def test_run_with_an_injected_predictor(tmp_path):
    sf = __import__("pytest").importorskip("soundfile")
    (tmp_path / "clips").mkdir()
    rows = []
    for i in range(3):
        sf.write(tmp_path / "clips" / f"g{i}.wav", np.zeros(1600, dtype=np.float32), 16_000)
        rows.append(
            {"filepath": f"${{DATA_ROOT}}/clips/g{i}.wav", "label": "bonafide", "tool": "none"}
        )
    pd.DataFrame(rows).to_csv(tmp_path / "eval_ext_genuine.csv", index=False)
    report = nat.run(str(tmp_path), str(tmp_path), 2, lambda wave, sr: 4.0)
    assert list(report) == ["genuine, new test speakers"]
    assert report["genuine, new test speakers"]["clips"] == 2
    assert "| genuine, new test speakers | 2 | 4.00" in nat.markdown(report)
