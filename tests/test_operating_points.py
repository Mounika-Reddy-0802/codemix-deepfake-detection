"""The old/new operating-point comparison reads files only and judges files as the voice check."""

from __future__ import annotations

import pandas as pd

from src.reporting import operating_points as op


def _windows(rows: list[tuple[str, str, list[float]]]) -> pd.DataFrame:
    out = []
    for path, label, scores in rows:
        for w, s in enumerate(scores):
            out.append(
                {
                    "filepath": path,
                    "label": label,
                    "tool": "none" if label == "bonafide" else "xtts_v2",
                    "window": w,
                    "end_seconds": 4.0 + 2.0 * w,
                    "level_dbfs": -25.0,
                    "score": s,
                    "smoothed": s,
                }
            )
    return pd.DataFrame(out)


def test_the_voice_check_decides_by_the_mean_window_score():
    windows = _windows(
        [
            ("g1", "bonafide", [0.9, 0.2]),  # mean 0.55: real at 0.5, cloned at 0.6
            ("g2", "bonafide", [0.8, None]),  # silent window left out: mean 0.8
            ("c1", "spoof", [0.1, 0.7]),  # mean 0.4: caught at either threshold
        ]
    )
    at_05 = op.file_verdicts(windows, 0.5)
    assert at_05 == {
        "genuine_files": 2,
        "genuine_called_cloned_pct": 0.0,
        "cloned_files": 1,
        "cloned_called_real_pct": 0.0,
    }
    assert op.file_verdicts(windows, 0.6)["genuine_called_cloned_pct"] == 50.0


def test_compare_reads_both_threshold_files_and_every_window_set(tmp_path):
    (tmp_path / "configs").mkdir()
    for name, threshold in (("threshold_final", 0.7), ("threshold_final_spont", 0.4)):
        (tmp_path / f"configs/{name}.yaml").write_text(
            f"threshold: {threshold}\nsuspicious_after: 2\nfake_after: 4\n"
        )
    folder = tmp_path / "experiments/calibration_final"
    folder.mkdir(parents=True)
    genuine = _windows([("g", "bonafide", [0.55] * 6), ("f", "spoof", [0.1] * 6)])
    genuine.to_csv(folder / "dev_windows.csv", index=False)
    genuine.to_csv(folder / "hiacc_windows.csv", index=False)

    report = op.compare("final", root=str(tmp_path))

    assert report["points"]["old"]["threshold"] == 0.7
    assert report["points"]["new"]["threshold"] == 0.4
    assert report["chosen_on"] == ["dev"]
    assert set(report["sets"]) == {"dev", "hiacc"}  # no real-call windows in this repository
    hiacc = report["sets"]["hiacc"]
    assert hiacc["old"]["voice_check"]["genuine_called_cloned_pct"] == 100.0
    assert hiacc["new"]["voice_check"]["genuine_called_cloned_pct"] == 0.0
    assert hiacc["new"]["voice_check"]["cloned_called_real_pct"] == 0.0
    assert hiacc["old"]["live_per_clip"]["real_calls_alerted_pct"] == 100.0
    assert hiacc["new"]["live_per_clip"]["real_calls_alerted_pct"] == 0.0
    assert "live_calls_60s" in hiacc["new"]


def test_separation_is_the_eer_of_spontaneous_genuine_files_against_each_tools_clones(tmp_path):
    genuine = _windows([(f"g{i}", "bonafide", [0.9, 0.8]) for i in range(4)])
    genuine.to_csv(tmp_path / "hiacc_windows.csv", index=False)
    caught = _windows([(f"t{i}", "spoof", [0.1]) for i in range(3)])  # all below every genuine
    caught.to_csv(tmp_path / "cm04_windows.csv", index=False)
    passing = _windows([(f"i{i}", "spoof", [0.95]) for i in range(3)])  # all above
    passing.to_csv(tmp_path / "cm07_windows.csv", index=False)

    out = op.separation(tmp_path)

    assert out["genuine_files"] == 4
    assert out["tortoise"] == {"clone_files": 3, "eer_pct": 0.0}
    assert out["indicf5"]["eer_pct"] == 100.0
    assert "chatterbox" not in out  # no cm09 windows here
