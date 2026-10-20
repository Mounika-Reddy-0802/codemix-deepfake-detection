"""Smoke tests for src.reporting.new_speakers."""

import pandas as pd

from src.reporting import new_speakers as ns


def _scores():
    rows = []
    for spk in ("a", "b", "c", "d"):
        rows += [
            {"label": "bonafide", "tool": "none", "speaker": spk, "score": 0.9},
            {"label": "spoof", "tool": "xtts_v2", "speaker": spk, "score": 0.1},
            {"label": "spoof", "tool": "tortoise", "speaker": spk, "score": 0.95},
        ]
    return pd.DataFrame(rows)


def test_attack_eer_separates_tools():
    assert ns.attack_eer(_scores(), "xtts_v2", n_boot=50)["eer"] == 0.0
    assert ns.attack_eer(_scores(), "tortoise", n_boot=50)["eer"] == 1.0


def test_analyse_and_markdown(tmp_path):
    _scores().to_csv(tmp_path / "deployed_s2__score_ext_channel_scores.csv", index=False)
    rows = ns.analyse(str(tmp_path), n_boot=20)
    assert len(rows) == 1 and rows[0]["condition"] == "channel"
    text = ns.markdown(rows)
    assert "(deployed) | phone line | 0.0" in text and "| - |" in text
