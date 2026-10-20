"""Smoke tests for src.reporting.amr_channel."""

import pandas as pd

from src.reporting import amr_channel as ar


def _scores(spoof_score):
    rows = []
    for spk in ("a", "b", "c"):
        rows += [
            {"label": "bonafide", "tool": "none", "speaker": spk, "score": 0.9},
            {"label": "spoof", "tool": "xtts_v2", "speaker": spk, "score": spoof_score},
        ]
    return pd.DataFrame(rows)


def test_conditions_read_from_their_own_folders(tmp_path):
    ext, amr = tmp_path / "ext", tmp_path / "amr"
    ext.mkdir()
    amr.mkdir()
    _scores(0.1).to_csv(ext / "deployed_s2__score_ext_channel_scores.csv", index=False)
    _scores(0.95).to_csv(amr / "deployed_s2__score_ext_amr_scores.csv", index=False)
    rows = {r["condition"]: r for r in ar.analyse(str(ext), str(amr), n_boot=20)}
    assert rows["G.711"]["new XTTS (CM05)"]["eer"] == 0.0
    assert rows["AMR-NB"]["new XTTS (CM05)"]["eer"] == 1.0
    assert "| Deployed S2 (trained on G.711) | AMR-NB | 100.0" in ar.markdown(list(rows.values()))
