"""Smoke tests for src.reporting.cmi_bands."""

import pandas as pd

from src.reporting import cmi_bands


def test_with_cmi_bands_and_flags():
    scores = pd.DataFrame(
        {
            "utt_id": ["u1", "u2", "u3"],
            "label": ["bonafide", "bonafide", "spoof"],
            "speaker": ["a", "b", "a"],
            "score": [0.2, 0.9, 0.1],
        }
    )
    index = pd.DataFrame(
        {"utt_id": ["u1", "u2", "u3"], "transcript": ["यह एक test है", "सिर्फ हिंदी वाक्य", "x"]}
    )
    out = cmi_bands.with_cmi(scores, index).set_index("utt_id")
    assert list(out.index) == ["u1", "u2"]
    assert out.loc["u1", "band"] == "10-25" and bool(out.loc["u1", "false_alarm"])
    assert out.loc["u2", "band"] == "0-10" and not bool(out.loc["u2", "false_alarm"])
    table = cmi_bands.band_table(out.reset_index()).set_index("band")
    assert table.loc["10-25", "false_alarm_rate"] == 1.0
