"""Smoke tests for src.reporting.any_audio_accuracy with injected scores (no files read)."""

import pandas as pd
import pytest

from src.reporting import any_audio_accuracy as acc


@pytest.fixture
def fake_sources(monkeypatch):
    scores = pd.DataFrame(
        {
            "label": ["bonafide", "bonafide", "spoof", "spoof", "spoof"],
            "tool": ["none", "none", "xtts_v2", "xtts_v2", "tortoise"],
            "p": [0.9, 0.4, 0.1, 0.7, 0.2],
        }
    )
    monkeypatch.setattr(acc, "sources", lambda: {"final": {"zip": {"new_speakers": scores}}})


def test_importing_reads_no_files():
    # the module must import on any machine (CI); files are read only on first use
    assert callable(acc.sources)


def test_rate_counts_correct_decisions_at_the_frozen_threshold(fake_sources):
    # final's threshold is 0.687: one of two genuine clips is above it
    assert acc.rate("final", "new_speakers", ("bonafide", None), "bonafide") == "50.0 (n=2)"
    # one of two XTTS clips is below it
    assert acc.rate("final", "new_speakers", ("spoof", "xtts_v2"), "spoof") == "50.0 (n=2)"
    assert acc.rate("final", "new_speakers", ("spoof", "tortoise"), "spoof") == "100.0 (n=1)"


def test_missing_sets_show_a_dash(fake_sources):
    assert acc.rate("final", "cm09", ("spoof", None), "spoof") == "-"


def test_retrained_columns_read_log_odds_and_keep_the_reference_hiacc_clips(tmp_path, monkeypatch):
    manifest = tmp_path / "hiacc.csv"
    pd.DataFrame({"filepath": ["${DATA_ROOT}/clips/AD01.wav"], "label": ["bonafide"]}).to_csv(
        manifest, index=False
    )
    monkeypatch.setattr(acc, "HIACC_CALIBRATION", manifest)
    hiacc = pd.DataFrame(
        {
            "filepath": ["${DATA_ROOT}/clips/AD01.wav", "${DATA_ROOT}/clips/AD02.wav"],
            "label": ["bonafide"] * 2,
            "score": [0.0, 3.0],
        }
    )
    hiacc.to_csv(tmp_path / "spont_s1234__hiacc_scores.csv", index=False)
    cm07 = pd.DataFrame({"label": ["spoof"], "tool": ["indicf5"], "score": [-2.0]})
    cm07.to_csv(tmp_path / "spont_s1234__cm07_scores.csv", index=False)

    out = acc.spontrain_scores(tmp_path, "ssl_spontrain_s1234")

    assert set(out) == {"hiacc", "cm07"}
    assert out["hiacc"]["filepath"].tolist() == ["${DATA_ROOT}/clips/AD01.wav"]  # AD02 not in it
    assert out["hiacc"]["p"].iloc[0] == pytest.approx(0.5)  # log-odds 0 -> P(genuine) 0.5
    assert out["cm07"]["p"].iloc[0] < 0.5
