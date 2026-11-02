"""Tests for voice_check.app: the answer card and the consent-gated example log."""

import csv

import numpy as np

from src.inference import any_audio as aa
from voice_check import app


def _result(p: float | None, verdict: str) -> aa.VoiceVerdict:
    windows = () if p is None else (aa.WindowScore(0.0, 4.0, -23.0, p),)
    return aa.VoiceVerdict(verdict, p, 0.58, 6.0, 4.0 if p else 1.0, "g711", windows)


def test_card_states_the_answer_and_its_numbers():
    html = app.verdict_html(_result(0.12, aa.CLONED))
    assert "Cloned voice" in html and "0.120" in html and "0.580" in html
    assert "Real voice" in app.verdict_html(_result(0.97, aa.REAL))
    assert "Not enough speech" in app.verdict_html(_result(None, aa.NO_SPEECH))


def test_nothing_is_kept_without_consent(tmp_path):
    clip = tmp_path / "x.wav"
    clip.write_bytes(b"RIFF")
    status = app.save_example(
        str(clip),
        "Real human voice",
        "Spoken live into this microphone",
        "",
        False,
        _result(0.9, aa.REAL),
        "ckpt.pt",
        folder=tmp_path / "field",
    )
    assert status.startswith("Not saved")
    assert not (tmp_path / "field").exists()


def test_a_labelled_recording_becomes_a_manifest_row(tmp_path):
    clip = tmp_path / "x.webm"
    clip.write_bytes(np.arange(10, dtype=np.int16).tobytes())
    folder = tmp_path / "field"
    status = app.save_example(
        str(clip),
        "Cloned or synthetic voice",
        "Played aloud from another device into the microphone",
        "XTTS-v2",
        True,
        _result(0.2, aa.CLONED),
        "ckpt.pt",
        folder=folder,
    )
    assert status == "Saved (1 labelled recordings so far)."
    rows = list(csv.DictReader((folder / "log.csv").open(encoding="utf-8")))
    assert rows[0]["label"] == "spoof" and rows[0]["tool"] == "XTTS-v2"
    assert rows[0]["condition"] == "mic_replay" and rows[0]["set"] == "field"
    assert rows[0]["filepath"].endswith(".webm")
    assert (folder / "clips").exists() and len(list((folder / "clips").iterdir())) == 1


def test_a_real_voice_is_never_given_a_tool(tmp_path):
    clip = tmp_path / "x.wav"
    clip.write_bytes(b"RIFF")
    app.save_example(
        str(clip),
        "Real human voice",
        "Phone-call recording",
        "XTTS-v2",
        True,
        _result(0.9, aa.REAL),
        "ckpt.pt",
        folder=tmp_path,
    )
    row = next(csv.DictReader((tmp_path / "log.csv").open(encoding="utf-8")))
    assert row["label"] == "bonafide" and row["tool"] == "none"
