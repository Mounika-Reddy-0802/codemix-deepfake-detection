"""Tests for the generated-clip quality screen (W4-T6).

The screen exists because of a real failure: one pilot job returned 0.83 s of
audio for a 150-character transcript and raised nothing, so it was written and
counted as a success. At 4,000 clips that class of failure is ~100 dead files
inside the training corpus. These tests pin the thresholds that catch it.

Pure numpy + the stdlib wave module -- no torch, no audio fixtures on disk.
"""

import wave

import numpy as np
import pandas as pd

from src.data import generation_qa as qa


def _write_wav(path, seconds=5.0, rate=16000, amplitude=0.2, clip=False):
    n = int(seconds * rate)
    t = np.linspace(0, seconds, n, endpoint=False)
    audio = amplitude * np.sin(2 * np.pi * 180 * t)
    if clip:
        audio = np.sign(audio) * 1.0
    data = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(data.tobytes())
    return path


def _jobs(**overrides):
    row = {
        "speaker": "trn0",
        "language": "hi",
        "transcript": "a" * 70,  # 70 chars -> 14 chars/s over 5 s, a normal rate
        "output_path": "outputs/clip.wav",
    }
    row.update(overrides)
    return pd.DataFrame([row])


# --------------------------------------------------------------------------- #
# The failure that motivated the module
# --------------------------------------------------------------------------- #
def test_a_truncated_clip_is_caught(tmp_path):
    # 150 chars in 0.83 s -- the real pilot failure, ~180 chars/sec.
    _write_wav(tmp_path / "clip.wav", seconds=0.83)
    report = qa.screen(_jobs(transcript="x" * 150), str(tmp_path))
    assert not report.loc[0, "ok"]
    assert "too short" in report.loc[0, "reason"] or "too fast" in report.loc[0, "reason"]


def test_a_normal_clip_passes(tmp_path):
    _write_wav(tmp_path / "clip.wav", seconds=5.0)
    report = qa.screen(_jobs(), str(tmp_path))
    assert report.loc[0, "ok"], report.loc[0, "reason"]


def test_a_stalled_clip_is_caught(tmp_path):
    # 30 chars stretched over 20 s -- the model looping rather than speaking.
    _write_wav(tmp_path / "clip.wav", seconds=20.0)
    report = qa.screen(_jobs(transcript="y" * 30), str(tmp_path))
    assert not report.loc[0, "ok"]
    assert "too slow" in report.loc[0, "reason"]


def test_near_silence_is_caught(tmp_path):
    _write_wav(tmp_path / "clip.wav", seconds=5.0, amplitude=0.0005)
    report = qa.screen(_jobs(), str(tmp_path))
    assert not report.loc[0, "ok"]
    assert "near-silent" in report.loc[0, "reason"]


def test_clipping_is_caught(tmp_path):
    _write_wav(tmp_path / "clip.wav", seconds=5.0, clip=True)
    report = qa.screen(_jobs(), str(tmp_path))
    assert not report.loc[0, "ok"]
    assert "clipped" in report.loc[0, "reason"]


def test_a_missing_file_is_a_failure_not_a_crash(tmp_path):
    report = qa.screen(_jobs(), str(tmp_path))
    assert not report.loc[0, "ok"]
    assert report.loc[0, "reason"] == "missing"


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def test_every_failure_names_itself(tmp_path):
    _write_wav(tmp_path / "clip.wav", seconds=0.4)
    report = qa.screen(_jobs(transcript="z" * 150), str(tmp_path))
    assert report.loc[0, "reason"].strip() != ""


def test_summary_counts_and_pass_rate(tmp_path):
    _write_wav(tmp_path / "good.wav", seconds=5.0)
    _write_wav(tmp_path / "bad.wav", seconds=0.3)
    jobs = pd.DataFrame(
        [
            {"speaker": "a", "language": "hi", "transcript": "a" * 70, "output_path": "o/good.wav"},
            {"speaker": "b", "language": "hi", "transcript": "b" * 150, "output_path": "o/bad.wav"},
        ]
    )
    stats = qa.summarise(qa.screen(jobs, str(tmp_path)))
    assert stats["clips"] == 2
    assert stats["passed"] == 1
    assert stats["failed"] == 1
    assert stats["pass_rate"] == 50.0


def test_summary_of_an_empty_report():
    assert qa.summarise(pd.DataFrame()) == {"clips": 0}


def test_thresholds_do_not_flag_the_measured_pilot_rates():
    # The pilot measured 13.9 and 14.4 chars/sec; natural variation must pass or
    # the screen would reject good audio.
    for rate in (10.0, 13.9, 14.4, 20.0):
        row = {
            "duration_sec": 10.0,
            "rms": 0.1,
            "clipped_fraction": 0.0,
            "transcript": "c" * int(rate * 10),
        }
        assert qa.reasons(row) == [], rate
