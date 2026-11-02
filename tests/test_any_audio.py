"""Tests for src.inference.any_audio with a stand-in score function (no torch)."""

import numpy as np
import pytest

from src.inference import any_audio as aa

SR = aa.SAMPLE_RATE


def tone(seconds: float, dbfs: float = -20.0, freq: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    amp = 10 ** (dbfs / 20) * np.sqrt(2)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def constant(p: float):
    seen = []

    def score(window: np.ndarray) -> float:
        seen.append(window)
        return p

    score.seen = seen
    return score


def test_window_starts_cover_the_tail():
    assert aa.window_starts(3 * SR) == [0]
    assert aa.window_starts(4 * SR) == [0]
    assert aa.window_starts(8 * SR) == [0, 2 * SR, 4 * SR]
    # 9 s: windows at 0, 2, 4 s end at 8 s, so one more ends at the last sample
    assert aa.window_starts(9 * SR) == [0, 2 * SR, 4 * SR, 5 * SR]


@pytest.mark.parametrize("p,expected", [(0.9, aa.REAL), (0.1, aa.CLONED)])
def test_verdict_follows_the_mean_against_the_threshold(p, expected):
    result = aa.judge(tone(9), constant(p), threshold=0.58, front="none")
    assert result.verdict == expected
    assert result.p_genuine == pytest.approx(p)
    assert result.windows_agreeing == len(result.scored) == 4


def test_mean_not_minimum_decides():
    scores = iter([0.95, 0.95, 0.2, 0.95])
    result = aa.judge(tone(9), lambda w: next(scores), threshold=0.58, front="none")
    assert result.verdict == aa.REAL
    assert result.windows_agreeing == 3


def test_windows_are_normalised_to_training_level():
    score = constant(0.9)
    aa.judge(tone(6, dbfs=-35), score, threshold=0.5, front="none")
    for window in score.seen:
        assert aa.rms_dbfs(window) == pytest.approx(-23.0, abs=0.1)


def test_silence_and_too_little_speech_are_not_judged():
    silent = np.zeros(6 * SR, dtype=np.float32)
    assert aa.judge(silent, constant(0.9), 0.5, front="none").verdict == aa.NO_SPEECH
    short = np.concatenate([np.zeros(2 * SR), tone(1.5), np.zeros(2 * SR)]).astype(np.float32)
    result = aa.judge(short, constant(0.9), 0.5, front="none")
    assert result.verdict == aa.NO_SPEECH
    assert result.speech_seconds == pytest.approx(1.5, abs=0.05)


def test_silent_ends_are_trimmed_before_windowing():
    audio = np.concatenate([np.zeros(5 * SR), tone(4), np.zeros(5 * SR)]).astype(np.float32)
    result = aa.judge(audio, constant(0.9), 0.5, front="none")
    assert result.seconds == pytest.approx(14.0)
    assert result.speech_seconds == pytest.approx(4.0, abs=0.05)
    assert len(result.windows) == 1


def test_g711_front_end_keeps_length_and_removes_the_upper_band():
    pytest.importorskip("librosa")  # resampling wheel, not installed in CI
    t = np.arange(4 * SR) / SR
    wide = (0.1 * np.sin(2 * np.pi * 300 * t) + 0.1 * np.sin(2 * np.pi * 6000 * t)).astype(
        np.float32
    )
    narrow = aa.front_end(wide, "g711")
    assert narrow.size == wide.size
    spectrum = np.abs(np.fft.rfft(narrow)) ** 2
    freqs = np.fft.rfftfreq(narrow.size, 1 / SR)
    assert spectrum[freqs > 4100].sum() / spectrum.sum() < 0.01
    with pytest.raises(ValueError):
        aa.front_end(wide, "amr")


def test_summary_is_flat_and_complete():
    result = aa.judge(tone(6), constant(0.3), 0.58, front="none", source="x.wav")
    summary = result.summary()
    assert summary["verdict"] == aa.CLONED and summary["source"] == "x.wav"
    assert summary["windows_scored"] == 2 and "windows" not in summary


def test_config_file_names_a_known_detector():
    config = aa.DetectorConfig.from_file()
    assert config.kind in ("ssl_aasist", "adapter")
    assert config.front_end in aa.FRONT_ENDS
    assert 0.0 < config.threshold() < 1.0
