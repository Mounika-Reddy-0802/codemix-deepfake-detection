"""Tests for speech-aware reference selection.

The failure this guards. Reference choice used to be "longest clip over 6 s",
which is a proxy for how much material XTTS gets to condition on. It is a bad
proxy: in the 4,000-clip Week-4 run, speaker 987461's reference was 25 s of which
89% was silence -- under 3 s of voice -- and that one speaker produced **95 of the
116 bad clips in the entire run**, stalling or truncating on 95 of its 160 jobs.
Nothing raised; the clips were written and counted as successes.

Ranking on speech content instead picks a 18 s clip with 12.3 s of voice over the
25 s clip with 2.65 s. These tests pin that preference.

Audio is synthesised in-memory, so nothing here needs a corpus on disk.
"""

import numpy as np
import pandas as pd
import pytest

from src.data import reference_quality as rq


def _tone(seconds: float, sr: int = 16000, amplitude: float = 0.2) -> np.ndarray:
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * 180 * t)).astype(np.float32)


def _with_silence(speech_s: float, silence_s: float, sr: int = 16000) -> np.ndarray:
    return np.concatenate([_tone(speech_s, sr), np.zeros(int(silence_s * sr), dtype=np.float32)])


# --------------------------------------------------------------------------- #
# silence_fraction
# --------------------------------------------------------------------------- #
def test_pure_speech_is_not_silent():
    assert rq.silence_fraction(_tone(3.0), 16000) < 0.1


def test_mostly_silence_is_detected():
    # 1 s of tone followed by 9 s of nothing -- 987461's shape.
    assert rq.silence_fraction(_with_silence(1.0, 9.0), 16000) > 0.8


def test_empty_audio_is_fully_silent():
    assert rq.silence_fraction(np.array([], dtype=np.float32), 16000) == 1.0


def test_a_quiet_speaker_is_not_judged_silent():
    # The threshold is relative to the clip's own peak, so a quietly recorded
    # speaker must not be mistaken for silence.
    quiet = _tone(3.0, amplitude=0.005)
    assert rq.silence_fraction(quiet, 16000) < 0.1


def test_a_clip_shorter_than_one_frame_is_not_silent_by_default():
    assert rq.silence_fraction(_tone(0.01), 16000) == 0.0


# --------------------------------------------------------------------------- #
# Ranking -- the actual fix
# --------------------------------------------------------------------------- #
def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"utt_id": "long_but_empty", "duration_seconds": 25.0, "speech": 2.6, "silence": 22.4},
            {"utt_id": "short_but_dense", "duration_seconds": 12.0, "speech": 11.0, "silence": 1.0},
        ]
    )


def _fake_probe(monkeypatch):
    def probe(row, target_sr=16000):
        return rq.ReferenceQuality(
            utt_id=str(row["utt_id"]),
            duration_seconds=float(row["duration_seconds"]),
            silence_fraction=float(row["silence"]) / float(row["duration_seconds"]),
            speech_seconds=float(row["speech"]),
        )

    monkeypatch.setattr(rq, "probe", probe)


def test_a_dense_short_clip_beats_a_long_silent_one(monkeypatch):
    # The whole point: duration-ranking picks the 25 s clip and loses 95 clips.
    _fake_probe(monkeypatch)
    assert rq.best_reference(_candidates()).utt_id == "short_but_dense"


def test_ranking_is_by_speech_not_duration(monkeypatch):
    _fake_probe(monkeypatch)
    ranked = rq.rank_candidates(_candidates())
    assert [q.utt_id for q in ranked] == ["short_but_dense", "long_but_empty"]


def test_an_empty_candidate_frame_yields_nothing():
    assert rq.best_reference(pd.DataFrame(columns=["utt_id", "duration_seconds"])) is None


def test_usable_tracks_the_speech_threshold():
    thin = rq.ReferenceQuality("a", 25.0, 0.89, 2.65)
    fat = rq.ReferenceQuality("b", 18.0, 0.31, 12.34)
    assert not thin.usable  # 987461 before the fix
    assert fat.usable  # 987461 after


@pytest.mark.parametrize("speech", [4.99, 5.0, 5.01])
def test_the_threshold_boundary(speech):
    assert rq.ReferenceQuality("x", 10.0, 0.5, speech).usable == (speech >= rq.MIN_SPEECH_SECONDS)
