"""The Streamlit demo's pure logic, tested without a Streamlit runtime.

These cover the parts that decide what an evaluator sees: which checkpoint is
offered, what counts as a clone, how a clip is prepared, and that the windowing
matches the live path. The Streamlit page itself is not imported here.
"""

from __future__ import annotations

import numpy as np
import pytest

from app import app_core as core


def _voice(seconds: float = 10.0, sr: int = core.SAMPLE_RATE) -> np.ndarray:
    """A harmonic tone with a syllable-rate envelope: a stand-in for speech."""
    t = np.arange(int(seconds * sr)) / sr
    tone = sum(np.sin(2 * np.pi * 120 * k * t) / k for k in range(1, 8))
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 4 * t))
    return (0.1 * tone * envelope).astype(np.float32)


# ------------------------------------------------------------------- models


def test_the_deployed_adapter_is_offered_first_when_configured():
    choices = core.model_choices(env_checkpoint="/somewhere/lora_norm_rvc_channel_best.pt")
    assert choices[0].key == "deployed"
    assert [c.key for c in choices[1:]] == ["s2_channel", "s2_clean", "s1"]


def test_without_the_env_checkpoint_the_channel_adapter_leads():
    assert core.model_choices(env_checkpoint=None)[0].key == "s2_channel"


def test_checkpoint_paths_are_resolved_from_the_repo_not_the_cwd():
    choice = core.choice_by_key("s1")
    assert choice.path.is_absolute()
    assert choice.path.parts[-3:] == ("checkpoints", "baseline", "best.pt")


def test_a_missing_checkpoint_explains_itself():
    choice = core.choice_by_key("s2_clean")
    message = core.missing_checkpoint_message(choice)
    assert "git lfs pull" in message
    assert str(choice.path) in message


def test_unknown_model_key_is_refused():
    with pytest.raises(KeyError):
        core.choice_by_key("no-such-model")


# -------------------------------------------------------------------- audio


def test_prepare_resamples_to_16k_and_normalises_the_level():
    pytest.importorskip("librosa")  # resampling wheel, not installed in CI
    from src.utils.audio_utils import amp_to_db, rms

    audio = _voice(2.0, sr=44_100) * 0.01  # quiet, and at the wrong rate
    out = core.prepare(audio, 44_100)
    assert out.dtype == np.float32
    assert abs(out.size - 2 * core.SAMPLE_RATE) <= 2
    assert amp_to_db(rms(out)) == pytest.approx(-23.0, abs=0.5)


def test_prepare_leaves_a_16k_clip_at_its_own_length():
    audio = _voice(3.0)
    assert core.prepare(audio, core.SAMPLE_RATE).size == audio.size


def test_the_phone_line_removes_energy_above_4_khz():
    """A narrowband line cannot carry 6 kHz. The tone must come back gutted."""
    pytest.importorskip("librosa")  # resampling wheel, not installed in CI
    t = np.arange(2 * core.SAMPLE_RATE) / core.SAMPLE_RATE
    wideband = (0.3 * np.sin(2 * np.pi * 300 * t) + 0.3 * np.sin(2 * np.pi * 6000 * t)).astype(
        np.float32
    )

    def energy_at(x: np.ndarray, hz: float, width: float = 200.0) -> float:
        spectrum = np.abs(np.fft.rfft(x)) ** 2
        freqs = np.fft.rfftfreq(x.size, 1 / core.SAMPLE_RATE)
        band = (freqs > hz - width) & (freqs < hz + width)
        return float(spectrum[band].sum() / spectrum.sum())

    carried = core.apply_phone_line(wideband, snr_db=30.0)
    assert energy_at(wideband, 6000) > 0.2, "the test signal must start with 6 kHz energy"
    assert energy_at(carried, 6000) < 0.01, "6 kHz must not survive an 8 kHz line"
    assert energy_at(carried, 300) > 0.5, "the speech band must survive"
    assert carried.size == wideband.size
    assert np.max(np.abs(carried)) <= 1.0


# ------------------------------------------------------------------ scoring


def test_scoring_windows_a_long_clip_every_two_seconds():
    audio = core.prepare(_voice(10.0), core.SAMPLE_RATE)
    result = core.score_audio(audio, lambda w: 0.9, threshold=0.5)
    assert [round(w.end_seconds, 1) for w in result.windows] == [4.0, 6.0, 8.0, 10.0]
    assert result.verdict == "bonafide"
    assert not result.is_clone


def test_a_clip_shorter_than_one_window_still_yields_a_verdict():
    audio = core.prepare(_voice(1.5), core.SAMPLE_RATE)
    result = core.score_audio(audio, lambda w: 0.1, threshold=0.5)
    assert len(result.windows) == 1
    assert result.is_clone


def test_the_clip_score_is_the_worst_window_not_the_average():
    audio = core.prepare(_voice(10.0), core.SAMPLE_RATE)
    scores = iter([0.99, 0.98, 0.02, 0.97])
    result = core.score_audio(audio, lambda w: next(scores), threshold=0.5)
    assert result.score == pytest.approx(0.02)
    assert result.is_clone, "one confidently fake window must not be averaged away"


def test_silence_is_gated_out_and_reported_as_no_speech():
    silence = np.zeros(10 * core.SAMPLE_RATE, dtype=np.float32)
    result = core.score_audio(silence, lambda w: 0.99, threshold=0.5)
    assert result.score is None
    assert result.verdict == "no speech"
    assert result.scored_windows == []


def test_the_verdict_card_states_the_score_and_the_threshold():
    audio = core.prepare(_voice(5.0), core.SAMPLE_RATE)
    result = core.score_audio(audio, lambda w: 0.02, threshold=0.5)
    headline, colour, detail = core.verdict_card(result)
    assert headline == "CLONED VOICE"
    assert colour == "red"
    assert "0.020" in detail and "0.500" in detail


def test_the_card_says_no_speech_rather_than_guessing():
    result = core.Result(None, "no speech", [], 3.0, 0.5)
    headline, colour, _ = core.verdict_card(result)
    assert headline == "NO SPEECH"
    assert colour == "gray"


# --------------------------------------------------------------- demo clips


def test_demo_clips_lists_real_ones_first_and_labels_the_truth(tmp_path):
    for name in ["xtts_clone.wav", "genuine.wav", "notes.txt", "mystery.wav"]:
        (tmp_path / name).write_bytes(b"")
    clips = core.demo_clips(tmp_path)
    assert [c.path.stem for c in clips] == ["genuine", "xtts_clone", "mystery"]
    assert clips[0].truth == "real" and not clips[0].is_clone
    assert clips[1].truth == "cloned" and clips[1].is_clone
    assert clips[2].truth == "unknown", "an unrecognised clip is listed, not silently dropped"


def test_demo_clips_is_empty_when_the_folder_is_absent(tmp_path):
    assert core.demo_clips(tmp_path / "nothing-here") == []
