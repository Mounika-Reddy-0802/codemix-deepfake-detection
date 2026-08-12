"""Tests for the SNR/duration speaker ranking (Week 3, W3-T4, owner L).

The ranking decides which voices get cloned, so two properties are pinned: a
noisy speaker must not outrank a clean one just by talking longer, and a speaker
that fails the thresholds must never be padded into the shortlist to hit the
target count.

Pure numpy/pandas, no audio files, so this runs in CI.
"""

import numpy as np
import pandas as pd
import pytest

from src.data import speaker_selection as ss


def _tone(seconds: float, sr: int = 16_000, amp: float = 0.3) -> np.ndarray:
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _speech_like(seconds: float, sr: int = 16_000, noise: float = 0.0) -> np.ndarray:
    """Alternating loud/quiet blocks, so there is a speech level and a floor."""
    rng = np.random.default_rng(0)
    signal = _tone(seconds, sr)
    block = sr // 4
    for start in range(0, signal.size, 2 * block):
        signal[start : start + block] *= 0.01  # a pause
    if noise:
        signal = signal + rng.normal(0, noise, signal.shape).astype(np.float32)
    return signal


def _clips(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "filepath": f"{spk}/clip{i}.wav",
                "speaker": spk,
                "source": "mucs2021",
                "duration_seconds": dur,
                "snr_db": snr,
            }
            for i, (spk, dur, snr) in enumerate(rows)
        ]
    )


# --------------------------------------------------------------------------- #
# SNR estimate
# --------------------------------------------------------------------------- #
def test_silence_scores_zero() -> None:
    assert ss.estimate_snr_db(np.zeros(16_000, dtype=np.float32)) == 0.0


def test_too_short_input_scores_zero() -> None:
    assert ss.estimate_snr_db(np.ones(10, dtype=np.float32)) == 0.0


def test_snr_is_never_negative() -> None:
    rng = np.random.default_rng(1)
    assert ss.estimate_snr_db(rng.normal(0, 0.1, 16_000).astype(np.float32)) >= 0.0


def test_clean_speech_scores_above_noisy_speech() -> None:
    clean = ss.estimate_snr_db(_speech_like(2.0))
    noisy = ss.estimate_snr_db(_speech_like(2.0, noise=0.05))
    assert clean > noisy


def test_added_noise_lowers_the_estimate_monotonically() -> None:
    scores = [ss.estimate_snr_db(_speech_like(2.0, noise=n)) for n in (0.0, 0.02, 0.08)]
    assert scores[0] > scores[1] > scores[2]


def test_frame_energies_are_empty_for_short_input() -> None:
    assert ss.frame_energies_db(np.ones(5, dtype=np.float32)).size == 0


# --------------------------------------------------------------------------- #
# Speaker id from path
# --------------------------------------------------------------------------- #
def test_speaker_is_the_parent_directory() -> None:
    assert ss.speaker_from_path("data/processed/clean/mucs2021/spk042/utt_seg001.wav") == "spk042"


def test_windows_separators_are_handled() -> None:
    assert ss.speaker_from_path(r"data\processed\mucs2021\spk007\a.wav") == "spk007"


def test_flat_tree_falls_back_to_the_stem() -> None:
    assert ss.speaker_from_path("utt_0001.wav") == "utt_0001"


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #
def test_empty_index_gives_an_empty_ranking() -> None:
    assert ss.rank_speakers(pd.DataFrame(columns=ss.CLIP_COLUMNS)).empty


def test_short_clips_are_dropped_before_aggregating() -> None:
    clips = _clips([("spk1", 1.0, 20.0), ("spk1", 1.5, 20.0), ("spk2", 10.0, 20.0)])
    ranked = ss.rank_speakers(clips)
    assert set(ranked["speaker"]) == {"spk2"}


def test_totals_and_medians_are_aggregated_per_speaker() -> None:
    clips = _clips([("spk1", 10.0, 12.0), ("spk1", 30.0, 18.0)])
    row = ss.rank_speakers(clips).iloc[0]
    assert row["n_clips"] == 2
    assert row["total_seconds"] == pytest.approx(40.0)
    assert row["median_snr_db"] == pytest.approx(15.0)


def test_clean_speaker_outranks_a_longer_noisy_one() -> None:
    # The whole point of ranking on SNR first: hours of noisy audio are worse
    # cloning references than half a minute of clean audio.
    clips = _clips([("noisy", 600.0, 11.0), ("clean", 40.0, 25.0)])
    assert ss.rank_speakers(clips).iloc[0]["speaker"] == "clean"


def test_eligibility_needs_both_duration_and_snr() -> None:
    clips = _clips([("short", 10.0, 30.0), ("noisy", 120.0, 5.0), ("good", 120.0, 30.0)])
    ranked = ss.rank_speakers(clips).set_index("speaker")
    assert ranked.loc["good", "eligible"]
    assert not ranked.loc["short", "eligible"]
    assert not ranked.loc["noisy", "eligible"]


# --------------------------------------------------------------------------- #
# Shortlist
# --------------------------------------------------------------------------- #
def test_shortlist_contains_only_eligible_speakers() -> None:
    clips = _clips([(f"spk{i}", 120.0, 30.0) for i in range(5)] + [("bad", 5.0, 2.0)])
    assert "bad" not in set(ss.shortlist(ss.rank_speakers(clips))["speaker"])


def test_shortlist_is_capped_at_n_max() -> None:
    clips = _clips([(f"spk{i}", 120.0, 30.0) for i in range(60)])
    cfg = ss.SelectionConfig(n_max=50)
    assert len(ss.shortlist(ss.rank_speakers(clips, cfg), cfg)) == 50


def test_shortlist_is_not_padded_with_ineligible_speakers() -> None:
    # Coming up short must be visible, not hidden by topping up the list.
    clips = _clips([(f"spk{i}", 120.0, 30.0) for i in range(4)] + [("bad", 5.0, 1.0)])
    picked = ss.shortlist(ss.rank_speakers(clips))
    assert len(picked) == 4
    assert not ss.enough_speakers(picked)


def test_enough_speakers_at_the_plan_minimum() -> None:
    clips = _clips([(f"spk{i}", 120.0, 30.0) for i in range(30)])
    assert ss.enough_speakers(ss.shortlist(ss.rank_speakers(clips)))


def test_summary_reports_the_numbers_the_doc_needs() -> None:
    clips = _clips([(f"spk{i}", 120.0, 30.0) for i in range(35)])
    summary = ss.selection_summary(ss.rank_speakers(clips))
    assert summary["speakers_indexed"] == 35
    assert summary["shortlisted"] == 35
    assert summary["enough"] is True
    assert summary["total_hours"] == pytest.approx(35 * 120 / 3600, abs=1e-3)


def test_summary_on_an_empty_ranking_does_not_crash() -> None:
    summary = ss.selection_summary(ss.rank_speakers(pd.DataFrame(columns=ss.CLIP_COLUMNS)))
    assert summary["shortlisted"] == 0
    assert summary["enough"] is False
