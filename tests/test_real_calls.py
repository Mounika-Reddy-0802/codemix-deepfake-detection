"""The real phone-call test through the live verdict path (week 14)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from live_call.verdict_engine import EngineConfig
from src.data import real_call_pack as rcp
from src.inference import real_calls as rc
from src.utils.audio_utils import load_wav, save_wav
from src.utils.paths import resolve

ROOT = Path(__file__).resolve().parents[1]


def _call_windows(call_id: str, scores: list[float]) -> list[dict]:
    rows, ema = [], None
    for w, s in enumerate(scores):
        ema = s if ema is None else 0.5 * s + 0.5 * ema
        rows.append(
            {
                "filepath": f"${{DATA_ROOT}}/g711/{call_id}.wav",
                "label": "x",
                "tool": "x",
                "window": w,
                "end_seconds": 4.0 + 2 * w,
                "level_dbfs": -23.0,
                "score": s,
                "smoothed": ema,
            }
        )
    return rows


def test_call_plan_has_thirty_calls_of_three_kinds() -> None:
    plan = rc.CALL_PLAN
    assert len(plan) == 30 and plan["call_id"].is_unique
    assert plan.groupby(["kind", "tool"]).size().to_dict() == {
        ("clone_replay", "chatterbox"): 6,
        ("clone_replay", "xtts_v2"): 6,
        ("genuine_live", "none"): 12,
        ("genuine_replay", "none"): 6,
    }
    assert set(plan.loc[plan["kind"] == "clone_replay", "label"]) == {"spoof"}
    assert set(plan.loc[plan["kind"] != "clone_replay", "label"]) == {"bonafide"}


def test_plan_playback_names_match_the_files_the_pack_builder_writes() -> None:
    expected = {
        rcp.playback_name(kind, speaker, script)
        for speaker in rcp.SPEAKERS
        for script in rcp.SCRIPTS
        for kind in ("genuine", *rcp.TOOLS)
    }
    replay = rc.CALL_PLAN[rc.CALL_PLAN["playback"] != ""]
    assert set(replay["playback"]) == expected


def test_telephone_leg_keeps_length_and_removes_the_upper_band() -> None:
    pytest.importorskip("librosa")  # resampling wheel, not installed in CI
    sr = 16_000
    tone = (0.3 * np.sin(2 * np.pi * 6_000 * np.arange(sr) / sr)).astype(np.float32)
    out = rc.telephone_leg(tone, sr)
    assert out.size == tone.size
    assert np.sqrt(np.mean(out**2)) < 0.05 * np.sqrt(np.mean(tone**2))


def test_prepare_decodes_both_conditions_and_writes_manifests(tmp_path: Path) -> None:
    pytest.importorskip("soundfile")  # audio wheels, not installed in CI
    pytest.importorskip("librosa")
    rec = tmp_path / "rec"
    rec.mkdir()
    signal = (0.1 * np.sin(np.linspace(0, 2_000, 48_000 * 3))).astype(np.float32)
    for call_id in ("C01", "C19"):
        save_wav(str(rec / f"{call_id}.wav"), signal, 48_000)
    table = rc.prepare(rec, tmp_path / "out")
    assert len(table) == 4
    manifest = pd.read_csv(tmp_path / "out" / "manifest_g711.csv")
    assert list(manifest["call_id"]) == ["C01", "C19"]
    assert list(manifest["label"]) == ["bonafide", "spoof"]
    audio, sr = load_wav(resolve(manifest["filepath"][0], tmp_path / "out"))
    assert sr == 16_000 and abs(audio.size / sr - 3.0) < 0.01


def test_two_recordings_for_one_call_are_refused(tmp_path: Path) -> None:
    (tmp_path / "C01.m4a").write_bytes(b"x")
    (tmp_path / "C01.wav").write_bytes(b"x")
    with pytest.raises(ValueError, match="more than one"):
        rc.find_recordings(tmp_path)


def test_call_verdicts_catch_a_clone_and_pass_a_genuine_call() -> None:
    config = EngineConfig(threshold=0.5, suspicious_after=2, fake_after=4)
    windows = pd.DataFrame(
        _call_windows("C01", [0.99] * 20) + _call_windows("C19", [0.99, 0.99] + [0.01] * 10)
    )
    calls = rc.call_verdicts(windows, config).set_index("call_id")
    assert not calls.loc["C01", "caution"] and calls.loc["C01", "fake_leaning_pct"] == 0.0
    assert calls.loc["C19", "likely_cloned"]
    assert calls.loc["C19", "first_caution_s"] < calls.loc["C19", "first_likely_cloned_s"]


def test_summary_and_window_eer_per_tool() -> None:
    config = EngineConfig(threshold=0.5)
    windows = pd.DataFrame(
        _call_windows("C01", [0.99] * 10)  # P1, genuine live
        + _call_windows("C13", [0.9] * 10)  # P1, genuine replay
        + _call_windows("C19", [0.01] * 10)  # P1, XTTS clone
        + _call_windows("C25", [0.95] * 10)  # P1, Chatterbox clone that passes
    )
    calls = rc.CALL_PLAN.merge(rc.call_verdicts(windows, config), on="call_id")
    summary = {(r["kind"], r["tool"]): r for r in rc.summarise(calls)}
    assert summary[("clone_replay", "xtts_v2")]["caution_pct"] == 100.0
    assert summary[("clone_replay", "chatterbox")]["caution_pct"] == 0.0
    assert summary[("genuine_live", "none")]["caution_pct"] == 0.0
    xtts = rc.window_eer(windows, calls, "genuine_replay", "xtts_v2")
    assert xtts["eer_pct"] == 0.0 and xtts["windows"] == [10, 10]
    assert rc.window_eer(windows, calls, "genuine_replay", "chatterbox")["eer_pct"] > 50


def test_report_writes_per_call_verdicts_and_markdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(ROOT)  # the frozen operating points are read from configs/
    audio = tmp_path / "audio"
    audio.mkdir()
    pd.DataFrame(
        {
            "filepath": ["${DATA_ROOT}/g711/C01.wav", "${DATA_ROOT}/g711/C19.wav"],
            "label": ["bonafide", "spoof"],
            "tool": ["none", "xtts_v2"],
            "source": "realcall",
            "call_id": ["C01", "C19"],
            "condition": "g711",
            "seconds": [60.0, 61.0],
        }
    ).to_csv(audio / "manifest_g711.csv", index=False)
    windows = tmp_path / "windows"
    windows.mkdir()
    pd.DataFrame(_call_windows("C01", [0.99] * 10) + _call_windows("C19", [0.01] * 10)).to_csv(
        windows / "windows_final_g711.csv", index=False
    )
    result = rc.report(audio, windows)
    assert result["calls_recorded"] == 2 and list(result["models"]) == ["final_g711"]
    calls = pd.read_csv(windows / "calls_final_g711.csv").set_index("call_id")
    assert calls.loc["C19", "likely_cloned"] and not calls.loc["C01", "caution"]
    assert calls.loc["C19", "seconds"] == 61.0
    text = rc.report_markdown(result)
    assert "final_g711" in text and "clone_replay" in text
    assert "by time in the call" in text and "after 20 s" in text


def test_score_by_phase_splits_each_kind_of_call_by_time() -> None:
    # windows end at 4, 6, ..., 26 s: four in 0-10 s, five in 10-20 s, three after 20 s
    genuine = [0.2] * 4 + [0.8] * 5 + [1.0] * 3
    windows = pd.DataFrame(_call_windows("C01", genuine) + _call_windows("C19", [0.1] * 12))
    calls = rc.CALL_PLAN.merge(rc.call_verdicts(windows, EngineConfig(threshold=0.5)), on="call_id")
    phases = {(r["kind"], r["tool"]): r for r in rc.score_by_phase(windows, calls)}
    assert phases[("genuine_live", "none")] == {
        "kind": "genuine_live",
        "tool": "none",
        "0-10 s": 0.2,
        "10-20 s": 0.8,
        "after 20 s": 1.0,
    }
    assert phases[("clone_replay", "xtts_v2")]["after 20 s"] == 0.1


def test_wilson_interval_is_sensible_for_small_counts() -> None:
    assert rc.wilson_interval(0, 0) == [0.0, 100.0]
    low, high = rc.wilson_interval(0, 4)
    assert low == 0.0 and 40 < high < 60  # none of four still allows about half
    low, high = rc.wilson_interval(4, 4)
    assert 40 < low < 60 and high == 100.0
    low, high = rc.wilson_interval(3, 4)
    assert low < 75 < high
    narrow = rc.wilson_interval(30, 40)
    assert narrow[1] - narrow[0] < high - low  # more calls, narrower interval


def test_summary_carries_an_interval_for_each_share() -> None:
    config = EngineConfig(threshold=0.5)
    windows = pd.DataFrame(_call_windows("C19", [0.01] * 10) + _call_windows("C21", [0.99] * 10))
    calls = rc.CALL_PLAN.merge(rc.call_verdicts(windows, config), on="call_id")
    row = next(r for r in rc.summarise(calls) if r["tool"] == "xtts_v2")
    assert row["calls"] == 2 and row["likely_cloned_pct"] == 50.0
    assert row["likely_cloned_ci"][0] < 50.0 < row["likely_cloned_ci"][1]


def _speechlike(seed: int, seconds: float = 30.0, sr: int = 16_000) -> np.ndarray:
    """Noise bursts with speech-like timing: 0.4-2 s of sound, 0.2-1 s pauses."""
    rng = np.random.default_rng(seed)
    audio = 0.001 * rng.standard_normal(int(seconds * sr))
    t = 0.5
    while t < seconds - 1:
        burst = rng.uniform(0.4, 2.0)
        a, b = int(t * sr), int(min(t + burst, seconds) * sr)
        audio[a:b] += 0.2 * rng.standard_normal(b - a)
        t += burst + rng.uniform(0.2, 1.0)
    return audio.astype(np.float32)


def _as_recorded(audio: np.ndarray, delay_s: float, sr: int = 16_000) -> np.ndarray:
    """A replay as a phone records it: later start, other gain, its own noise floor."""
    rng = np.random.default_rng(99)
    shifted = np.concatenate([np.zeros(int(delay_s * sr), np.float32), 0.3 * audio])
    return (shifted + 0.002 * rng.standard_normal(shifted.size)).astype(np.float32)


def test_speech_activity_finds_the_played_file_despite_delay_and_gain() -> None:
    played, other = _speechlike(1), _speechlike(2)
    recorded = rc.speech_activity(_as_recorded(played, delay_s=1.3))
    assert rc.overlap_correlation(recorded, rc.speech_activity(played)) > 0.8
    assert rc.overlap_correlation(recorded, rc.speech_activity(other)) < 0.5
    assert rc.overlap_correlation(np.ones(5), np.ones(5)) == -1.0  # flat tracks never match


def test_check_replays_flags_a_swapped_file(tmp_path: Path) -> None:
    pytest.importorskip("soundfile")  # audio wheels, not installed in CI
    pytest.importorskip("librosa")
    playback, recordings = tmp_path / "playback", tmp_path / "recordings"
    playback.mkdir()
    recordings.mkdir()
    a, b = _speechlike(3), _speechlike(4)
    save_wav(str(playback / "xtts_v2_P2_A.wav"), a, 16_000)
    save_wav(str(playback / "xtts_v2_P2_B.wav"), b, 16_000)
    save_wav(str(recordings / "C21.wav"), _as_recorded(a, 0.8), 16_000)  # as planned
    save_wav(str(recordings / "C22.wav"), _as_recorded(a, 1.1), 16_000)  # B planned, A played
    table = rc.check_replays(recordings, playback).set_index("call_id")
    assert list(table.index) == ["C21", "C22"]
    assert bool(table.loc["C21", "ok"]) and not bool(table.loc["C22", "ok"])
    assert table.loc["C22", "best"] == "xtts_v2_P2_A.wav"


def test_skipping_the_first_window_resmooths_and_can_clear_an_early_caution() -> None:
    config = EngineConfig(threshold=0.687, suspicious_after=3, fake_after=5)
    # a quiet first window scored as fake, then speech that settles high
    early = pd.DataFrame(_call_windows("C05", [0.0, 0.5, 0.6, 1.0, 1.0, 1.0]))
    clone = pd.DataFrame(_call_windows("C21", [0.0] + [0.05] * 6))
    windows = pd.concat([early, clone], ignore_index=True)
    skipped = rc.without_first_window(windows)
    first = skipped[skipped["window"] == 0]
    assert first["score"].isna().all() and first["smoothed"].isna().all()
    c05 = skipped[skipped["filepath"].str.contains("C05")].set_index("window")
    assert c05.loc[1, "smoothed"] == pytest.approx(0.5)  # the average restarts at window 1
    assert c05.loc[2, "smoothed"] == pytest.approx(0.55)
    before = rc.call_verdicts(windows, config).set_index("call_id")
    after = rc.call_verdicts(skipped, config).set_index("call_id")
    assert bool(before.loc["C05", "caution"]) and not bool(after.loc["C05", "caution"])
    assert bool(after.loc["C21", "likely_cloned"])  # a clone is still caught without it
    assert windows["score"].notna().all()  # the input is left alone
