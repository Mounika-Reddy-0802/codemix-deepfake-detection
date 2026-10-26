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
