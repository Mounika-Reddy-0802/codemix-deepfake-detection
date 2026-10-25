"""Real-call test preparation: scripts, cloning packs and playback files (week 14)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import real_call_pack as rcp
from src.utils.audio_utils import amp_to_db, load_wav, rms, save_wav

ROOT = Path(__file__).resolve().parents[1]
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
ALL_LINES = (*rcp.REFERENCE, *rcp.SCRIPT_A, *rcp.SCRIPT_B)


def _speech(seconds: float, sr: int) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (0.1 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


def _padded(seconds: float, sr: int) -> np.ndarray:
    pad = np.zeros(sr // 2, dtype=np.float32)
    return np.concatenate([pad, _speech(seconds, sr), pad])


def test_scripts_have_twelve_lines_and_unique_keys() -> None:
    assert len(rcp.SCRIPT_A) == len(rcp.SCRIPT_B) == 12
    keys = [line.key for line in ALL_LINES]
    assert len(keys) == len(set(keys))


def test_every_line_is_code_mixed_and_romanises_within_the_xtts_limit() -> None:
    from src.data.transliteration import to_roman

    for line in ALL_LINES:
        roman = to_roman(line.devanagari)
        assert DEVANAGARI.search(line.devanagari), line.key
        assert re.search(r"[A-Za-z]", line.devanagari), line.key
        assert not DEVANAGARI.search(roman), line.key
        assert len(roman) <= rcp.XTTS_MAX_CHARS, line.key
        assert not DEVANAGARI.search(line.reading), line.key


def test_job_tables_carry_every_column_their_drivers_read() -> None:
    tables = rcp.job_tables()
    xtts = pd.read_csv(ROOT / "data/manifests/eval_ext_xtts_jobs.csv", nrows=0).columns
    chatterbox = pd.read_csv(
        ROOT / "data/manifests/chatterbox_generation_jobs.csv", nrows=0
    ).columns
    assert set(xtts) <= set(tables["xtts_v2"].columns)
    assert set(chatterbox) <= set(tables["chatterbox"].columns)
    for tool, table in tables.items():
        assert len(table) == len(rcp.SPEAKERS) * 24
        assert set(table["pool"]) == {rcp.POOL}
        assert set(table["tool"]) == {tool}
        assert table["output_path"].is_unique and table["seed"].is_unique


def test_xtts_table_loads_as_clone_jobs(tmp_path: Path) -> None:
    from src.data.pilot_jobs import load_pilot_jobs

    path = tmp_path / "generation_jobs.csv"
    rcp.job_tables()["xtts_v2"].to_csv(path, index=False)
    jobs = load_pilot_jobs(str(path), str(tmp_path), out_dir=str(tmp_path / "out"))
    assert len(jobs) == 72
    assert jobs[0].reference_wav.endswith("P1.wav") and jobs[0].pool == rcp.POOL


def test_build_pack_writes_trimmed_level_references_and_tables(tmp_path: Path) -> None:
    rec = tmp_path / "rec"
    rec.mkdir()
    for speaker in rcp.SPEAKERS:
        save_wav(str(rec / f"{speaker}_ref.wav"), _padded(8.0, 44_100), 44_100)
    summary = rcp.build_pack(rec, tmp_path / "pack")
    assert list(summary["speaker"]) == list(rcp.SPEAKERS)
    for tool, folder in rcp.PACK_DIRS.items():
        table = pd.read_csv(tmp_path / "pack" / folder / "generation_jobs.csv")
        assert set(table["tool"]) == {tool}
        audio, sr = load_wav(str(tmp_path / "pack" / folder / "refs" / "P1.wav"))
        assert sr == rcp.REFERENCE_SR
        assert abs(audio.size / sr - 8.0) < 0.1
        assert abs(amp_to_db(rms(audio)) - rcp.PLAYBACK_DBFS) < 0.5


def test_short_reference_is_refused(tmp_path: Path) -> None:
    save_wav(str(tmp_path / "P1_ref.wav"), _speech(3.0, 16_000), 16_000)
    with pytest.raises(ValueError, match="at least"):
        rcp.build_pack(tmp_path, tmp_path / "pack", speakers=("P1",))


def test_missing_or_duplicate_recording_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        rcp.find_recording(tmp_path, "P1_ref")
    (tmp_path / "P1_ref.wav").write_bytes(b"x")
    (tmp_path / "P1_ref.m4a").write_bytes(b"x")
    with pytest.raises(FileNotFoundError):
        rcp.find_recording(tmp_path, "P1_ref")


def test_assemble_calls_joins_clones_and_prepares_the_readings(tmp_path: Path) -> None:
    clones = tmp_path / "clones" / "rc_xtts"
    clones.mkdir(parents=True)
    rec = tmp_path / "rec"
    rec.mkdir()
    for line in rcp.SCRIPT_A[:-1]:  # the last sentence failed screening
        save_wav(str(clones / rcp.clip_name("xtts_v2", "P1", line)), _padded(2.0, 24_000), 24_000)
    for script in rcp.SCRIPTS:
        save_wav(str(rec / f"P1_read_{script}.wav"), _padded(5.0, 48_000), 48_000)

    table = rcp.assemble_calls(tmp_path / "clones", rec, tmp_path / "out", speakers=("P1",))
    row = table[(table["tool"] == "xtts_v2") & (table["script"] == "A")].iloc[0]
    assert row["clips"] == 11
    assert row["missing"] == rcp.clip_name("xtts_v2", "P1", rcp.SCRIPT_A[-1])
    audio, sr = load_wav(str(tmp_path / "out" / row["file"]))
    expected = 11 * 2.0 + 10 * rcp.GAP_SECONDS + 2 * rcp.LEAD_SECONDS
    assert sr == rcp.PLAYBACK_SR and abs(audio.size / sr - expected) < 0.2
    assert abs(amp_to_db(rms(audio)) - rcp.PLAYBACK_DBFS) < 0.5

    empty = table[(table["tool"] == "chatterbox") & (table["script"] == "A")].iloc[0]
    assert empty["clips"] == 0 and not (tmp_path / "out" / empty["file"]).exists()
    genuine = table[(table["kind"] == "genuine") & (table["script"] == "B")].iloc[0]
    audio, sr = load_wav(str(tmp_path / "out" / genuine["file"]))
    assert abs(audio.size / sr - (5.0 + 2 * rcp.LEAD_SECONDS)) < 0.1
    assert abs(amp_to_db(rms(audio)) - rcp.PLAYBACK_DBFS) < 0.5


def test_scripts_markdown_lists_every_line() -> None:
    text = rcp.scripts_markdown()
    for line in ALL_LINES:
        assert line.reading in text and line.devanagari in text
