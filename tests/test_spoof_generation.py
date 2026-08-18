"""Tests for speaker selection + generation metadata (Week 2, owner SK).

Logic-only: no XTTS/torch needed (those are lazy). Actual audio generation is a
data + GPU + human-listening step, not unit-tested here.
"""

import pandas as pd
import pytest

from src.data import spoof_generation as sg


def _clips() -> pd.DataFrame:
    rows = []
    # two speakers with plenty of audio, one too short, one child (must be excluded)
    rows += [{"speaker": "spk_a", "source": "mucs2021", "duration": 4.0} for _ in range(12)]
    rows += [{"speaker": "spk_b", "source": "hiacc", "duration": 4.0} for _ in range(10)]
    rows.append({"speaker": "spk_short", "source": "mucs2021", "duration": 5.0})
    rows += [{"speaker": "child_03", "source": "hiacc_child", "duration": 4.0} for _ in range(20)]
    return pd.DataFrame(rows)


def test_selection_excludes_child_and_short_speakers() -> None:
    selected = sg.select_reference_speakers(
        _clips(), min_total_seconds=30.0, min_clip_seconds=3.0, n_min=1, n_max=50
    )
    assert "spk_a" in selected
    assert "spk_b" in selected
    assert "spk_short" not in selected  # only 5 s total < 30 s
    assert all("child" not in s for s in selected)


def test_selection_respects_n_max() -> None:
    df = pd.DataFrame(
        [{"speaker": f"spk{i}", "source": "mucs2021", "duration": 40.0} for i in range(80)]
    )
    assert len(sg.select_reference_speakers(df, n_max=50)) == 50


def test_enough_speakers() -> None:
    assert sg.enough_speakers(["a"] * 30, n_min=30)
    assert not sg.enough_speakers(["a"] * 10, n_min=30)


def test_metadata_roundtrip(tmp_path) -> None:
    path = tmp_path / "meta.jsonl"
    rec = sg.GenerationRecord(
        output_path="out/x.wav",
        tool="xtts_v2",
        speaker="spk_a",
        reference_wav="ref.wav",
        transcript="hello dost",
        language="hi",
        pool="adaptation",
        seed=1,
    )
    sg.append_metadata(rec, str(path))
    sg.append_metadata(rec, str(path))
    records = sg.read_metadata(str(path))
    assert len(records) == 2
    assert records[0]["tool"] == "xtts_v2"
    assert records[0]["speaker"] == "spk_a"


def test_read_metadata_missing_file_is_empty() -> None:
    assert sg.read_metadata("does/not/exist.jsonl") == []


def test_heldout_tool_must_be_eval_pool(open_ethics_gate) -> None:
    # A Tortoise (held-out) clone tagged for the adaptation pool must be rejected
    # before any generation happens. The ethics gate is the outer check and fires
    # first in real use; it is opened here so the tool firewall is what is tested.
    job = sg.CloneJob(
        speaker="spk_a",
        reference_wav="ref.wav",
        transcript="text",
        output_path="out/y.wav",
        pool="adaptation",
        tool=sg.HELD_OUT_TOOL,
    )
    with pytest.raises(AssertionError):
        sg.generate_batch([job], model=None, metadata_path="unused.jsonl")


def test_generation_is_blocked_before_the_tool_check_when_unsigned(closed_ethics_gate) -> None:
    # With the gate closed (the repo's real state) nothing reaches the firewall.
    from src.data.ethics_gate import EthicsGateError

    job = sg.CloneJob(
        speaker="spk_a",
        reference_wav="ref.wav",
        transcript="text",
        output_path="out/y.wav",
        pool="train",
        tool=sg.TRAINING_TOOL,
    )
    with pytest.raises(EthicsGateError):
        sg.generate_batch([job], model=None, metadata_path="unused.jsonl")


# --------------------------------------------------------------------------- #
# Metadata reconciliation
# --------------------------------------------------------------------------- #
# The log is append-only, so regenerating a clip (delete the wav, run again)
# leaves the superseded record beside the new one. The real pilot ended up with
# 25 records for 20 files, five describing transcripts no longer on disk. This
# file is the provenance the datasheet is built from, so a stale record is worse
# than a missing one -- it misreports what the corpus contains.
def _record(path: str, transcript: str) -> dict:
    return {
        "output_path": path,
        "tool": "xtts_v2",
        "speaker": "trn0",
        "reference_wav": "refs/trn0.wav",
        "transcript": transcript,
        "language": "hi",
        "pool": "train",
        "seed": 1,
        "settings": {},
    }


def test_last_record_wins_for_a_regenerated_clip(tmp_path) -> None:
    clip = tmp_path / "a.wav"
    clip.write_bytes(b"RIFF")
    records = [_record(str(clip), "old 250-char text"), _record(str(clip), "new 150-char text")]
    kept = sg.reconcile_metadata(records)
    assert len(kept) == 1
    assert kept[0]["transcript"] == "new 150-char text"


def test_records_without_a_file_are_dropped(tmp_path) -> None:
    present = tmp_path / "present.wav"
    present.write_bytes(b"RIFF")
    records = [_record(str(present), "kept"), _record(str(tmp_path / "gone.wav"), "deleted")]
    kept = sg.reconcile_metadata(records)
    assert [r["transcript"] for r in kept] == ["kept"]


def test_require_file_false_keeps_orphans(tmp_path) -> None:
    records = [_record(str(tmp_path / "gone.wav"), "deleted")]
    assert len(sg.reconcile_metadata(records, require_file=False)) == 1


def test_rewrite_metadata_makes_the_log_match_disk(tmp_path) -> None:
    clip = tmp_path / "a.wav"
    clip.write_bytes(b"RIFF")
    log = tmp_path / "metadata.jsonl"
    for record in (
        _record(str(clip), "old"),
        _record(str(clip), "new"),
        _record(str(tmp_path / "gone.wav"), "orphan"),
    ):
        sg.append_metadata(sg.GenerationRecord(**record), str(log))

    assert len(sg.read_metadata(str(log))) == 3
    assert sg.rewrite_metadata(str(log)) == 1
    remaining = sg.read_metadata(str(log))
    assert [r["transcript"] for r in remaining] == ["new"]


def test_stats_over_a_reconciled_log_count_each_file_once(tmp_path) -> None:
    clip = tmp_path / "a.wav"
    clip.write_bytes(b"RIFF")
    records = [_record(str(clip), "old"), _record(str(clip), "new")]
    assert sg.generation_stats(records)["total"] == 2  # raw log double-counts
    assert sg.generation_stats(sg.reconcile_metadata(records))["total"] == 1
