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


def test_generation_is_blocked_before_the_tool_check_when_unsigned() -> None:
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
