"""Tests for the held-out Tortoise driver + firewall (Week 3, owner SK).

Logic-only (no tortoise/torchaudio needed). Confirms every held-out job is tagged
tortoise + eval, and that generation refuses anything else.
"""

import pytest

from src.data import heldout_tts as ht
from src.data.spoof_generation import CloneJob


def test_build_heldout_jobs_are_all_tortoise_eval() -> None:
    refs = {"spk_a": "a.wav", "spk_b": "b.wav"}
    transcripts = [("spk_a", "hello"), ("spk_b", "namaste"), ("spk_c", "skipped")]
    jobs = ht.build_heldout_jobs(refs, transcripts, out_dir="out")
    assert len(jobs) == 2  # spk_c has no reference
    assert all(j.tool == "tortoise" for j in jobs)
    assert all(j.pool == "eval" for j in jobs)


def test_build_heldout_respects_target() -> None:
    refs = {"s": "s.wav"}
    transcripts = [("s", f"t{i}") for i in range(50)]
    assert len(ht.build_heldout_jobs(refs, transcripts, "o", n_target=5)) == 5


def test_generate_refuses_non_heldout_tool() -> None:
    bad = CloneJob(
        speaker="s",
        reference_wav="r.wav",
        transcript="t",
        output_path="o.wav",
        pool="eval",
        tool="xtts_v2",
    )
    with pytest.raises(AssertionError):
        ht.generate_heldout_clone(model=None, job=bad)


def test_generate_refuses_non_eval_pool() -> None:
    bad = CloneJob(
        speaker="s",
        reference_wav="r.wav",
        transcript="t",
        output_path="o.wav",
        pool="adaptation",
        tool="tortoise",
    )
    with pytest.raises(AssertionError):
        ht.generate_heldout_batch([bad], model=None, metadata_path="x.jsonl")
