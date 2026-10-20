"""Tests for the week-13 generation entry point and the held-out pool rule."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data import generate_cli, heldout_tts


def test_shards_split_a_table_without_overlap():
    jobs = pd.DataFrame({"job_id": range(1, 8)})
    a, b = generate_cli.shard(jobs, 0, 2), generate_cli.shard(jobs, 1, 2)
    assert sorted(a["job_id"].tolist() + b["job_id"].tolist()) == list(range(1, 8))
    assert not set(a["job_id"]) & set(b["job_id"])
    with pytest.raises(ValueError):
        generate_cli.shard(jobs, 2, 2)


def test_heldout_jobs_accept_the_extension_eval_pool_only():
    refs = {"s1": "refs/s1.wav"}
    jobs = heldout_tts.build_heldout_jobs(refs, [("s1", "text")], "out", pool="eval_ext")
    assert jobs[0].pool == "eval_ext"
    with pytest.raises(ValueError):
        heldout_tts.build_heldout_jobs(refs, [("s1", "text")], "out", pool="adapt_ext")


def test_indicf5_audio_is_returned_as_float():
    np = pytest.importorskip("numpy")
    from src.data.indicf5_generation import to_float

    out = to_float(np.array([0, 16384, -32768], dtype=np.int16))
    assert out.dtype == np.float32 and out.max() == pytest.approx(0.5)
