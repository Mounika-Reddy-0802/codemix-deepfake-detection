"""Tests for src/data/indicvoices_jobs.py on a synthetic clip index (no audio, no network)."""

import pandas as pd
import pytest

from src.data import indicvoices_jobs as ivj
from src.data.pilot_jobs import PilotError

WORDS = ["ek", "do", "teen", "chaar", "paanch", "chhah", "saat", "aath", "nau", "das"]


def _clips() -> pd.DataFrame:
    # transcripts without digits (the XTTS filter rejects them), unique per clip
    rows = []
    for pool, speakers in (("iv_train", ["iv_a", "iv_b", "iv_c"]), ("iv_test", ["iv_t"])):
        for s in speakers:
            for k in range(25):
                words = f"{WORDS[k % 10]} {WORDS[k // 10]} {s[-1]}"
                rows.append(
                    {
                        "speaker": s,
                        "pool": pool,
                        "utt_id": f"{s}_{k:03d}",
                        "transcript": f"{pool} ham pahle taalaab men jaate the vaakya {words}",
                        "duration_seconds": 3.0 + k * 0.3,
                    }
                )
    return pd.DataFrame(rows)


def _pools() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "speaker": ["iv_a", "iv_b", "iv_c", "iv_t"],
            "pool": ["iv_train"] * 3 + ["iv_test"],
            "source": "indicvoices",
        }
    )


def test_twenty_jobs_per_speaker_from_the_pool_only() -> None:
    jobs = ivj.build_jobs(_clips(), _pools(), "iv_train")
    assert jobs.groupby("speaker").size().to_dict() == {"iv_a": 20, "iv_b": 20, "iv_c": 20}
    assert jobs["transcript_source"].str.startswith("iv_train").all()
    assert (jobs["seed"] - ivj.SEEDS["iv_train"]).between(1, len(jobs)).all()


def test_test_pool_clones_read_test_text_only() -> None:
    jobs = ivj.build_jobs(_clips(), _pools(), "iv_test")
    assert set(jobs["speaker"]) == {"iv_t"}
    assert jobs["transcript_source"].str.startswith("iv_test").all()


def test_unknown_pool_is_refused() -> None:
    with pytest.raises((PilotError, KeyError)):
        ivj.build_jobs(_clips(), _pools(), "train")


def test_seed_blocks_never_overlap() -> None:
    sizes = {"iv_train": 150 * 20, "iv_dev": 20 * 20, "iv_test": 30 * 20}
    blocks = sorted((ivj.SEEDS[p] + 1, ivj.SEEDS[p] + n) for p, n in sizes.items())
    assert all(a[1] < b[0] for a, b in zip(blocks, blocks[1:], strict=False))
    assert blocks[0][0] > 9999  # clear of the earlier runs
