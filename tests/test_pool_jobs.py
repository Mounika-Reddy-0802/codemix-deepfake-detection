"""Tests for the adaptation/eval generation job table.

Same invariants as ``test_scale_jobs.py`` (pool firewall, no duplicate
(speaker, transcript) pairs, reproducibility) checked against the other two
pools, plus the guard that this module refuses to touch the train pool at all.

Pure pandas, no audio, no model.
"""

import pandas as pd
import pytest

from src.data import pool_jobs as pj
from src.data.pilot_jobs import PilotError

DEVA = "मुझे कल बैंक जाना है पैसे निकालने के लिए और फिर ऑफिस भी जाना है"


def _pools(n_adapt: int = 5, n_eval: int = 4) -> pd.DataFrame:
    rows = [{"speaker": f"trn{i}", "pool": "train", "source": "mucs2021"} for i in range(3)]
    rows += [
        {"speaker": f"adp{i}", "pool": "adaptation", "source": "mucs2021"} for i in range(n_adapt)
    ]
    rows += [{"speaker": f"evl{i}", "pool": "eval", "source": "mucs2021"} for i in range(n_eval)]
    return pd.DataFrame(rows)


def _index(n_adapt: int = 5, n_eval: int = 4, texts_per_speaker: int = 8) -> pd.DataFrame:
    rows = []
    for prefix, count in (("trn", 3), ("adp", n_adapt), ("evl", n_eval)):
        for i in range(count):
            for j in range(texts_per_speaker):
                rows.append(
                    {
                        "utt_id": f"{prefix}{i}_{j:03d}",
                        "speaker": f"{prefix}{i}",
                        "source": "mucs2021",
                        "wav_path": f"/data/{prefix}{i}.wav",
                        "start_seconds": 0.0,
                        "end_seconds": 12.0,
                        "duration_seconds": 12.0,
                        "transcript": f"{DEVA} {prefix}{chr(97 + i)}{chr(97 + j)}",
                    }
                )
    return pd.DataFrame(rows)


def _cfg(**kw):
    return pj.PoolJobConfig(**{"pool": "adaptation", "n_target": 30, **kw})


def test_train_pool_is_refused():
    with pytest.raises(PilotError, match="scale_jobs.py, not this module"):
        pj.PoolJobConfig(pool="train")


def test_adaptation_jobs_only_clone_adaptation_speakers():
    jobs = pj.build_pool_jobs(_index(), _pools(), config=_cfg(pool="adaptation"))
    assert set(jobs["speaker"]) <= {f"adp{i}" for i in range(5)}
    assert set(jobs["pool"]) == {"adaptation"}


def test_eval_jobs_only_clone_eval_speakers():
    jobs = pj.build_pool_jobs(_index(), _pools(), config=_cfg(pool="eval", n_target=20))
    assert set(jobs["speaker"]) <= {f"evl{i}" for i in range(4)}
    assert set(jobs["pool"]) == {"eval"}


def test_default_target_scales_with_speaker_count():
    # enough texts per speaker that 5 x CLIPS_PER_SPEAKER stays inside capacity
    index = _index(texts_per_speaker=200)
    jobs = pj.build_pool_jobs(index, _pools(), config=pj.PoolJobConfig(pool="adaptation"))
    assert len(jobs) == 5 * pj.CLIPS_PER_SPEAKER


def test_no_duplicate_speaker_transcript_pairs():
    jobs = pj.build_pool_jobs(_index(), _pools(), config=_cfg())
    pairs = jobs[["speaker", "transcript"]].astype(str)
    assert not pairs.duplicated().any()


def test_over_capacity_raises_rather_than_repeating():
    pools = _pools(n_adapt=2, n_eval=2)
    with pytest.raises(PilotError, match="only .* unique"):
        pj.build_pool_jobs(_index(n_adapt=2, n_eval=2), pools, config=_cfg(n_target=1000))


def test_reproducible_regardless_of_row_order():
    index, pools = _index(), _pools()
    shuffled = index.sample(frac=1.0, random_state=99).reset_index(drop=True)
    baseline = pj.build_pool_jobs(index, pools, config=_cfg())
    again = pj.build_pool_jobs(shuffled, pools, config=_cfg())
    pd.testing.assert_frame_equal(baseline, again)


def test_manually_injected_intruder_is_caught():
    """The firewall itself, not just the pool filter that feeds it."""
    jobs = pj.build_pool_jobs(_index(), _pools(), config=_cfg())
    jobs = jobs.copy()
    jobs.loc[0, "speaker"] = "evl0"  # an eval-pool speaker, injected by hand
    with pytest.raises(PilotError, match="outside the adaptation pool"):
        pj.assert_pool_job_invariants(jobs, _pools(), "adaptation")


def test_summary_reports_the_pool():
    jobs = pj.build_pool_jobs(_index(), _pools(), config=_cfg())
    summary = pj.pool_job_summary(jobs)
    assert summary["pools_used"] == ["adaptation"]
    assert summary["jobs"] == 30


# --------------------------------------------------------------------------- #
# CM04: the held-out tool rides the same builder, pointed at the eval pool
# --------------------------------------------------------------------------- #
def test_heldout_tool_is_recorded_on_every_row():
    """Tortoise (CM04) reuses this builder; the tool tag is what firewalls it."""
    jobs = pj.build_pool_jobs(
        _index(), _pools(), config=_cfg(pool="eval", n_target=20, tool="tortoise")
    )
    assert set(jobs["tool"]) == {"tortoise"}
    assert set(jobs["pool"]) == {"eval"}


def test_heldout_tool_never_touches_the_train_pool():
    """A Tortoise clip in a training manifest voids the unseen-attack claim."""
    pools = _pools()
    jobs = pj.build_pool_jobs(
        _index(), pools, config=_cfg(pool="eval", n_target=20, tool="tortoise")
    )
    train = set(pools.loc[pools["pool"] == "train", "speaker"].astype(str))
    assert not (set(jobs["speaker"].astype(str)) & train)
