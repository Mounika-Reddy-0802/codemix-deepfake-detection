"""Tests for the speaker-held-out CM02 split (P-025).

The property that makes the RVC ablation measurable: no speaker -- target or
source -- is on both sides, and every fake travels with its real source.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data import rvc_holdout as rh


def _jobs() -> pd.DataFrame:
    rows = []
    speakers = [f"s{i}" for i in range(8)]
    targets = speakers[:4]
    n = 0
    for t in targets:
        for s in speakers:
            if s == t:
                continue
            n += 1
            rows.append(
                {
                    "target_speaker": t,
                    "source_speaker": s,
                    "source_utt_id": f"{s}_u{n % 3}",
                    "output_path": f"/kaggle/working/rvc_outputs/rvc_{t}_{n:05d}.wav",
                }
            )
    return pd.DataFrame(rows)


def _qa(jobs: pd.DataFrame, fail: set[str] = frozenset()) -> pd.DataFrame:
    clips = jobs["output_path"].map(lambda p: p.split("/")[-1])
    return pd.DataFrame({"clip": clips, "ok": [c not in fail for c in clips]})


def test_no_speaker_is_on_both_sides():
    jobs = _jobs()
    train, test = rh.holdout_split(jobs, _qa(jobs))
    assert not set(train.speaker) & set(test.speaker)


def test_targets_are_split_evenly():
    jobs = _jobs()
    train, test = rh.holdout_split(jobs, _qa(jobs))
    assert train[train.label == "spoof"].speaker.nunique() == 2
    assert test[test.label == "spoof"].speaker.nunique() == 2


def test_every_half_carries_real_sources():
    jobs = _jobs()
    train, test = rh.holdout_split(jobs, _qa(jobs))
    for half in (train, test):
        assert (half.label == "bonafide").sum() > 0
        assert (half.label == "spoof").sum() > 0


def test_cross_half_conversions_are_dropped_not_kept():
    jobs = _jobs()
    train, test = rh.holdout_split(jobs, _qa(jobs))
    assert (train.label == "spoof").sum() + (test.label == "spoof").sum() < len(jobs)


def test_failed_qa_clips_are_excluded():
    jobs = _jobs()
    first = jobs["output_path"].iloc[0].split("/")[-1]
    train, test = rh.holdout_split(jobs, _qa(jobs, {first}))
    everything = pd.concat([train, test])
    assert not everything.filepath.str.endswith(first).any()


def test_split_labels_and_portable_paths():
    jobs = _jobs()
    train, test = rh.holdout_split(jobs, _qa(jobs))
    assert set(train.split) == {"train"} and set(test.split) == {"eval"}
    both = pd.concat([train, test])
    assert both.filepath.str.startswith("${DATA_ROOT}/").all()
    assert not both.filepath.str.contains("kaggle").any()


def test_check_disjoint_catches_a_leak():
    a = pd.DataFrame({"speaker": ["x", "y"]})
    b = pd.DataFrame({"speaker": ["y", "z"]})
    with pytest.raises(rh.HoldoutError, match="both sides"):
        rh.check_disjoint(a, b)
