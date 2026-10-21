"""Tests for the extension pools and for the guard that keeps them out of training."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data import speaker_pools_ext as ext
from src.training import config_guard as cg

HI = "मुझे"


def _index(n_speakers=12, english_for=("u3", "u4")):
    rows = []
    for k in range(n_speakers):
        speaker = f"u{k}"
        for i in range(50):
            text = (
                "plain english words here"
                if speaker in english_for and i == 0
                else f"{HI} {HI} {HI}"
            )
            rows.append(
                {
                    "utt_id": f"{speaker}_{i}",
                    "speaker": speaker,
                    "source": "mucs2021",
                    "duration_seconds": 3.0,
                    "transcript": text,
                }
            )
    rows.append(
        {
            "utt_id": "f_0",
            "speaker": "frozen1",
            "source": "mucs2021",
            "duration_seconds": 3.0,
            "transcript": f"{HI} {HI}",
        }
    )
    return pd.DataFrame(rows)


FROZEN = pd.DataFrame({"speaker": ["frozen1"], "pool": ["eval"], "source": ["mucs2021"]})


def test_extension_skips_frozen_speakers_and_is_disjoint():
    frame = ext.carve_extension(_index(), FROZEN, n_eval=4, min_clips=40, seed=1)
    assert "frozen1" not in set(frame["speaker"])
    assert frame["speaker"].is_unique
    assert (frame["pool"] == "eval_ext").sum() == 4
    control = set(frame.loc[frame["pool"] == "control", "speaker"])
    assert control <= {"u3", "u4"}  # only speakers with an all-English clip


def test_extension_is_deterministic_and_checks_supply():
    a = ext.carve_extension(_index(), FROZEN, n_eval=4, seed=3)
    b = ext.carve_extension(_index(), FROZEN, n_eval=4, seed=3)
    assert a.equals(b)
    with pytest.raises(ValueError):
        ext.carve_extension(_index(n_speakers=3), FROZEN, n_eval=4)


def test_guard_rejects_evaluation_only_extension_speakers(tmp_path):
    manifests = tmp_path / "data/manifests"
    manifests.mkdir(parents=True)
    FROZEN.to_csv(manifests / "speaker_pools.csv", index=False)
    pd.DataFrame(
        {
            "speaker": ["c1", "x1", "a1"],
            "pool": ["control", "eval_ext", "adapt_ext"],
            "source": "mucs2021",
        }
    ).to_csv(manifests / "speaker_pools_ext.csv", index=False)
    (tmp_path / "configs").mkdir()
    for speaker, should_fail in (("c1", True), ("x1", True), ("a1", False)):
        pd.DataFrame(
            [
                {
                    "filepath": "${DATA_ROOT}/clips/a.wav",
                    "label": "bonafide",
                    "language": "hi-en",
                    "speaker": speaker,
                    "source": "mucs2021",
                    "tool": "none",
                    "condition": "clean",
                    "split": "train",
                }
            ]
        ).to_csv(tmp_path / "data/m.csv", index=False)
        (tmp_path / "configs/train_x.yaml").write_text(
            "train_manifest: data/m.csv\nsystem: s3_native\nlora: false\n"
        )
        report = cg.check_config("configs/train_x.yaml", root=str(tmp_path))
        failed = any("eval-pool" in p for p in report.problems)
        assert failed == should_fail, (speaker, report.problems)
