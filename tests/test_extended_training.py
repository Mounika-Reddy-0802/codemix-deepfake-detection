"""Smoke tests for src.data.extended_training (the corpus-overfit fix)."""

import pandas as pd
import pytest

from src.data import extended_training as et


def _rows(stem, label, speakers, split="train"):
    return pd.DataFrame(
        {
            "filepath": [
                f"${{DATA_ROOT}}/clips/{stem}_{s}_{i}.wav" for i, s in enumerate(speakers)
            ],
            "label": label,
            "language": "hi-en",
            "speaker": [str(s) for s in speakers],
            "source": "mucs2021",
            "tool": "none" if label == "bonafide" else "xtts_v2",
            "condition": "channel_g711_20db",
            "utt_id": [f"{stem}_{i}" for i in range(len(speakers))],
            "split": split,
        }
    )


def _write(tmp_path, extra_genuine=(10, 11), dev=(1,)):
    _rows("base", "bonafide", [2, 3]).to_csv(tmp_path / f"{et.BASE}.csv", index=False)
    _rows("dev", "bonafide", list(dev), "dev").to_csv(tmp_path / f"{et.DEV}.csv", index=False)
    _rows("ext", "bonafide", list(extra_genuine)).to_csv(
        tmp_path / f"{et.ADDITIONS[0]}.csv", index=False
    )
    _rows("cm08", "spoof", [10]).to_csv(tmp_path / f"{et.ADDITIONS[1]}.csv", index=False)
    pd.DataFrame(
        {
            "speaker": ["10", "11", "50"],
            "pool": ["adapt_ext", "adapt_ext", "eval_ext"],
            "source": "mucs",
        }
    ).to_csv(tmp_path / et.POOLS_EXT, index=False)


def test_build_appends_both_additions(tmp_path):
    _write(tmp_path)
    out = et.build(str(tmp_path))
    assert len(out) == 5
    assert out["label"].value_counts().to_dict() == {"bonafide": 4, "spoof": 1}


def test_eval_speaker_is_refused(tmp_path):
    _write(tmp_path, extra_genuine=(10, 50))
    with pytest.raises(ValueError, match="outside the adapt_ext pool"):
        et.build(str(tmp_path))


def test_dev_speaker_is_refused(tmp_path):
    _write(tmp_path, dev=(11,))
    with pytest.raises(ValueError, match="dev set"):
        et.build(str(tmp_path))
