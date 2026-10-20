"""Smoke tests for src.data.codec_mix."""

import pandas as pd
import pytest

from src.data import codec_mix as cm


def _half(condition, prefix, ids=("u1", "u2")):
    return pd.DataFrame(
        {
            "filepath": [f"${{DATA_ROOT}}/clips/{prefix}{u}.wav" for u in ids],
            "label": "bonafide",
            "speaker": "s",
            "condition": condition,
            "utt_id": list(ids),
        }
    )


def _write(tmp_path, g711, amr):
    g711.to_csv(tmp_path / f"{cm.G711}.csv", index=False)
    amr.to_csv(tmp_path / f"{cm.AMR}.csv", index=False)


def test_build_stacks_both_codecs(tmp_path):
    _write(tmp_path, _half("channel_g711_20db", ""), _half("amr_nb_12k2", "amr/"))
    out = cm.build(str(tmp_path))
    assert len(out) == 4 and set(out["condition"]) == {"channel_g711_20db", "amr_nb_12k2"}


def test_different_clips_are_refused(tmp_path):
    _write(tmp_path, _half("channel_g711_20db", ""), _half("amr_nb_12k2", "amr/", ("u1", "u3")))
    with pytest.raises(ValueError, match="different clips"):
        cm.build(str(tmp_path))


def test_colliding_paths_are_refused(tmp_path):
    _write(tmp_path, _half("channel_g711_20db", ""), _half("amr_nb_12k2", ""))
    with pytest.raises(ValueError, match="own folder"):
        cm.build(str(tmp_path))
