"""Smoke tests for src.data.score_sets (week-13 scoring sets)."""

import zipfile

import pandas as pd
import pytest

from src.data import score_sets


def _rows(label, source, tool, speakers, condition="clean"):
    return pd.DataFrame(
        {
            "filepath": [
                f"${{DATA_ROOT}}/clips/{source}_{s}_{i}.wav" for i, s in enumerate(speakers)
            ],
            "label": label,
            "language": "hi-en",
            "speaker": [str(s) for s in speakers],
            "source": source,
            "tool": tool,
            "condition": condition,
            "split": "eval",
        }
    )


def _zip(path, attack, speakers):
    with zipfile.ZipFile(path, "w") as zf:
        for kind in ("clean", "channel"):
            for i, s in enumerate(speakers):
                zf.writestr(f"gen_bundle/{kind}/clips/{attack}_{s}_{i}.wav", b"RIFF")
        zf.writestr(
            f"gen_bundle/manifests/{attack}.csv",
            _rows("spoof", attack, "xtts_v2", speakers).to_csv(index=False),
        )
        zf.writestr(
            f"gen_bundle/manifests/{attack}_channel20.csv",
            _rows("spoof", attack, "xtts_v2", speakers, "channel_g711_20db").to_csv(index=False),
        )
        zf.writestr("gen_bundle/summary.txt", "ok")


def test_unpack_copies_both_conditions(tmp_path):
    _zip(tmp_path / "a.zip", "cm05", [1, 2])
    counts = score_sets.unpack(
        str(tmp_path / "a.zip"), str(tmp_path / "clean"), str(tmp_path / "ch")
    )
    assert counts == {"clean": 2, "channel": 2}
    assert len(list((tmp_path / "clean" / "clips").glob("cm05_*.wav"))) == 2


def test_score_set_refuses_unknown_speakers():
    genuine = _rows("bonafide", "mucs2021", "none", [1, 2])
    with pytest.raises(ValueError, match="no genuine clips"):
        score_sets.score_set(genuine, [_rows("spoof", "cm05", "xtts_v2", [3])])


def test_score_set_refuses_training_attack():
    genuine = _rows("bonafide", "mucs2021", "none", [1])
    with pytest.raises(ValueError, match="training attack"):
        score_sets.score_set(genuine, [_rows("spoof", "cm08", "xtts_v2", [1])])


def test_build_writes_sets_only_when_all_attacks_exist(tmp_path):
    for suffix, cond in score_sets.CONDITIONS.items():
        _rows("bonafide", "mucs2021", "none", [1, 2], cond).to_csv(
            tmp_path / f"eval_ext_genuine{suffix}.csv", index=False
        )
    attacks = {
        "cm05": _rows("spoof", "cm05", "xtts_v2", [1]),
        "cm05_channel20": _rows("spoof", "cm05", "xtts_v2", [1], "channel"),
    }
    assert score_sets.build(str(tmp_path), attacks) == {}  # cm06 still missing
    attacks.update(
        {
            "cm06": _rows("spoof", "cm06", "tortoise", [2]),
            "cm06_channel20": _rows("spoof", "cm06", "tortoise", [2], "channel"),
        }
    )
    written = score_sets.build(str(tmp_path), attacks)
    assert written == {"score_ext": 4, "score_ext_channel20": 4}
    scored = pd.read_csv(tmp_path / "score_ext.csv")
    assert scored["label"].tolist() == ["bonafide", "bonafide", "spoof", "spoof"]
