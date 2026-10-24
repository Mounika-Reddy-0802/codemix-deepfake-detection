"""Smoke tests for src.training.floor_final (clip lookup, selection, summary)."""

import pandas as pd
import pytest

from src.training import floor_final as ff


def _frame():
    return pd.DataFrame(
        {
            "filepath": ["${DATA_ROOT}/clips/a.wav", "${DATA_ROOT}/clips/amr/a.wav"],
            "label": ["bonafide", "spoof"],
            "tool": ["none", "tortoise"],
            "speaker": ["s1", "s1"],
        }
    )


def test_index_keeps_the_first_folder_and_separates_amr(tmp_path):
    first, second, amr = tmp_path / "one", tmp_path / "two", tmp_path / "amr"
    for d in (first, second, amr):
        d.mkdir()
        (d / "a.wav").write_bytes(b"x")
    index = ff.clip_index([str(first), str(second)], [str(amr)])
    assert index["a.wav"] == str(first / "a.wav")
    assert index["amr/a.wav"] == str(amr / "a.wav")
    located = ff.locate(_frame(), index)
    assert located["filepath"].tolist() == [str(first / "a.wav"), str(amr / "a.wav")]
    assert _frame()["filepath"][0].startswith("${DATA_ROOT}")  # the original is untouched


def test_a_missing_clip_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="2 clips not found"):
        ff.locate(_frame(), {})


def test_select_keeps_genuine_and_the_named_tool():
    other = _frame()
    other.loc[1, "tool"] = "xtts_v2"
    frame = pd.concat([_frame(), other], ignore_index=True)
    kept = ff.select(frame, ("tortoise",))
    assert kept["label"].tolist() == ["bonafide", "spoof", "bonafide"]
    assert set(kept.loc[kept["label"] == "spoof", "tool"]) == {"tortoise"}
    assert len(ff.select(frame, None)) == 4


def test_summary_counts_seeds_that_beat_the_floor():
    rows = [
        {
            "detector": "adapter",
            "test_set": "cm04",
            "eer_a": 0.1,
            "eer_b": 0.3,
            "diff": -0.2,
            "verdict": "beats floor",
        },
        {
            "detector": "adapter",
            "test_set": "cm04",
            "eer_a": 0.2,
            "eer_b": 0.3,
            "diff": -0.1,
            "verdict": "inconclusive",
        },
    ]
    (s,) = ff.summarise(rows)
    assert s["seeds"] == 2 and s["beats_floor"] == 1
    assert s["detector_eer"] == pytest.approx(0.15)
    assert "| adapter | cm04 | 2 |" in ff.markdown([s])


def test_the_feature_cache_gives_the_same_numbers_and_resumes(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    clips = tmp_path / "clips"
    clips.mkdir()
    rng = np.random.default_rng(0)
    for name in ("a.wav", "b.wav"):
        sf.write(str(clips / name), rng.normal(0, 0.1, 8000).astype("float32"), 16000)
    frame = pd.DataFrame(
        {
            "filepath": ["${DATA_ROOT}/clips/a.wav", "${DATA_ROOT}/clips/b.wav"],
            "label": ["bonafide", "spoof"],
            "speaker": ["s1", "s2"],
        }
    )
    index = ff.clip_index([str(clips)], [])
    monkeypatch.setattr(ff, "CACHE", None)
    _, x_plain, y_plain = ff.featurise(frame, index)
    monkeypatch.setattr(ff, "CACHE", tmp_path / "cache")
    rows, x_cached, y_cached = ff.featurise(frame, index)
    _, x_again, _ = ff.featurise(frame, index)  # read back from the cache
    assert np.allclose(x_plain, x_cached) and np.allclose(x_cached, x_again)
    assert y_plain.tolist() == y_cached.tolist() == [1, 0]
    assert rows["filepath"].tolist() == frame["filepath"].tolist()


def test_a_floor_above_chance_is_flagged_as_a_weak_bar():
    rows = [
        {
            "detector": "adapter",
            "test_set": "cm04",
            "eer_a": 0.1,
            "eer_b": 0.6,
            "diff": -0.5,
            "verdict": "beats floor",
        },
        {
            "detector": "adapter",
            "test_set": "eval_pool",
            "eer_a": 0.1,
            "eer_b": 0.2,
            "diff": -0.1,
            "verdict": "beats floor",
        },
    ]
    text = ff.markdown(ff.summarise(rows))
    assert "above chance on cm04:" in text and "weak bar" in text
