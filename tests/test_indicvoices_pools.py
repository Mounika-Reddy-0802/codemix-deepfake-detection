"""Tests for src/data/indicvoices_pools.py on synthetic metadata (no network)."""

from pathlib import Path

import pandas as pd
import pytest

from src.data import indicvoices_pools as ivp
from src.data import speaker_pools as sp


def _counts(n: int = 60) -> pd.Series:
    # every third speaker has too few usable clips and is not eligible
    return pd.Series({f"S{i:03d}": (5 if i % 3 == 0 else 25) for i in range(n)})


def test_pools_are_disjoint_sized_and_prefixed() -> None:
    frame = ivp.carve(_counts(), n_train=20, n_dev=5, n_test=10)
    assert frame["pool"].value_counts().to_dict() == {"iv_train": 20, "iv_test": 10, "iv_dev": 5}
    assert frame["speaker"].is_unique
    assert frame["speaker"].str.startswith("iv_").all()
    sp.check_disjoint({name: list(g["speaker"]) for name, g in frame.groupby("pool")})


def test_only_eligible_speakers_are_drawn() -> None:
    frame = ivp.carve(_counts(), n_train=20, n_dev=5, n_test=10)
    short = {f"iv_S{i:03d}" for i in range(0, 60, 3)}
    assert not short & set(frame["speaker"])


def test_carve_is_seeded() -> None:
    a = ivp.carve(_counts(), n_train=20, n_dev=5, n_test=10, seed=7)
    assert a.equals(ivp.carve(_counts(), n_train=20, n_dev=5, n_test=10, seed=7))
    assert not a.equals(ivp.carve(_counts(), n_train=20, n_dev=5, n_test=10, seed=8))


def test_too_few_eligible_speakers_is_refused() -> None:
    with pytest.raises(ValueError, match="extempore"):
        ivp.carve(_counts(30), n_train=20, n_dev=5, n_test=10)


def test_test_pool_is_evaluation_only_and_dev_pool_dev_only(tmp_path) -> None:
    path = tmp_path / "pools.csv"
    sp.freeze(ivp.carve(_counts(), n_train=20, n_dev=5, n_test=10), str(path))
    frame = pd.read_csv(path)
    assert ivp.evaluation_only_speakers(str(path)) == set(
        frame.loc[frame["pool"] == "iv_test", "speaker"]
    )
    assert ivp.dev_only_speakers(str(path)) == set(frame.loc[frame["pool"] == "iv_dev", "speaker"])
    assert ivp.evaluation_only_speakers(str(tmp_path / "absent.csv")) == set()


def _groups() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "shard": ["a.parquet"] * 3 + ["b.parquet"] * 2,
            "group": [0, 1, 2, 0, 1],
            "first_row": [0, 10, 20, 0, 10],
            "rows": [10, 10, 5, 10, 10],
        }
    )


def test_clips_get_their_row_group() -> None:
    clips = pd.DataFrame(
        {"shard": ["a.parquet", "a.parquet", "a.parquet", "b.parquet"], "row": [0, 10, 24, 15]}
    )
    assert ivp.assign_groups(clips, _groups()).tolist() == [0, 1, 2, 1]


def test_group_choice_is_seeded_and_sized() -> None:
    a = ivp.choose_groups(_groups(), n=3, seed=1)
    assert len(a) == 3
    assert a.equals(ivp.choose_groups(_groups(), n=3, seed=1))


def test_only_usable_clips_inside_chosen_groups_count() -> None:
    clips = pd.DataFrame(
        {
            "speaker_id": ["x", "x", "x", "y", "y"],
            "scenario": ["Extempore", "Extempore", "Read", "Extempore", "Extempore"],
            "duration": [5.0, 30.0, 5.0, 5.0, 6.0],
            "shard": ["a.parquet"] * 5,
            "group": [0, 0, 0, 1, 0],
        }
    )
    chosen = pd.DataFrame({"shard": ["a.parquet"], "group": [0]})
    # x: one usable clip (one is too long, one is read); y: one clip in the chosen group
    assert ivp.eligible_counts(clips, chosen).to_dict() == {"x": 1, "y": 1}


def test_the_committed_pools_are_still_the_frozen_ones():
    root = Path(__file__).resolve().parents[1]
    sp.verify_frozen(str(root / ivp.POOLS_PATH), ivp.FROZEN_SHA256)
