"""Tests for the three-way speaker pool carve and freeze (Week 3, W3-T4, SK).

If a speaker ever appears in two pools, a clone of a training voice ends up in the
eval column and every generalisation number in the paper is worthless. These tests
pin disjointness, determinism, and the freeze hash that detects a pool silently
moving after generation has started.

Pure pandas, no audio.
"""

import pandas as pd
import pytest

from src.data import speaker_pools as sp


def _speakers(n: int = 40) -> list[str]:
    return [f"spk{i:03d}" for i in range(n)]


# --------------------------------------------------------------------------- #
# Carving
# --------------------------------------------------------------------------- #
def test_three_pools_are_produced() -> None:
    pools = sp.carve_pools(_speakers())
    assert set(pools) == {"train", "adaptation", "eval"}


def test_every_speaker_lands_in_exactly_one_pool() -> None:
    speakers = _speakers(37)  # deliberately not divisible by the fractions
    pools = sp.carve_pools(speakers)
    flat = [s for members in pools.values() for s in members]
    assert sorted(flat) == sorted(speakers)
    assert len(flat) == len(set(flat))


def test_pools_are_disjoint() -> None:
    sp.check_disjoint(sp.carve_pools(_speakers()))  # must not raise


def test_carve_is_deterministic_for_a_seed() -> None:
    assert sp.carve_pools(_speakers(), seed=7) == sp.carve_pools(_speakers(), seed=7)


def test_different_seeds_give_different_carves() -> None:
    assert sp.carve_pools(_speakers(), seed=1) != sp.carve_pools(_speakers(), seed=2)


def test_fractions_are_respected_approximately() -> None:
    pools = sp.carve_pools(_speakers(100))
    assert abs(len(pools["train"]) - 50) <= 1
    assert abs(len(pools["adaptation"]) - 20) <= 1
    assert abs(len(pools["eval"]) - 30) <= 1


def test_duplicate_input_speakers_are_collapsed() -> None:
    pools = sp.carve_pools(["a", "a", "b", "c"])
    flat = [s for members in pools.values() for s in members]
    assert sorted(flat) == ["a", "b", "c"]


def test_fractions_must_sum_to_one() -> None:
    with pytest.raises(ValueError):
        sp.carve_pools(_speakers(), fractions={"train": 0.5, "adaptation": 0.2, "eval": 0.9})


def test_missing_pool_in_fractions_is_rejected() -> None:
    with pytest.raises(ValueError):
        sp.carve_pools(_speakers(), fractions={"train": 0.5, "eval": 0.5})


def test_overlap_is_detected() -> None:
    bad = {"train": ["a", "b"], "adaptation": ["b"], "eval": ["c"]}
    with pytest.raises(sp.PoolError):
        sp.check_disjoint(bad)


# --------------------------------------------------------------------------- #
# Frame + hashing
# --------------------------------------------------------------------------- #
def test_frame_has_one_row_per_speaker() -> None:
    frame = sp.pools_to_frame(sp.carve_pools(_speakers(40)))
    assert len(frame) == 40
    assert list(frame.columns) == sp.POOL_COLUMNS


def test_pool_lookup_works() -> None:
    pools = sp.carve_pools(_speakers(30), seed=3)
    frame = sp.pools_to_frame(pools)
    a_train_speaker = pools["train"][0]
    assert sp.pool_of(frame, a_train_speaker) == "train"
    assert sp.pool_of(frame, "not_a_speaker") is None


def test_hash_is_stable_under_row_order() -> None:
    frame = sp.pools_to_frame(sp.carve_pools(_speakers(20)))
    shuffled = frame.sample(frac=1.0, random_state=0)
    assert sp.frame_hash(frame) == sp.frame_hash(shuffled)


def test_hash_changes_when_a_speaker_moves_pool() -> None:
    frame = sp.pools_to_frame(sp.carve_pools(_speakers(20)))
    before = sp.frame_hash(frame)
    moved = frame.copy()
    moved.loc[moved.index[0], "pool"] = "eval"
    assert sp.frame_hash(moved) != before


# --------------------------------------------------------------------------- #
# Freeze + verify
# --------------------------------------------------------------------------- #
def test_freeze_writes_the_file_and_reports_counts(tmp_path) -> None:
    frame = sp.pools_to_frame(sp.carve_pools(_speakers(50)))
    record = sp.freeze(frame, str(tmp_path / "speaker_pools.csv"))
    assert (tmp_path / "speaker_pools.csv").is_file()
    assert record.n_speakers == 50
    assert sum(record.counts.values()) == 50


def test_verify_passes_on_an_unchanged_file(tmp_path) -> None:
    path = str(tmp_path / "speaker_pools.csv")
    record = sp.freeze(sp.pools_to_frame(sp.carve_pools(_speakers(30))), path)
    sp.verify_frozen(path, record.sha256)  # must not raise


def test_verify_fails_when_a_speaker_moved_pool(tmp_path) -> None:
    # The scenario this whole mechanism exists for.
    path = str(tmp_path / "speaker_pools.csv")
    record = sp.freeze(sp.pools_to_frame(sp.carve_pools(_speakers(30))), path)

    tampered = pd.read_csv(path)
    tampered.loc[0, "pool"] = "eval"
    tampered.to_csv(path, index=False)

    with pytest.raises(sp.PoolError, match="changed since it was frozen"):
        sp.verify_frozen(path, record.sha256)


def test_freeze_refuses_an_overlapping_frame(tmp_path) -> None:
    frame = pd.DataFrame(
        [
            {"speaker": "a", "pool": "train", "source": "mucs2021"},
            {"speaker": "a", "pool": "eval", "source": "mucs2021"},
        ]
    )
    with pytest.raises(sp.PoolError):
        sp.freeze(frame, str(tmp_path / "pools.csv"))


def test_freeze_record_describes_itself(tmp_path) -> None:
    record = sp.freeze(sp.pools_to_frame(sp.carve_pools(_speakers(40))), str(tmp_path / "p.csv"))
    described = record.describe()
    assert "train=" in described and "eval=" in described
