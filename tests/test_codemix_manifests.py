"""Tests for the Stage-3 manifest assembly (bonafide MUCS + generated spoof).

Pure pandas; ``spoof_rows`` needs real files on disk to check against (that is
the point of the check), so those tests use ``tmp_path``.
"""

import pandas as pd
import pytest

from src.data import codemix_manifests as cm


def _pools() -> pd.DataFrame:
    rows = [{"speaker": f"trn{i}", "pool": "train", "source": "mucs2021"} for i in range(3)]
    rows += [{"speaker": f"adp{i}", "pool": "adaptation", "source": "mucs2021"} for i in range(5)]
    rows += [{"speaker": f"evl{i}", "pool": "eval", "source": "mucs2021"} for i in range(4)]
    return pd.DataFrame(rows)


def _index() -> pd.DataFrame:
    rows = []
    for prefix, count in (("trn", 3), ("adp", 5), ("evl", 4)):
        for i in range(count):
            for j in range(3):
                rows.append(
                    {
                        "utt_id": f"{prefix}{i}_{j:03d}",
                        "speaker": f"{prefix}{i}",
                        "source": "mucs2021",
                        "wav_path": f"/data/{prefix}{i}.wav",
                        "start_seconds": float(j) * 10,
                        "end_seconds": float(j) * 10 + 8,
                        "duration_seconds": 8.0,
                        "transcript": f"transcript {prefix}{i} {j}",
                    }
                )
    return pd.DataFrame(rows)


#: matches _pools(): 5 adaptation speakers (adp0..adp4), 4 eval speakers (evl0..evl3)
_POOL_SPEAKER_COUNT = {"adp": 5, "evl": 4}


def _jobs(pool_prefix: str, n: int) -> pd.DataFrame:
    n_speakers = _POOL_SPEAKER_COUNT[pool_prefix]
    return pd.DataFrame(
        [
            {
                "job_id": i + 1,
                "speaker": f"{pool_prefix}{i % n_speakers}",
                "pool": "adaptation" if pool_prefix == "adp" else "eval",
                "transcript": f"job transcript {i}",
                "language": "hi",
                "tool": "xtts_v2",
                "output_path": f"outputs/xtts_v2_{pool_prefix}{i % n_speakers}_{i:05d}.wav",
            }
            for i in range(n)
        ]
    )


# --------------------------------------------------------------------------- #
# bonafide_rows
# --------------------------------------------------------------------------- #
def test_bonafide_rows_only_the_requested_pool():
    rows = cm.bonafide_rows(_index(), _pools(), "adaptation")
    assert set(rows["speaker"]) == {f"adp{i}" for i in range(5)}
    assert (rows["label"] == "bonafide").all()
    assert (rows["tool"] == "none").all()


def test_bonafide_rows_deduplicate_by_filepath():
    rows = cm.bonafide_rows(_index(), _pools(), "adaptation")
    # 5 speakers x 3 utterances share a wav_path per speaker (whole-recording path)
    assert len(rows) == 5


# --------------------------------------------------------------------------- #
# spoof_rows
# --------------------------------------------------------------------------- #
def test_spoof_rows_skips_missing_files(tmp_path):
    pack = tmp_path / "pack"
    (pack / "outputs").mkdir(parents=True)
    jobs = _jobs("adp", 4)
    # only materialise half the files
    for p in jobs["output_path"][:2]:
        (pack / p).write_bytes(b"RIFF....WAVEfmt ")

    rows = cm.spoof_rows(jobs, str(pack), "adaptation", _pools())
    assert len(rows) == 2
    assert (rows["label"] == "spoof").all()


def test_spoof_rows_empty_when_nothing_exists(tmp_path):
    pack = tmp_path / "pack"
    (pack / "outputs").mkdir(parents=True)
    jobs = _jobs("adp", 3)
    rows = cm.spoof_rows(jobs, str(pack), "adaptation", _pools())
    assert rows.empty


def test_spoof_rows_rejects_jobs_outside_the_pool(tmp_path):
    """A stale/mismatched generation_jobs.csv must fail loudly, not silently."""
    pack = tmp_path / "pack"
    (pack / "outputs").mkdir(parents=True)
    jobs = _jobs("evl", 3)  # eval-pool speakers, handed to the adaptation pool
    for p in jobs["output_path"]:
        (pack / p).write_bytes(b"RIFF....WAVEfmt ")

    with pytest.raises(cm.ManifestError, match="outside the adaptation pool"):
        cm.spoof_rows(jobs, str(pack), "adaptation", _pools())


# --------------------------------------------------------------------------- #
# split_speakers
# --------------------------------------------------------------------------- #
def test_split_speakers_is_disjoint_and_deterministic():
    speakers = [f"adp{i}" for i in range(10)]
    train_a, dev_a = cm.split_speakers(speakers, dev_frac=0.2, seed=1)
    train_b, dev_b = cm.split_speakers(speakers, dev_frac=0.2, seed=1)
    assert train_a == train_b and dev_a == dev_b
    assert set(train_a) & set(dev_a) == set()
    assert set(train_a) | set(dev_a) == set(speakers)
    assert len(dev_a) == 2


# --------------------------------------------------------------------------- #
# assert_pool_disjoint
# --------------------------------------------------------------------------- #
def test_pool_disjoint_passes_on_frozen_pools():
    cm.assert_pool_disjoint(_pools())  # must not raise


def test_pool_disjoint_catches_a_hand_edited_overlap():
    pools = _pools().copy()
    # a corrupted speaker_pools.csv: the same speaker id appears in both a
    # "adaptation" row and an "eval" row (e.g. a bad merge of two edits)
    dup = pools[pools["speaker"] == "adp0"].copy()
    dup["pool"] = "eval"
    pools = pd.concat([pools, dup], ignore_index=True)
    with pytest.raises(cm.ManifestError, match="both adaptation and eval"):
        cm.assert_pool_disjoint(pools)


# --------------------------------------------------------------------------- #
# build_adaptation_manifests / build_eval_manifest
# --------------------------------------------------------------------------- #
def _materialise(tmp_path, pool_prefix, n):
    pack = tmp_path / f"pack_{pool_prefix}"
    (pack / "outputs").mkdir(parents=True)
    jobs = _jobs(pool_prefix, n)
    for p in jobs["output_path"]:
        (pack / p).write_bytes(b"RIFF....WAVEfmt ")
    return jobs, str(pack)


def test_build_adaptation_manifests_train_dev_disjoint_by_speaker(tmp_path):
    jobs, pack_dir = _materialise(tmp_path, "adp", 20)
    train_df, dev_df = cm.build_adaptation_manifests(
        _index(), _pools(), jobs, pack_dir, dev_frac=0.2, seed=1
    )
    assert set(train_df["speaker"]) & set(dev_df["speaker"]) == set()
    assert set(train_df["split"]) == {"train"}
    assert set(dev_df["split"]) == {"dev"}
    assert {"bonafide", "spoof"} <= set(train_df["label"])


def test_build_adaptation_manifests_raises_without_spoof(tmp_path):
    pack = tmp_path / "empty_pack"
    (pack / "outputs").mkdir(parents=True)
    jobs = _jobs("adp", 3)  # no files written
    with pytest.raises(cm.ManifestError, match="no bonafide or no spoof"):
        cm.build_adaptation_manifests(_index(), _pools(), jobs, str(pack))


def test_build_eval_manifest_uses_only_eval_pool(tmp_path):
    jobs, pack_dir = _materialise(tmp_path, "evl", 16)
    eval_df = cm.build_eval_manifest(_index(), _pools(), jobs, pack_dir)
    assert set(eval_df["speaker"]) <= {f"evl{i}" for i in range(4)}
    assert set(eval_df["split"]) == {"eval"}
    assert {"bonafide", "spoof"} <= set(eval_df["label"])
