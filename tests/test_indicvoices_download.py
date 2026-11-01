"""Tests for src/data/indicvoices_download.py: row selection, ids, manifest, decoding."""

import io

import numpy as np
import pandas as pd

from src.data import indicvoices_download as dl


def _pools() -> pd.DataFrame:
    return pd.DataFrame(
        {"speaker": ["iv_A", "iv_B"], "pool": ["iv_train", "iv_test"], "source": "indicvoices"}
    )


def _group() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "speaker_id": ["A", "A", "B", "C", "B"],
            "scenario": ["Extempore", "Read", "Extempore", "Extempore", "Extempore"],
            "duration": [5.0, 5.0, 20.0, 6.0, 3.0],
            "text": ["एक", "दो", "तीन", "चार", "पाँच"],
            "gender": ["Female", "Female", "Male", "Male", "Male"],
            "age_group": ["18-30"] * 5,
            "area": ["Rural"] * 5,
            "state": ["Bihar"] * 5,
        }
    )


def test_only_usable_clips_of_pooled_speakers_are_kept() -> None:
    kept = dl.select_rows(_group(), _pools())
    # A's read clip, B's 20 s clip and unpooled C are left out
    assert kept.index.tolist() == [0, 4]
    assert kept["split"].tolist() == ["iv_train", "iv_test"]


def test_ids_are_stable_and_name_the_source_row() -> None:
    assert dl.utt_id("A", "train-00006-of-00082.parquet", 4392) == "iv_A_t00006_04392"
    assert dl.utt_id("A", "valid-00000-of-00001.parquet", 7) == "iv_A_v00000_00007"


def test_manifest_rows_have_the_project_columns_and_offsets() -> None:
    kept = dl.select_rows(_group(), _pools())
    rows = dl.manifest_rows(kept, "train-00006-of-00082.parquet", first_row=100)
    assert list(rows.columns) == dl.COLUMNS
    assert rows["utt_id"].tolist() == ["iv_A_t00006_00100", "iv_B_t00006_00104"]
    assert rows["filepath"].iloc[0] == "${DATA_ROOT}/clips/iv_A_t00006_00100.wav"
    assert set(rows["label"]) == {"bonafide"} and set(rows["language"]) == {"hi"}
    assert rows["text"].tolist() == ["एक", "पाँच"]


def test_decode_resamples_to_16k_mono() -> None:
    import soundfile as sf

    buf = io.BytesIO()
    stereo = np.zeros((8000, 2), dtype=np.float32)
    sf.write(buf, stereo, 8000, format="WAV")
    out = dl.decode(buf.getvalue())
    assert out.ndim == 1 and len(out) == 16000
