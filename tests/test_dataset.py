"""Tests for the dataset collate/padding logic (Week 2, owner M)."""

import numpy as np
import pytest

from src.data import dataset as ds


def test_pad_batch_shapes_and_lengths() -> None:
    arrays = [np.ones(3, np.float32), np.ones(5, np.float32), np.ones(1, np.float32)]
    padded, lengths = ds.pad_batch(arrays)
    assert padded.shape == (3, 5)
    assert list(lengths) == [3, 5, 1]


def test_pad_batch_zero_pads_tail() -> None:
    padded, _ = ds.pad_batch([np.array([1.0, 2.0], np.float32), np.array([9.0], np.float32)])
    assert np.array_equal(padded[1], np.array([9.0, 0.0], np.float32))


def test_pad_batch_empty() -> None:
    padded, lengths = ds.pad_batch([])
    assert padded.size == 0 and lengths.size == 0


def test_label_mapping() -> None:
    assert ds.LABEL_TO_INT["bonafide"] == 1
    assert ds.LABEL_TO_INT["spoof"] == 0


def test_collate_fn_builds_tensors() -> None:
    torch = pytest.importorskip("torch")
    batch = [
        ds.AudioSample(np.ones(3, np.float32), 1, "en", "spk1", "clean"),
        ds.AudioSample(np.ones(5, np.float32), 0, "hi", "spk2", "clean"),
    ]
    out = ds.collate_fn(batch)
    assert out["waveforms"].shape == (2, 5)
    assert out["labels"].tolist() == [1, 0]
    assert isinstance(out["waveforms"], torch.Tensor)
