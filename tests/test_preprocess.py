"""Unit tests for preprocessing segmentation + pipeline (Week 2, owner L)."""

import numpy as np

from src.data import preprocess as pp


def test_short_signal_within_window_is_one_segment() -> None:
    sr = 16_000
    audio = np.ones(int(5 * sr), dtype=np.float32) * 0.1  # 5 s
    segs = pp.segment(audio, sr, min_seconds=2.0, max_seconds=10.0)
    assert len(segs) == 1
    assert segs[0].size == audio.size


def test_long_signal_is_chunked_by_max_seconds() -> None:
    sr = 16_000
    audio = np.ones(int(25 * sr), dtype=np.float32) * 0.1  # 25 s -> 10+10+5
    segs = pp.segment(audio, sr, min_seconds=2.0, max_seconds=10.0)
    assert len(segs) == 3
    assert segs[0].size == 10 * sr
    assert segs[-1].size == 5 * sr


def test_too_short_tail_is_dropped() -> None:
    sr = 16_000
    audio = np.ones(int(21 * sr), dtype=np.float32) * 0.1  # 10+10+1 -> tail dropped
    segs = pp.segment(audio, sr, min_seconds=2.0, max_seconds=10.0)
    assert len(segs) == 2


def test_signal_below_min_yields_nothing() -> None:
    sr = 16_000
    audio = np.ones(int(1 * sr), dtype=np.float32) * 0.1  # 1 s < 2 s min
    assert pp.segment(audio, sr, min_seconds=2.0, max_seconds=10.0) == []


def test_pipeline_without_vad_produces_normalised_segments() -> None:
    sr = 16_000
    audio = np.random.default_rng(0).standard_normal(int(12 * sr)).astype(np.float32) * 0.01
    cfg = pp.PreprocessConfig(vad=False, target_dbfs=-20.0)
    segs = pp.preprocess_signal(audio, sr, cfg)
    assert len(segs) == 2  # 12 s -> 10 + 2
    for s in segs:
        assert np.max(np.abs(s)) <= 1.0
