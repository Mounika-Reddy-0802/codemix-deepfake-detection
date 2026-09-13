"""Rolling-window streaming inference (W5-T5, owner SK): windowing, gate, smoothing."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from live_call.audio_source import PcmFrame
from src.inference import streaming as st

SR = st.SAMPLE_RATE


def _tone(seconds: float, amplitude: float = 0.1) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_window_count_on_synthetic_stream() -> None:
    windower = st.RollingWindower(4.0, 2.0)
    windows = list(windower.push(_tone(10.0)))
    # windows end at 4, 6, 8, 10 s
    assert [end for end, _ in windows] == [4.0, 6.0, 8.0, 10.0]
    assert all(w.size == 4 * SR for _, w in windows)


def test_windows_are_identical_however_the_stream_is_chunked() -> None:
    audio = _tone(9.0)
    whole = [w for _, w in st.RollingWindower().push(audio)]
    chunked_windower = st.RollingWindower()
    chunked = []
    for start in range(0, audio.size, 320):  # 20 ms frames
        chunked += [w for _, w in chunked_windower.push(audio[start : start + 320])]
    assert len(whole) == len(chunked)
    for a, b in zip(whole, chunked, strict=True):
        np.testing.assert_array_equal(a, b)


def test_hop_longer_than_window_is_rejected() -> None:
    with pytest.raises(ValueError):
        st.RollingWindower(2.0, 4.0)


def test_exponential_smoothing_is_stable() -> None:
    ema = st.EmaSmoother(0.5)
    values = [ema.update(x) for x in (1.0, 0.0, 1.0, 0.0, 1.0, 0.0)]
    assert values[0] == 1.0
    swings = np.abs(np.diff(values))
    # Input flips by 1.0 every window; the smoothed output never moves more than
    # alpha of that, and settles at alpha / (2 - alpha) instead of flipping.
    assert (swings <= 0.5 + 1e-12).all()
    assert swings[-1] == pytest.approx(1 / 3, abs=0.02)


def test_pcm16_round_trip() -> None:
    samples = np.array([0, 16384, -16384, 32767, -32768], dtype="<i2")
    out = st.pcm16_to_float(samples.tobytes() + b"\x01")  # stray odd byte is ignored
    np.testing.assert_allclose(out, samples / 32768.0)


def test_normalise_reaches_target_and_never_clips() -> None:
    quiet = _tone(4.0, amplitude=0.001)
    assert st.rms_dbfs(st.normalise(quiet)) == pytest.approx(st.TARGET_DBFS, abs=0.1)
    spiky = np.zeros(SR, dtype=np.float32)
    spiky[100] = 0.001
    assert np.max(np.abs(st.normalise(spiky))) <= 0.999 + 1e-6


def test_silent_windows_are_skipped_not_scored() -> None:
    calls = []
    scorer = st.StreamingScorer(lambda w: calls.append(w) or 0.9)
    audio = np.concatenate([np.zeros(6 * SR, dtype=np.float32), _tone(6.0)])
    results = list(scorer.push(audio))
    assert results[0].skipped and results[0].score is None
    assert any(not r.skipped for r in results)
    assert len(calls) == sum(not r.skipped for r in results)


def test_model_receives_normalised_windows() -> None:
    levels = []
    scorer = st.StreamingScorer(lambda w: levels.append(st.rms_dbfs(w)) or 0.5)
    list(scorer.push(_tone(8.0, amplitude=0.01)))
    assert levels and all(abs(level - st.TARGET_DBFS) < 0.1 for level in levels)


def test_run_consumes_a_pcm_frame_source() -> None:
    pcm = (_tone(6.0) * 32767).astype("<i2").tobytes()

    class Source:
        async def frames(self):
            for start in range(0, len(pcm), 640):
                yield PcmFrame(pcm=pcm[start : start + 640])

    async def collect():
        scorer = st.StreamingScorer(lambda w: 0.2)
        return [r async for r in scorer.run(Source())]

    results = asyncio.run(collect())
    assert [r.end_seconds for r in results] == [4.0, 6.0]
    assert results[-1].smoothed == pytest.approx(0.2)


def test_streaming_module_is_importable() -> None:
    import importlib

    mod = importlib.import_module("src.inference.streaming")
    assert mod.__doc__ is not None
