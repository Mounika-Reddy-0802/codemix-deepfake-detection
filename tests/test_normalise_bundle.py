"""Tests for the bundle level-normalisation fix (W5-T4).

The guard check is pure numpy and runs in CI. The file round-trip needs
``soundfile``, which CI does not install, so it skips there and runs locally.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import normalise_bundle as nb
from src.utils.audio_utils import rms, rms_normalize


def _tone(amplitude: float, n: int = 16_000) -> np.ndarray:
    return (amplitude * np.sin(np.linspace(0, 200 * np.pi, n))).astype(np.float32)


def test_loud_and_quiet_clips_land_at_the_same_rms():
    """The whole point of the fix: level stops being a class label."""
    loud, quiet = rms_normalize(_tone(0.9)), rms_normalize(_tone(0.05))
    assert rms(loud) == pytest.approx(rms(quiet), rel=1e-3)


def test_a_smooth_tone_does_not_trip_the_guard():
    assert nb.trips_peak_guard(_tone(0.9)) is False  # sine crest factor ~3 dB


def test_a_spiky_clip_trips_the_guard():
    spiky = np.zeros(16_000, dtype=np.float32)
    spiky[::4000] = 1.0  # tiny RMS, full-scale peaks: gain to -23 dBFS would clip
    assert nb.trips_peak_guard(spiky) is True


def test_silence_does_not_trip_the_guard():
    assert nb.trips_peak_guard(np.zeros(1000, dtype=np.float32)) is False


def test_normalise_clips_round_trip_and_counts_per_class(tmp_path):
    sf = pytest.importorskip("soundfile")
    src = tmp_path / "src" / "clips"
    src.mkdir(parents=True)
    sf.write(src / "a.wav", _tone(0.9), 16_000)
    sf.write(src / "b.wav", _tone(0.05), 16_000)
    manifest = pd.DataFrame(
        {
            "filepath": ["${DATA_ROOT}/clips/a.wav", "${DATA_ROOT}/clips/b.wav"],
            "label": ["spoof", "bonafide"],
        }
    )
    stats = nb.normalise_clips(manifest, str(tmp_path / "src"), str(tmp_path / "out"))
    assert stats["spoof"]["clips"] == 1 and stats["bonafide"]["clips"] == 1
    a, _ = sf.read(tmp_path / "out" / "clips" / "a.wav", dtype="float32")
    b, _ = sf.read(tmp_path / "out" / "clips" / "b.wav", dtype="float32")
    assert rms(a) == pytest.approx(rms(b), rel=1e-2)

    again = nb.normalise_clips(manifest, str(tmp_path / "src"), str(tmp_path / "out"))
    assert again["spoof"]["written"] == 0  # resumable: nothing rewritten
