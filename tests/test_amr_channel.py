"""Smoke tests for src.data.amr_channel (needs ffmpeg with libopencore_amrnb)."""

import shutil

import numpy as np
import pandas as pd
import pytest

from src.data import amr_channel


def test_condition_name():
    assert amr_channel.condition_name("12.2k") == "amr_nb_12k2"
    assert amr_channel.condition_name("4.75k") == "amr_nb_4k75"


def test_unknown_bitrate_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no 64k mode"):
        amr_channel.encode_decode("x.wav", str(tmp_path / "y.wav"), "64k")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_render_band_limits_and_rewrites_manifest(tmp_path):
    sf = pytest.importorskip("soundfile")
    t = np.arange(16_000) / 16_000
    tone = (0.2 * np.sin(2 * np.pi * 300 * t) + 0.2 * np.sin(2 * np.pi * 6000 * t)).astype(
        np.float32
    )
    (tmp_path / "src" / "clips").mkdir(parents=True)
    sf.write(tmp_path / "src" / "clips" / "a.wav", tone, 16_000)
    manifest = pd.DataFrame({"filepath": ["${DATA_ROOT}/clips/a.wav"], "condition": ["clean"]})
    try:
        out = amr_channel.render(manifest, str(tmp_path / "out"), str(tmp_path / "src"))
    except Exception as exc:  # an ffmpeg build without the AMR encoder
        pytest.skip(f"AMR-NB encoder unavailable: {exc}")
    assert out["condition"].iloc[0] == "amr_nb_12k2"
    audio, sr = sf.read(tmp_path / "out" / "clips" / "a.wav")
    spectrum = np.abs(np.fft.rfft(audio)) ** 2
    freqs = np.fft.rfftfreq(len(audio), 1 / sr)
    assert sr == 16_000 and spectrum[freqs > 4000].sum() / spectrum.sum() < 0.01
