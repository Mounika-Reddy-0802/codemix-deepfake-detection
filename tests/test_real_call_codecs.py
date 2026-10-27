"""The call chain rebuilt on direct recordings (week 14)."""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from src.inference import real_call_codecs as rcc


def _needs_codec_tools() -> None:
    pytest.importorskip("soundfile")  # audio wheels, not installed in CI
    pytest.importorskip("librosa")
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")


def _voice(seconds: float = 6.0, sr: int = 16_000) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    wave = sum(np.sin(2 * np.pi * 140 * k * t) / k for k in range(1, 10))
    return (0.05 * wave * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)


def test_versions_are_the_chain_in_order() -> None:
    assert rcc.VERSIONS == ("direct", "g711", "amr_nb", "amr_nb_rec", "amr_wb_rec")


def test_every_version_keeps_the_length_and_the_narrow_ones_lose_the_upper_band() -> None:
    _needs_codec_tools()
    audio = _voice()
    versions = rcc.chain_versions(audio)
    assert list(versions) == ["direct", "g711", "amr_nb", "amr_nb_rec", "amr_wb_rec"]
    spectrum = {}
    for name, signal in versions.items():
        assert abs(signal.size - audio.size) < 0.2 * rcc.SAMPLE_RATE, name
        power = np.abs(np.fft.rfft(signal)) ** 2
        freqs = np.fft.rfftfreq(signal.size, 1 / rcc.SAMPLE_RATE)
        spectrum[name] = power[freqs > 4_200].sum() / power.sum()
    # everything through an 8 kHz stage has (almost) nothing above 4 kHz
    for name in ("g711", "amr_nb", "amr_nb_rec", "amr_wb_rec"):
        assert spectrum[name] < 0.01, name


def test_score_chain_scores_every_version_for_every_model() -> None:
    _needs_codec_tools()
    models = {"steady": (lambda window: 0.9, 0.5), "doubtful": (lambda window: 0.3, 0.5)}
    table = rcc.score_chain({"P2_read_A": _voice(8.0)}, models)
    assert len(table) == 2 * len(rcc.VERSIONS)
    steady = table[table["model"] == "steady"]
    assert (steady["mean_p_genuine"] == 0.9).all() and (steady["fake_leaning_pct"] == 0).all()
    doubtful = table[table["model"] == "doubtful"]
    assert (doubtful["fake_leaning_pct"] == 100).all()
