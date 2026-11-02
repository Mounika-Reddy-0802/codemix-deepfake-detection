"""Smoke tests for src.reporting.channel_spectrogram on synthetic audio."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.reporting import channel_spectrogram as cs


def _tone(freqs, seconds=1.0, sr=cs.SR):
    t = np.arange(int(seconds * sr)) / sr
    return sum(0.1 * np.sin(2 * np.pi * f * t) for f in freqs).astype(np.float32)


def test_log_spectrogram_finds_the_tone():
    db, freqs, times = cs.log_spectrogram(_tone([1000]))
    assert db.shape == (cs.N_FFT // 2 + 1, times.size)
    assert abs(freqs[db.mean(axis=1).argmax()] - 1000) < cs.SR / cs.N_FFT


def test_figure_writes_pdf_and_png(tmp_path: Path):
    pytest.importorskip("matplotlib")
    sf = pytest.importorskip("soundfile")  # audio wheel, not installed in CI

    paths = {}
    for row, _ in cs.ROWS:
        paths[row] = {}
        for col, _ in cs.COLUMNS:
            p = tmp_path / f"{row}_{col}.wav"
            sf.write(p, _tone([300, 2000] if col != "clean" else [300, 2000, 6000]), cs.SR)
            paths[row][col] = p
    written = cs.figure(paths, tmp_path / "out", seconds=0.5)
    assert sorted(Path(w).suffix for w in written) == [".pdf", ".png"]
    assert all(Path(w).stat().st_size > 1000 for w in written)


def test_pick_pair_needs_the_same_speaker_in_every_version(tmp_path: Path):
    roots = {k: tmp_path / k for k in ("clean", "g711", "amr")}
    for root in roots.values():
        root.mkdir()
    names = ["111_a_0001.wav", "222_b_0001.wav", "cm05_xtts_v2_222_00001.wav"]
    for root in roots.values():
        for n in names:
            (root / n).write_bytes(b"x")
    manifest = pd.DataFrame(
        {
            "filepath": [f"${{DATA_ROOT}}/clips/{n}" for n in names],
            "label": ["bonafide", "bonafide", "spoof"],
            "tool": ["none", "none", "xtts_v2"],
        }
    )
    assert cs.pick_pair(manifest, roots) == ("222_b_0001.wav", "cm05_xtts_v2_222_00001.wav")
    (roots["amr"] / "cm05_xtts_v2_222_00001.wav").unlink()
    with pytest.raises(LookupError):
        cs.pick_pair(manifest, roots)
