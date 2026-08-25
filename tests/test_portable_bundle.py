"""Unit tests for the portable bundle writer.

``build`` had no direct test, which is how a missing loudness normalisation reached
every result this project has reported (W5-T4). These cover the write path itself:
naming, span cutting, and the level the audio lands at.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import portable_bundle as pb

soundfile = pytest.importorskip("soundfile")

SR = 16_000


def _tone(seconds: float = 1.0, amp: float = 0.5, freq: float = 440.0) -> np.ndarray:
    t = np.arange(int(seconds * SR), dtype=np.float64) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _dbfs(signal: np.ndarray) -> float:
    from src.utils.audio_utils import amp_to_db, rms

    return amp_to_db(rms(signal))


def _standalone(tmp_path, amps=(0.9, 0.05)):
    """Two clips at deliberately different levels, as XTTS and MUCS arrive."""
    src = tmp_path / "src"
    src.mkdir()
    rows = []
    for i, amp in enumerate(amps):
        path = src / f"c{i}.wav"
        soundfile.write(str(path), _tone(amp=amp), SR)
        rows.append(
            {
                "filepath": str(path),
                "label": "spoof" if i == 0 else "bonafide",
                "utt_id": f"c{i}",
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# clip_name
# --------------------------------------------------------------------------- #
def test_clip_name_uses_utt_id():
    assert pb.clip_name({"utt_id": "spk_rec_0007"}) == "spk_rec_0007.wav"


def test_clip_name_falls_back_to_a_hash_when_utt_id_is_missing():
    """NaN is truthy, so a missing utt_id once collapsed 1,280 rows onto nan.wav."""
    a = pb.clip_name({"utt_id": float("nan"), "filepath": "x.wav", "start_seconds": 0})
    b = pb.clip_name({"utt_id": float("nan"), "filepath": "y.wav", "start_seconds": 0})
    assert a != b
    assert "nan" not in (a, b)
    assert a.endswith(".wav") and len(a) == 24


def test_clip_name_is_stable_for_the_same_row():
    row = {"filepath": "x.wav", "start_seconds": 1.0, "end_seconds": 2.0}
    assert pb.clip_name(row) == pb.clip_name(dict(row))


# --------------------------------------------------------------------------- #
# Normalisation -- the W5-T4 fix
# --------------------------------------------------------------------------- #
def test_build_normalises_clips_to_the_target_level(tmp_path):
    """Both classes must leave the bundle at the same loudness.

    Without this, level alone separates XTTS from MUCS and a detector can score
    well without hearing anything -- the shortcut W5-T4 measured at 1.39% EER.
    """
    manifest = _standalone(tmp_path)
    pb.build(manifest, tmp_path / "bundle")

    levels = []
    for name in ("c0", "c1"):
        audio, _ = soundfile.read(str(tmp_path / "bundle" / "clips" / f"{name}.wav"))
        levels.append(_dbfs(audio.astype(np.float32)))

    for level in levels:
        assert level == pytest.approx(pb.TARGET_DBFS, abs=0.5)
    # 25 dB apart going in, together coming out.
    assert abs(levels[0] - levels[1]) < 1.0


def test_build_can_reproduce_the_unnormalised_bundles(tmp_path):
    """``normalise=False`` must still reproduce the pre-fix behaviour exactly."""
    manifest = _standalone(tmp_path)
    pb.build(manifest, tmp_path / "raw", normalise=False)

    a, _ = soundfile.read(str(tmp_path / "raw" / "clips" / "c0.wav"))
    b, _ = soundfile.read(str(tmp_path / "raw" / "clips" / "c1.wav"))
    assert abs(_dbfs(a.astype(np.float32)) - _dbfs(b.astype(np.float32))) > 20.0


def test_build_honours_a_custom_target_dbfs(tmp_path):
    manifest = _standalone(tmp_path, amps=(0.5,))
    pb.build(manifest, tmp_path / "bundle", target_dbfs=-30.0)
    audio, _ = soundfile.read(str(tmp_path / "bundle" / "clips" / "c0.wav"))
    assert _dbfs(audio.astype(np.float32)) == pytest.approx(-30.0, abs=0.5)


def test_normalisation_does_not_clip(tmp_path):
    """Gain is applied with a peak guard, so a loud clip cannot come back distorted."""
    src = tmp_path / "src"
    src.mkdir()
    soundfile.write(str(src / "loud.wav"), _tone(amp=0.99), SR)
    manifest = pd.DataFrame([{"filepath": str(src / "loud.wav"), "label": "spoof", "utt_id": "l"}])

    pb.build(manifest, tmp_path / "bundle")
    audio, _ = soundfile.read(str(tmp_path / "bundle" / "clips" / "l.wav"))
    assert np.max(np.abs(audio)) <= 1.0


def test_build_leaves_silence_alone(tmp_path):
    """A silent clip has no RMS to scale; it must pass through, not divide by zero."""
    src = tmp_path / "src"
    src.mkdir()
    soundfile.write(str(src / "q.wav"), np.zeros(SR, dtype=np.float32), SR)
    manifest = pd.DataFrame([{"filepath": str(src / "q.wav"), "label": "spoof", "utt_id": "q"}])

    out = pb.build(manifest, tmp_path / "bundle")
    audio, _ = soundfile.read(str(tmp_path / "bundle" / "clips" / "q.wav"))
    assert len(out) == 1
    assert np.isfinite(audio).all()


# --------------------------------------------------------------------------- #
# Manifest shape
# --------------------------------------------------------------------------- #
def test_build_rewrites_paths_to_the_data_root_token(tmp_path):
    manifest = _standalone(tmp_path)
    out = pb.build(manifest, tmp_path / "bundle")
    assert out["filepath"].tolist() == [
        "${DATA_ROOT}/clips/c0.wav",
        "${DATA_ROOT}/clips/c1.wav",
    ]
    assert out["label"].tolist() == manifest["label"].tolist()


def test_build_drops_the_span_columns_from_the_portable_copy(tmp_path):
    """Spans describe the source recording; a materialised clip no longer has one."""
    src = tmp_path / "src"
    src.mkdir()
    soundfile.write(str(src / "rec.wav"), _tone(seconds=4.0), SR)
    manifest = pd.DataFrame(
        [
            {
                "filepath": str(src / "rec.wav"),
                "label": "bonafide",
                "utt_id": "u0",
                "start_seconds": 1.0,
                "end_seconds": 2.0,
            }
        ]
    )
    out = pb.build(manifest, tmp_path / "bundle")
    assert "start_seconds" not in out.columns
    assert "end_seconds" not in out.columns


def test_build_reuses_an_existing_clip_instead_of_rewriting(tmp_path):
    manifest = _standalone(tmp_path)
    pb.build(manifest, tmp_path / "bundle")
    target = tmp_path / "bundle" / "clips" / "c0.wav"
    marker = target.stat().st_mtime_ns

    out = pb.build(manifest, tmp_path / "bundle")
    assert len(out) == 2
    assert target.stat().st_mtime_ns == marker


def test_build_skips_an_unreadable_clip_without_stopping(tmp_path):
    manifest = _standalone(tmp_path)
    manifest = pd.concat(
        [manifest, pd.DataFrame([{"filepath": "nowhere.wav", "label": "spoof", "utt_id": "gone"}])],
        ignore_index=True,
    )
    out = pb.build(manifest, tmp_path / "bundle")
    assert len(out) == 2


def test_build_resolves_a_portable_manifest_against_data_root(tmp_path):
    """A bundle must be re-buildable from its own portable manifest.

    ``data_root`` was accepted and then ignored, so feeding a previous build's
    output back in handed ``${DATA_ROOT}/clips/x.wav`` straight to the loader and
    every row failed -- which is what blocked re-normalising the bundles in place.
    """
    manifest = _standalone(tmp_path)
    first = pb.build(manifest, tmp_path / "one")

    second = pb.build(first, tmp_path / "two", data_root=str(tmp_path / "one"))
    assert len(second) == len(first)
    assert (tmp_path / "two" / "clips" / "c0.wav").is_file()
    audio, _ = soundfile.read(str(tmp_path / "two" / "clips" / "c0.wav"))
    assert _dbfs(audio.astype(np.float32)) == pytest.approx(pb.TARGET_DBFS, abs=0.5)
