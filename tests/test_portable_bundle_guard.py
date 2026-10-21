"""A bundle must never write a whole recording where a clip was meant."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

sf = pytest.importorskip("soundfile")

from src.data import portable_bundle as pb  # noqa: E402


def test_whole_recordings_are_refused_and_clips_written(tmp_path):
    long_wav, short_wav = tmp_path / "lecture.wav", tmp_path / "clip.wav"
    sf.write(long_wav, np.zeros(16_000 * 61, dtype=np.float32) + 0.01, 16_000)
    sf.write(short_wav, np.zeros(16_000 * 3, dtype=np.float32) + 0.01, 16_000)
    manifest = pd.DataFrame(
        [
            {"filepath": str(long_wav), "label": "bonafide", "speaker": "a", "utt_id": "u1"},
            {"filepath": str(short_wav), "label": "bonafide", "speaker": "a", "utt_id": "u2"},
        ]
    )
    out = pb.build(manifest, tmp_path / "bundle")
    assert out["utt_id"].tolist() == ["u2"]
    assert not (tmp_path / "bundle/clips/u1.wav").exists()
