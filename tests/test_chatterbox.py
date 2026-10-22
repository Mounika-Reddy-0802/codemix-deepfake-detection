"""Tests for the Chatterbox driver: job table, audio conversion, resumable generation."""

import numpy as np
import pandas as pd
import pytest

from src.data import chatterbox_generation as cg
from src.data import generate_cli
from src.training import config_guard as guard


def _table():
    return pd.DataFrame(
        {
            "job_id": [0, 1],
            "speaker": ["a", "b"],
            "reference_wav": ["refs/a.wav", "refs/b.wav"],
            "transcript": ["नमस्ते दुनिया", "यह एक परीक्षण है"],
            "language": "hi",
            "tool": "indicf5",
            "output_path": ["outputs/indicf5_a_00001.wav", "outputs/indicf5_b_00002.wav"],
            "seed": [11, 12],
        }
    )


def test_job_table_keeps_speakers_and_sentences_and_renames_outputs():
    out = cg.chatterbox_jobs(_table())
    assert set(out["tool"]) == {"chatterbox"}
    assert out["output_path"].tolist() == ["chatterbox_a_00001.wav", "chatterbox_b_00002.wav"]
    assert out["transcript"].tolist() == _table()["transcript"].tolist()
    assert out["speaker"].tolist() == ["a", "b"]


def test_audio_is_returned_as_flat_float32():
    torch = pytest.importorskip("torch")
    wave = cg.to_numpy(torch.ones(1, 7, dtype=torch.float64))
    assert wave.shape == (7,) and wave.dtype == np.float32


def test_chatterbox_is_a_registered_tool_and_held_out():
    assert "chatterbox" in generate_cli.TOOLS
    row = pd.DataFrame(
        [
            {
                "filepath": "${DATA_ROOT}/clips/a.wav",
                "label": "spoof",
                "language": "hi-en",
                "speaker": "t1",
                "source": "cm09",
                "tool": "chatterbox",
                "condition": "clean",
                "split": "train",
            }
        ]
    )
    problems = guard.check_rows(row, {"lora": True}, set(), set())
    assert any("held-out tool" in p for p in problems)
    assert guard.check_rows(row.assign(tool="xtts_v2"), {"lora": True}, set(), set()) == []


class _FakeModel:
    sr = 24_000

    def __init__(self):
        self.calls = []

    def generate(self, text, language_id, audio_prompt_path):
        self.calls.append((text, language_id, audio_prompt_path))
        return np.zeros((1, 2400), dtype=np.float32)


def test_generation_is_resumable_and_uses_hindi(tmp_path):
    pytest.importorskip("torch")
    sf = pytest.importorskip("soundfile")
    jobs = cg.chatterbox_jobs(_table())
    model = _FakeModel()
    first = cg.generate_jobs(jobs, str(tmp_path / "pack"), str(tmp_path / "out"), model)
    assert len(first) == 2 and {c[1] for c in model.calls} == {"hi"}
    assert model.calls[0][2].replace("\\", "/").endswith("pack/refs/a.wav")
    audio, rate = sf.read(first[0])
    assert rate == 24_000 and len(audio) == 2400
    again = cg.generate_jobs(jobs, str(tmp_path / "pack"), str(tmp_path / "out"), model)
    assert again == [] and len(model.calls) == 2
    meta = (tmp_path / "out" / "generation_metadata.jsonl").read_text(encoding="utf-8")
    assert meta.count("\n") == 2 and "chatterbox" in meta


def test_a_job_for_another_tool_is_refused(tmp_path):
    pytest.importorskip("torch")
    with pytest.raises(AssertionError, match="Chatterbox driver"):
        cg.generate_jobs(_table(), str(tmp_path), str(tmp_path / "out"), _FakeModel())
