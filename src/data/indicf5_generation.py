"""IndicF5 clones: the second held-out attack, a modern flow-matching TTS for Hindi.

IndicF5 (AI4Bharat) is a zero-shot voice cloner trained on Indian-language speech. It
needs a reference clip *and that clip's transcript*, and it reads Devanagari, so the
job table (``src.data.extension_sets.indicf5_jobs``) pairs each speaker's all-Hindi
reference with all-Hindi sentences of other speakers. Like Tortoise it is
evaluation-only: it never enters a training manifest, which ``tests/test_splits.py``
and ``src.training.config_guard`` enforce through the ``tool`` column.

Heavy imports (transformers, the remote model code) are lazy so this module imports
cheaply in CI. The model is gated on Hugging Face, so loading needs a token.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

INDICF5_MODEL = "ai4bharat/IndicF5"
INDICF5_SR = 24_000
TOOL = "indicf5"


def load_indicf5(token: str | None = None, device: str = "cuda"):
    """Load IndicF5 through ``transformers`` remote code. Gated before the download."""
    from src.data.ethics_gate import require_signoff

    require_signoff(action="loading IndicF5")
    from transformers import AutoModel

    model = AutoModel.from_pretrained(INDICF5_MODEL, trust_remote_code=True, token=token)
    return model.to(device) if hasattr(model, "to") else model


def to_float(audio) -> object:
    """IndicF5 may return int16 or float audio; always hand back float32 in [-1, 1]."""
    import numpy as np

    audio = np.asarray(audio)
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    return audio.astype(np.float32)


def generate_jobs(jobs: pd.DataFrame, pack_dir: str, out_dir: str, model) -> list[str]:
    """Generate every pending job; one metadata line per clip, failures logged, resumable."""
    import soundfile as sf

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = out / "generation_metadata.jsonl"
    failures = out / "generation_failures.jsonl"
    written = []
    for job in jobs.itertuples(index=False):
        if str(job.tool) != TOOL:
            raise AssertionError(f"IndicF5 driver given a {job.tool!r} job")
        target = out / Path(str(job.output_path)).name
        if target.exists():
            continue
        try:
            audio = model(
                str(job.transcript),
                ref_audio_path=str(Path(pack_dir) / str(job.reference_wav)),
                ref_text=str(job.reference_text),
            )
            sf.write(str(target), to_float(audio), INDICF5_SR)
            written.append(str(target))
            record = {
                "output_path": target.name,
                "tool": TOOL,
                "speaker": str(job.speaker),
                "transcript": str(job.transcript),
                "language": "hi",
                "pool": "eval",
                "seed": int(job.seed),
            }
            with meta.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: BLE001 - one sentence must not stop the run
            with failures.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"job_id": int(job.job_id), "error": str(exc)[:300]}) + "\n")
    return written
