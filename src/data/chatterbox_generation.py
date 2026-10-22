"""Chatterbox clones: the third held-out attack, a recent multilingual voice cloner.

Chatterbox Multilingual (Resemble AI, MIT licence) clones a voice from a short reference
clip and speaks Hindi (``language_id="hi"``) from Devanagari text. It is a different
model family from the three earlier tools (XTTS-v2 is an autoregressive GPT decoder,
Tortoise autoregressive plus diffusion, IndicF5 flow matching), which makes it the most
independent test of whether a detector generalises across generators.

It uses the same job table as IndicF5 (same speakers, same all-Hindi sentences), so
CM09 pairs clip for clip with CM07. Like Tortoise and IndicF5 it is evaluation-only:
it never enters a training manifest, which ``tests/test_splits.py`` and
``src.training.config_guard`` enforce through the ``tool`` column.

One property to report: Chatterbox embeds an imperceptible watermark (Resemble's Perth)
in everything it generates. We keep its output exactly as produced; the watermark is
not something our detectors were trained on, and it is a possible shortcut that a
detector could in principle use, which is why the paper says so.

Heavy imports are lazy so the module imports cheaply in CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

TOOL = "chatterbox"
LANGUAGE = "hi"


def load_chatterbox(device: str = "cuda"):
    """Load Chatterbox Multilingual. Gated before the model is fetched."""
    from src.data.ethics_gate import require_signoff

    require_signoff(action="loading Chatterbox")
    import os

    # Chatterbox passes HF_TOKEN straight to the download; an empty value is sent as
    # "Bearer " and refused, so a blank token is removed (the model is public).
    if not os.environ.get("HF_TOKEN", "").strip():
        os.environ.pop("HF_TOKEN", None)
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    return ChatterboxMultilingualTTS.from_pretrained(device=device)


def to_numpy(audio) -> object:
    """Chatterbox returns a (1, samples) float tensor; hand back 1-D float32."""
    import numpy as np

    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def generate_jobs(jobs: pd.DataFrame, pack_dir: str, out_dir: str, model) -> list[str]:
    """Generate every pending job; one metadata line per clip, failures logged, resumable."""
    import soundfile as sf
    import torch

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = out / "generation_metadata.jsonl"
    failures = out / "generation_failures.jsonl"
    sample_rate = int(getattr(model, "sr", 24_000))
    written = []
    for job in jobs.itertuples(index=False):
        if str(job.tool) != TOOL:
            raise AssertionError(f"Chatterbox driver given a {job.tool!r} job")
        target = out / Path(str(job.output_path)).name
        if target.exists():
            continue
        try:
            torch.manual_seed(int(job.seed))
            audio = model.generate(
                str(job.transcript),
                language_id=LANGUAGE,
                audio_prompt_path=str(Path(pack_dir) / str(job.reference_wav)),
            )
            sf.write(str(target), to_numpy(audio), sample_rate)
            written.append(str(target))
            record = {
                "output_path": target.name,
                "tool": TOOL,
                "speaker": str(job.speaker),
                "transcript": str(job.transcript),
                "language": LANGUAGE,
                "seed": int(job.seed),
            }
            with meta.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: BLE001 - one sentence must not stop the run
            with failures.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"job_id": int(job.job_id), "error": str(exc)[:300]}) + "\n")
    return written


def chatterbox_jobs(indicf5_table: pd.DataFrame) -> pd.DataFrame:
    """The IndicF5 job table re-labelled for Chatterbox (same speakers and sentences)."""
    out = indicf5_table.copy()
    out["tool"] = TOOL
    out["output_path"] = out["output_path"].map(
        lambda p: Path(str(p)).name.replace("indicf5", TOOL, 1)
    )
    if out["output_path"].duplicated().any():
        raise ValueError("output names are not unique")
    return out


def main() -> None:
    """Write the Chatterbox job table, and optionally a pack (table + reference clips)."""
    import argparse
    import shutil

    parser = argparse.ArgumentParser(description="Build the Chatterbox job table and pack")
    parser.add_argument("--indicf5-jobs", default="data/manifests/indicf5_generation_jobs.csv")
    parser.add_argument("--indicf5-pack", default=None, help="IndicF5 pack folder with refs/")
    parser.add_argument("--out", default="data/manifests/chatterbox_generation_jobs.csv")
    parser.add_argument("--pack-out", default=None, help="folder to build the pack in")
    args = parser.parse_args()

    table = chatterbox_jobs(pd.read_csv(args.indicf5_jobs))
    table.to_csv(args.out, index=False)
    print(f"{args.out}: {len(table)} jobs, {table['speaker'].nunique()} speakers")
    if args.pack_out:
        if not args.indicf5_pack:
            raise SystemExit("--pack-out needs --indicf5-pack (the reference clips)")
        pack = Path(args.pack_out)
        (pack / "refs").mkdir(parents=True, exist_ok=True)
        for name in table["reference_wav"].map(lambda p: Path(str(p)).name).unique():
            shutil.copy2(Path(args.indicf5_pack) / "refs" / name, pack / "refs" / name)
        table.to_csv(pack / "generation_jobs.csv", index=False)
        print(f"pack: {pack} with {len(list((pack / 'refs').iterdir()))} reference clips")


if __name__ == "__main__":
    main()
