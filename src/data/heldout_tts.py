"""Held-out Tortoise-TTS driver for the unseen-attack split (Week 3, owner SK).

Tortoise is the **eval-only held-out** tool: its clones represent an attack the
detector has never seen in training. They must NEVER enter a training manifest
(enforced by ``tests/test_splits.py``). This module reuses the ``CloneJob`` /
``GenerationRecord`` contract from ``spoof_generation`` and hard-guards both
``tool == tortoise`` and ``pool == eval`` before any file is written. Heavy imports
(tortoise / torchaudio) are lazy so the module imports cheaply (CI-safe).
"""

from __future__ import annotations

from pathlib import Path

from src.data.spoof_generation import (
    HELD_OUT_TOOL,
    CloneJob,
    GenerationRecord,
    append_metadata,
)

TORTOISE_SR = 24_000


def build_heldout_jobs(
    speaker_refs: dict[str, str],
    transcripts: list[tuple[str, str]],
    out_dir: str,
    language: str = "hi",
    n_target: int | None = None,
) -> list[CloneJob]:
    """Build Tortoise clone jobs. Every job is ``tool=tortoise`` and ``pool='eval'``."""
    jobs: list[CloneJob] = []
    for i, (speaker, text) in enumerate(transcripts):
        if speaker not in speaker_refs:
            continue
        out = str(Path(out_dir) / f"tortoise_{speaker}_{i:05d}.wav")
        jobs.append(
            CloneJob(
                speaker=speaker,
                reference_wav=speaker_refs[speaker],
                transcript=text,
                output_path=out,
                pool="eval",  # held-out is always eval-only
                language=language,
                tool=HELD_OUT_TOOL,
                seed=i,
            )
        )
        if n_target is not None and len(jobs) >= n_target:
            break
    return jobs


def _assert_heldout(job: CloneJob) -> None:
    """Firewall: refuse to generate anything that isn't the eval-only held-out tool."""
    assert job.tool == HELD_OUT_TOOL, "heldout driver only generates the held-out tool"
    assert job.pool == "eval", "held-out clones must be eval-only (never in training)"


def load_tortoise():
    """Load Tortoise-TTS (lazy import)."""
    from tortoise.api import TextToSpeech

    return TextToSpeech()


def _load_voice_samples(reference_wav: str) -> list:
    """Load a reference clip as Tortoise voice samples (lazy torchaudio)."""
    import torchaudio

    wav, sr = torchaudio.load(reference_wav)
    if sr != TORTOISE_SR:
        wav = torchaudio.functional.resample(wav, sr, TORTOISE_SR)
    return [wav]


def generate_heldout_clone(model, job: CloneJob) -> str:
    """Synthesise one Tortoise clone to ``job.output_path`` (guards tool + pool)."""
    _assert_heldout(job)
    import torchaudio

    Path(job.output_path).parent.mkdir(parents=True, exist_ok=True)
    voice = _load_voice_samples(job.reference_wav)
    audio = model.tts_with_preset(job.transcript, voice_samples=voice, preset="fast")
    torchaudio.save(job.output_path, audio.squeeze(0).cpu(), TORTOISE_SR)
    return job.output_path


def generate_heldout_batch(jobs: list[CloneJob], model, metadata_path: str) -> list[str]:
    """Generate the unseen-attack split, logging one metadata record per clone."""
    written: list[str] = []
    for job in jobs:
        _assert_heldout(job)
        out = generate_heldout_clone(model, job)
        append_metadata(
            GenerationRecord(
                output_path=out,
                tool=job.tool,
                speaker=job.speaker,
                reference_wav=job.reference_wav,
                transcript=job.transcript,
                language=job.language,
                pool=job.pool,
                seed=job.seed,
                settings={"model": "tortoise-tts", "preset": "fast"},
            ),
            metadata_path,
        )
        written.append(out)
    return written
