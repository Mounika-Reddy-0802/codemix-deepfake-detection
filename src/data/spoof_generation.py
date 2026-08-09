"""Voice-cloning drivers (XTTS-v2 / Tortoise) + speaker selection + metadata logging.

Week 2 (SK): set up XTTS-v2, select adult reference speakers from MUCS/HiACC-adult,
and generate pilot clones. Week 3 (L): run generation at scale; (SK) add the
held-out Tortoise tool for the unseen-attack split.

Golden-rule guards baked in here:
- **child audio is never a cloning reference** (excluded in ``select_reference_speakers``),
- each clone records the **tool** so the held-out Tortoise set can be firewalled
  from training manifests (``tests/test_splits.py``),
- each clone is tagged with its speaker **pool** (eval vs Stage-3 adaptation), which
  come from ``build_manifests.carve_pools`` so speaker-disjointness precedes generation.

Heavy imports (``TTS``/torch) are lazy so this module imports cheaply (CI-safe).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
TRAINING_TOOL = "xtts_v2"  # seen attack (may enter training)
HELD_OUT_TOOL = "tortoise"  # unseen attack (never in training)


# --------------------------------------------------------------------------- #
# Reference-speaker selection (pure pandas, testable)
# --------------------------------------------------------------------------- #
def select_reference_speakers(
    clips: pd.DataFrame,
    min_total_seconds: float = 30.0,
    min_clip_seconds: float = 3.0,
    n_min: int = 30,
    n_max: int = 50,
) -> list[str]:
    """Pick adult speakers with enough clean reference audio for cloning.

    ``clips`` needs columns ``speaker``, ``source``, ``duration``. Child speakers
    are excluded defensively (never a cloning reference). A speaker qualifies with
    at least ``min_total_seconds`` of clips each >= ``min_clip_seconds``. Returns
    up to ``n_max`` speakers, longest-reference first.
    """
    df = clips.copy()
    df = df[~df["speaker"].astype(str).str.lower().str.contains("child")]
    if "source" in df.columns:
        df = df[~df["source"].astype(str).str.lower().str.contains("child")]
    df = df[df["duration"] >= min_clip_seconds]

    totals = df.groupby("speaker")["duration"].agg(total="sum", clips="count")
    eligible = totals[totals["total"] >= min_total_seconds].sort_values("total", ascending=False)
    return list(eligible.index[:n_max])


def enough_speakers(selected: list[str], n_min: int = 30) -> bool:
    """True when selection meets the minimum speaker count for the spoof set."""
    return len(selected) >= n_min


# --------------------------------------------------------------------------- #
# Generation metadata (testable)
# --------------------------------------------------------------------------- #
@dataclass
class GenerationRecord:
    """Provenance for one generated clone (one JSON line per file)."""

    output_path: str
    tool: str
    speaker: str
    reference_wav: str
    transcript: str
    language: str
    pool: str  # "eval" | "adaptation"
    seed: int
    settings: dict = field(default_factory=dict)


def append_metadata(record: GenerationRecord, jsonl_path: str) -> None:
    """Append a generation record as one JSON line."""
    path = Path(jsonl_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def read_metadata(jsonl_path: str) -> list[dict]:
    """Read all generation records from a JSONL file."""
    path = Path(jsonl_path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# --------------------------------------------------------------------------- #
# Clone jobs + generation (heavy imports lazy)
# --------------------------------------------------------------------------- #
@dataclass
class CloneJob:
    """One clone to synthesise."""

    speaker: str
    reference_wav: str
    transcript: str
    output_path: str
    pool: str
    language: str = "hi"
    tool: str = TRAINING_TOOL
    seed: int = 0


def load_xtts(model_name: str = XTTS_MODEL, use_gpu: bool = True):
    """Load a Coqui XTTS-v2 model (lazy ``TTS`` import)."""
    from TTS.api import TTS

    return TTS(model_name, gpu=use_gpu)


def generate_clone(model, job: CloneJob) -> str:
    """Synthesise one clone to ``job.output_path`` and return the path."""
    Path(job.output_path).parent.mkdir(parents=True, exist_ok=True)
    model.tts_to_file(
        text=job.transcript,
        speaker_wav=job.reference_wav,
        language=job.language,
        file_path=job.output_path,
    )
    return job.output_path


def generate_batch(jobs: list[CloneJob], model, metadata_path: str) -> list[str]:
    """Generate every job, logging one metadata record per successful clone."""
    written: list[str] = []
    for job in jobs:
        if job.tool == HELD_OUT_TOOL:
            # Sanity: held-out tool clones must be tagged eval-only downstream.
            assert job.pool == "eval", "held-out tool clones must be eval-only"
        out = generate_clone(model, job)
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
                settings={"model": XTTS_MODEL},
            ),
            metadata_path,
        )
        written.append(out)
    return written


# --------------------------------------------------------------------------- #
# Generation at scale (Week 3, L) -- job assembly + stats (pure logic, testable)
# --------------------------------------------------------------------------- #
def build_clone_jobs(
    speaker_refs: dict[str, str],
    transcripts: list[tuple[str, str]],
    pools: dict[str, str],
    out_dir: str,
    tool: str = TRAINING_TOOL,
    language: str = "hi",
    n_target: int | None = None,
) -> list[CloneJob]:
    """Pair code-mixed transcripts with speaker references into clone jobs.

    ``speaker_refs``: speaker -> reference wav; ``transcripts``: (speaker, text)
    pairs; ``pools``: speaker -> "eval"|"adaptation". A transcript is skipped if its
    speaker has no reference or no pool assignment. Caps at ``n_target`` jobs.
    """
    jobs: list[CloneJob] = []
    for i, (speaker, text) in enumerate(transcripts):
        if speaker not in speaker_refs or speaker not in pools:
            continue
        out = str(Path(out_dir) / f"{tool}_{speaker}_{i:05d}.wav")
        jobs.append(
            CloneJob(
                speaker=speaker,
                reference_wav=speaker_refs[speaker],
                transcript=text,
                output_path=out,
                pool=pools[speaker],
                language=language,
                tool=tool,
                seed=i,
            )
        )
        if n_target is not None and len(jobs) >= n_target:
            break
    return jobs


def generation_stats(records: list[dict]) -> dict:
    """Summarise generation metadata for the audit report (reviewers ask for this)."""
    from collections import Counter

    tools = Counter(r["tool"] for r in records)
    langs = Counter(r["language"] for r in records)
    pools = Counter(r["pool"] for r in records)
    per_speaker = Counter(r["speaker"] for r in records)
    return {
        "total": len(records),
        "by_tool": dict(tools),
        "by_language": dict(langs),
        "by_pool": dict(pools),
        "n_speakers": len(per_speaker),
        "per_speaker_min": min(per_speaker.values()) if per_speaker else 0,
        "per_speaker_max": max(per_speaker.values()) if per_speaker else 0,
    }
