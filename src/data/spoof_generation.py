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


def reconcile_metadata(records: list[dict], require_file: bool = True) -> list[dict]:
    """Reduce an append-only metadata log to one record per file that exists.

    The log is append-only, so regenerating a clip -- deleting the wav and running
    again -- leaves the superseded record in place next to the new one. That is
    how the pilot ended up with 25 records for 20 files, five of them describing
    transcripts that no longer exist on disk. Since this file is the provenance
    the dataset datasheet is built from, a stale record is worse than a missing
    one: it misreports what the corpus contains.

    **Last record wins**, because the log is chronological and the newest write
    describes the file currently on disk. Records whose file is gone are dropped
    unless ``require_file`` is False.
    """
    latest: dict[str, dict] = {}
    for record in records:  # later entries overwrite earlier ones
        latest[str(record.get("output_path", ""))] = record
    if not require_file:
        return list(latest.values())
    return [r for path, r in latest.items() if path and Path(path).is_file()]


def rewrite_metadata(jsonl_path: str, require_file: bool = True) -> int:
    """Rewrite a metadata log in place, reconciled. Returns the record count."""
    records = reconcile_metadata(read_metadata(jsonl_path), require_file=require_file)
    path = Path(jsonl_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


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


def load_xtts(
    model_name: str = XTTS_MODEL,
    use_gpu: bool | None = None,
    device: str | None = None,
):
    """Load a Coqui XTTS-v2 model (lazy ``TTS`` import).

    ``use_gpu=None`` and ``device=None`` detect the device, so the same call works
    in a Colab GPU session and in a CPU run on the dev laptop.

    Two packages provide this model — the original ``TTS`` (Python <= 3.11) and the
    maintained ``coqui-tts`` fork (current Python). The fork moved device selection
    from a ``gpu=`` constructor argument to ``.to(device)``, so both spellings are
    handled rather than pinning the project to one of them.

    Gated: the ethics check runs before the model is fetched, so a blocked run
    costs nothing and fails with a readable reason instead of a CUDA trace.
    """
    from src.data.ethics_gate import require_signoff
    from src.utils.device import CPU, CUDA, is_cuda, resolve_device

    require_signoff(action="loading the XTTS-v2 model")
    if device is None:
        device = resolve_device() if use_gpu is None else (CUDA if use_gpu else CPU)

    from TTS.api import TTS

    try:
        model = TTS(model_name)
    except TypeError:  # very old TTS: device is a constructor argument
        return TTS(model_name, gpu=is_cuda(device))
    return model.to(device) if hasattr(model, "to") else model


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


def pending_jobs(jobs: list[CloneJob]) -> list[CloneJob]:
    """The jobs whose output does not exist yet.

    Generation is resumable by design (W4-T1): a CPU pilot run takes tens of
    minutes and a Colab session can be reclaimed mid-batch, so a re-run must
    continue rather than start over and duplicate metadata records.
    """
    return [job for job in jobs if not Path(job.output_path).is_file()]


def generate_batch(
    jobs: list[CloneJob],
    model,
    metadata_path: str,
    skip_existing: bool = True,
    on_progress=None,
    continue_on_error: bool = True,
    failures: list[dict] | None = None,
) -> list[str]:
    """Generate every job, logging one metadata record per successful clone.

    ``skip_existing`` makes the batch resumable. ``on_progress`` is called as
    ``on_progress(index, total, job)`` before each clone, so a long CPU run is not
    silent.

    ``continue_on_error`` keeps one unsynthesisable transcript from killing the
    whole run. That is not defensive padding: XTTS raises ``NotImplementedError``
    on any Hindi transcript containing a digit (num2words has no Hindi support),
    and MUCS is lecture speech full of numbers. Losing 4,000 clips to clip 7 is
    the failure mode this prevents. Failed jobs are appended to ``failures`` and
    reported by the caller, never silently dropped -- the failure rate is a number
    the dataset datasheet has to state.
    """
    from src.data.ethics_gate import require_signoff

    require_signoff(action="XTTS-v2 clone generation")
    todo = pending_jobs(jobs) if skip_existing else list(jobs)
    written: list[str] = []
    for i, job in enumerate(todo, 1):
        if job.tool == HELD_OUT_TOOL:
            # Sanity: held-out tool clones must be tagged eval-only downstream.
            assert job.pool == "eval", "held-out tool clones must be eval-only"
        if on_progress is not None:
            on_progress(i, len(todo), job)
        try:
            out = generate_clone(model, job)
        except Exception as exc:  # noqa: BLE001 - one bad clip must not end the run
            record = {
                "output_path": job.output_path,
                "speaker": job.speaker,
                "language": job.language,
                "transcript": job.transcript,
                "error": f"{type(exc).__name__}: {exc}",
            }
            if failures is not None:
                failures.append(record)
            print(f"    [FAILED] {Path(job.output_path).name}: {record['error']}")
            if not continue_on_error:
                raise
            continue
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


def run_jobs_file(
    jobs_csv: str,
    pack_dir: str,
    out_dir: str,
    metadata_path: str | None = None,
    limit: int | None = None,
    device: str | None = None,
    expect_pool: str | None = "train",
) -> list[str]:
    """Generate every clone in a job table. The one entry point both demos share.

    ``expect_pool`` is the speaker-pool firewall: the pilot and the Week-4 seen
    attack clone **train-pool** speakers only, and a reference drawn from the
    adaptation or eval pool would put a fake of an evaluation voice into training
    data and void the gap matrix. Checked before the model is loaded, so a
    mistake costs a second rather than an hour.
    """
    from src.data.pilot_jobs import load_pilot_jobs

    table = pd.read_csv(jobs_csv)
    if expect_pool is not None:
        pools = set(table["pool"].astype(str))
        if pools != {expect_pool}:
            raise AssertionError(
                f"FIREWALL BREACH: job table has pools {sorted(pools)}, expected "
                f"only {{{expect_pool!r}}}. Refusing to generate."
            )

    jobs = load_pilot_jobs(jobs_csv, pack_dir, out_dir=out_dir)
    if limit is not None:
        jobs = jobs[:limit]
    todo = pending_jobs(jobs)
    print(f"{len(jobs)} job(s); {len(jobs) - len(todo)} already generated; {len(todo)} to go")
    if not todo:
        return []

    model = load_xtts(device=device)

    def progress(i: int, total: int, job: CloneJob) -> None:
        print(f"  [{i}/{total}] {job.speaker} lang={job.language} -> {Path(job.output_path).name}")

    metadata = metadata_path or str(Path(out_dir) / "generation_metadata.jsonl")
    failures: list[dict] = []
    written = generate_batch(
        jobs, model, metadata_path=metadata, on_progress=progress, failures=failures
    )
    if failures:
        # Logged as a file, not just printed: the datasheet must report how many
        # transcripts the generator could not say, and why.
        failure_log = Path(out_dir) / "generation_failures.jsonl"
        with failure_log.open("a", encoding="utf-8") as fh:
            for record in failures:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"\n{len(failures)} job(s) failed -> {failure_log}")
    return written


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


def main() -> None:
    """CLI: ``python -m src.data.spoof_generation --jobs JOBS.csv --pack-dir PACK``.

    Runs anywhere — CPU laptop or Colab GPU — because the device is detected
    (``--device`` / ``$DFD_DEVICE`` override) and the run is resumable.
    """
    import argparse
    import json
    import os

    from src.data.ethics_gate import signoff_status
    from src.utils.device import describe, resolve_device
    from src.utils.paths import data_root

    parser = argparse.ArgumentParser(description="Generate voice clones from a job table")
    parser.add_argument("--jobs", required=True, help="generation_jobs.csv from the pack")
    parser.add_argument("--pack-dir", required=True, help="pack root holding refs/")
    parser.add_argument("--out-dir", default=None, help="default: <pack-dir>/outputs")
    parser.add_argument(
        "--metadata", default=None, help="default: <out-dir>/generation_metadata.jsonl"
    )
    parser.add_argument("--limit", type=int, default=None, help="generate only the first N jobs")
    parser.add_argument("--device", default=None, help="auto | cpu | cuda[:N]")
    parser.add_argument(
        "--expect-pool",
        default="train",
        help="required speaker pool for every job ('none' disables the check)",
    )
    args = parser.parse_args()

    status = signoff_status()
    print(status.describe())
    if not status.signed:
        raise SystemExit(1)

    # XTTS-v2 ships under the Coqui Public Model Licence and prompts interactively
    # without this. Non-commercial research use is exactly what the signed mentor
    # note covers; the same line is in notebooks/pilot_xtts_colab.ipynb.
    os.environ.setdefault("COQUI_TOS_AGREED", "1")
    print("COQUI_TOS_AGREED=1 (CPML, non-commercial research use per the signed note)")
    print(describe(resolve_device(args.device)))

    out_dir = args.out_dir or str(Path(args.pack_dir) / "outputs")
    written = run_jobs_file(
        jobs_csv=args.jobs,
        pack_dir=args.pack_dir,
        out_dir=out_dir,
        metadata_path=args.metadata,
        limit=args.limit,
        device=args.device,
        expect_pool=None if args.expect_pool.lower() == "none" else args.expect_pool,
    )
    print(f"\ngenerated {len(written)} clip(s) -> {out_dir}")

    metadata = args.metadata or str(Path(out_dir) / "generation_metadata.jsonl")
    # Reconcile before reporting: a resumed or regenerated run leaves superseded
    # records in the append-only log, and the stats below feed the datasheet.
    kept = rewrite_metadata(metadata)
    records = read_metadata(metadata)
    print(f"metadata reconciled: {kept} record(s) matching files on disk")
    if records:
        print(json.dumps(generation_stats(records), indent=2, ensure_ascii=False))
    print(f"(DATA_ROOT={data_root()})")


if __name__ == "__main__":
    main()
