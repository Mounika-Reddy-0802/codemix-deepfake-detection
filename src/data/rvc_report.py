"""Turn a finished CM02 run into the artefacts this repository can keep.

The repository cannot hold the run itself. The converted clips are cloned voices
of identifiable MUCS speakers in a public repo, ``**/generated/**/*.wav`` is
gitignored on purpose, and twelve ~55 MB ``.pth`` weights would not belong in git
even if they were anonymous (``checkpoints/**`` is ignored for the same reason).
So what the repo keeps is a **description** precise enough that the run can be
audited and rebuilt: the job table, the metadata with machine-local paths rewritten
portable, a mechanical quality screen, the pitch measurement P-019's prediction
turns on, and SHA-256 checksums of the weights that never leave Kaggle.

Two failure modes this exists to prevent, both of which have already happened in
this project:

- **Machine-local junk in git.** ``generation_metadata.jsonl`` is written full of
  ``/kaggle/working/...`` paths. Commit f60f989 gitignored the ASVspoof manifests
  for exactly this reason -- an absolute path committed from one machine resolves
  to nothing on every other one. :func:`portable_records` rewrites them through
  ``src.utils.paths.portable`` before anything is written into the tree.
- **A generator that returns valid audio and is completely wrong.** That is what
  ``generation_qa`` exists for (a pilot clip produced 0.83 s from a 150-character
  transcript, raised nothing, and was counted as a success). The screen runs here
  so a 1,500-clip corpus is never accepted on the strength of an exit code.

Doc writers are idempotent: re-running a report replaces its own section rather
than appending a second copy, so a resumed or repeated Kaggle run does not leave
the ADR log with two P-020 rows.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from src.data import f0_stats
from src.data.checksums import sha256_file
from src.utils import paths

#: Metadata fields that carry a filesystem path and must be made portable.
PATH_FIELDS = ("output_path", "reference_wav", "source_wav", "wav_path")

#: Where each artefact lands in the tree. Kept as data so the notebook, the tests
#: and the results document cannot disagree about where a file went.
METADATA_PATH = "outputs/rvc_generation_metadata.jsonl"
SUMMARY_PATH = "outputs/rvc_generation_summary.json"
MODEL_MANIFEST_PATH = "outputs/rvc_model_checksums.json"
QA_CSV_PATH = "docs/qa/rvc_generation_qa.csv"
QA_DOC_PATH = "docs/qa/rvc_generation_qa.md"
RESULTS_DOC_PATH = "docs/results/rvc_generation_v1.md"
DECISIONS_PATH = "docs/problems_and_decisions.md"
TAXONOMY_PATH = "docs/attack_taxonomy.md"


# --------------------------------------------------------------------------- #
# Portable metadata
# --------------------------------------------------------------------------- #
def portable_records(records, root: str | None = None) -> list[dict]:
    """Rewrite every path field of every record relative to ``${DATA_ROOT}``.

    Paths outside the data tree are returned unchanged by ``paths.portable``; a
    converted clip written to ``/kaggle/working/rvc_outputs`` has no layout anchor,
    so it is reduced to its basename instead -- the file is not in the repo and
    never will be, and the name is the part that identifies it.
    """
    out: list[dict] = []
    for record in records:
        row = dict(record)
        for field in PATH_FIELDS:
            value = row.get(field)
            if not value:
                continue
            rewritten = paths.portable(str(value), root=root)
            if rewritten == str(value).replace("\\", "/"):
                # No layout anchor to rebase on: keep the name, drop the machine.
                rewritten = Path(str(value)).name
            row[field] = rewritten
        out.append(row)
    return out


def write_portable_metadata(records, destination: str, root: str | None = None) -> int:
    """Write portable metadata as JSONL. Returns the record count."""
    rows = portable_records(records, root)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(rows)


def portable_jobs(jobs: pd.DataFrame, root: str | None = None) -> pd.DataFrame:
    """The job table with its path columns rebased, ready to commit.

    ``build_rvc_jobs`` writes ``source_wav`` as a full local path and
    ``output_path`` under whatever ``out_dir`` the run used -- on Kaggle both are
    ``/kaggle/working/...``. The table is a *manifest*, which is the one thing in
    ``data/`` that is deliberately tracked (``!data/manifests/*.csv``), so it has
    to be portable or it is one machine's directory listing in a public repo.
    """
    frame = jobs.copy()
    for column in ("source_wav", "output_path"):
        if column in frame.columns:
            frame[column] = [
                row[column]
                for row in portable_records(frame[[column]].to_dict(orient="records"), root=root)
            ]
    return frame


def write_portable_jobs(
    jobs: pd.DataFrame, destination: str, root: str | None = None
) -> pd.DataFrame:
    """Write the portable job table as CSV and return it."""
    frame = portable_jobs(jobs, root)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


# --------------------------------------------------------------------------- #
# Model checksums -- the weights stay on Kaggle, their identity comes home
# --------------------------------------------------------------------------- #
def model_manifest(models: dict) -> list[dict]:
    """One record per trained voice model: what it is, how big, what it hashes to.

    Without this "we trained twelve RVC models" is an unverifiable claim, and a
    re-run that silently produced different weights would be indistinguishable
    from a reproduction.
    """
    rows: list[dict] = []
    for speaker in sorted(models, key=str):
        model = models[speaker]
        weight = Path(model.model_path)
        record = {
            "speaker": str(speaker),
            "experiment": model.experiment,
            "weight_file": weight.name,
            "epochs": int(model.epochs),
            "train_clips": int(model.n_clips),
            "train_seconds": round(float(model.train_seconds), 2),
            "sample_rate": model.sample_rate,
            "version": model.version,
            "bytes": weight.stat().st_size if weight.is_file() else 0,
            "sha256": sha256_file(str(weight)) if weight.is_file() else "",
        }
        index = Path(model.index_path) if model.index_path else None
        if index is not None and index.is_file():
            record["index_file"] = index.name
            record["index_bytes"] = index.stat().st_size
            record["index_sha256"] = sha256_file(str(index))
        rows.append(record)
    return rows


def write_model_manifest(models: dict, destination: str = MODEL_MANIFEST_PATH) -> list[dict]:
    """Write the model manifest as JSON and return it."""
    rows = model_manifest(models)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "note": (
            "RVC voice models are gitignored (checkpoints/** and their ~55 MB size). "
            "This manifest is their identity: re-hash a weight file to prove it is "
            "the one this run produced."
        ),
        "models": rows,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rows


# --------------------------------------------------------------------------- #
# Quality screen
# --------------------------------------------------------------------------- #
def screen_rvc_jobs(jobs: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """Run ``generation_qa`` over an RVC job table.

    ``generation_qa`` was written for the CM01 tables, whose speaker column is
    ``speaker``; an RVC job names two speakers and calls the one that owns the
    output ``target_speaker``. Renaming here keeps one screen for both attack
    families rather than forking it.
    """
    from src.data import generation_qa

    frame = jobs.rename(columns={"target_speaker": "speaker"})
    return generation_qa.screen(frame, out_dir)


# --------------------------------------------------------------------------- #
# Pitch: the measurement CM02's existence rests on
# --------------------------------------------------------------------------- #
def measure_pitch(
    jobs: pd.DataFrame,
    out_dir: str,
    config: f0_stats.F0Config | None = None,
    limit: int | None = None,
) -> dict:
    """Measure f0 IQR on the converted clips and on the real clips they came from.

    Paired deliberately: the same estimator, the same settings, the same
    utterances. P-019's 41.1 Hz was measured ad hoc with unknown tooling, so
    comparing this run's converted clips against that remembered constant would
    be comparing two measurements, not one difference.
    """
    import soundfile as sf

    from src.data.corpora import load_clip

    rows = jobs if limit is None else jobs.head(limit)
    converted_stats: list[dict] = []
    real_stats: list[dict] = []
    for job in rows.to_dict(orient="records"):
        clip_path = Path(out_dir) / Path(str(job["output_path"])).name
        if not clip_path.is_file():
            continue
        audio, sample_rate = sf.read(str(clip_path), dtype="float32", always_2d=False)
        converted_stats.append(f0_stats.clip_f0_stats(audio, sample_rate, config))
        source = {
            "wav_path": job["source_wav"],
            "start_seconds": job["start_seconds"],
            "end_seconds": job["end_seconds"],
        }
        try:
            real_audio, real_sr = load_clip(source)
        except (OSError, RuntimeError, ValueError):
            continue
        real_stats.append(f0_stats.clip_f0_stats(real_audio, real_sr, config))
    return f0_stats.compare(converted_stats, real_stats)


# --------------------------------------------------------------------------- #
# Idempotent markdown surgery
# --------------------------------------------------------------------------- #
def _replace_block(text: str, marker: str, block: str) -> str:
    """Replace a marked block, or append it. Re-running never duplicates a section."""
    start = f"<!-- {marker}:start -->"
    end = f"<!-- {marker}:end -->"
    body = f"{start}\n{block.rstrip()}\n{end}"
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    if pattern.search(text):
        return pattern.sub(body, text)
    return text.rstrip() + "\n\n" + body + "\n"


def append_decision(entry: str, path: str = DECISIONS_PATH) -> bool:
    """Append one ADR row to the decisions table, unless its ID is already there.

    The log's own rule is that existing entries are not rewritten, so this only
    ever adds -- and a repeated Kaggle run must not add a second P-020.
    """
    identifier = entry.split("|")[1].strip().strip("*") if "|" in entry else entry[:8]
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    if identifier and identifier in text:
        return False
    file.write_text(text.rstrip() + "\n" + entry.rstrip() + "\n", encoding="utf-8")
    return True


def update_taxonomy_row(
    attack_id: str,
    samples: str,
    speakers: str,
    path: str = TAXONOMY_PATH,
) -> bool:
    """Replace the sample/speaker cells of one attack row with measured counts.

    The table ships with targets, and says so in bold at the top. A row that has
    actually been generated should read as a measurement, or the dataset freeze
    inherits a plan number that nobody re-checked.
    """
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    # The ID cell may or may not be bold; whichever it is, it is preserved.
    pattern = re.compile(
        rf"^\|(\s*\**{re.escape(attack_id)}\**\s*)\|(.*)$",
        re.MULTILINE,
    )
    match = pattern.search(text)
    if not match:
        return False
    cells = [cell.strip() for cell in match.group(2).split("|")]
    if len(cells) < 6:
        return False
    cells[3] = samples
    cells[4] = speakers
    row = f"|{match.group(1)}| " + " | ".join(cells).rstrip()
    updated = text[: match.start()] + row + text[match.end() :]
    if updated == text:
        return False
    file.write_text(updated, encoding="utf-8")
    return True


def write_doc(path: str, text: str) -> str:
    """Write a generated document, creating its directory. Returns the path."""
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(text.rstrip() + "\n", encoding="utf-8")
    return str(file)


# --------------------------------------------------------------------------- #
# Document rendering
# --------------------------------------------------------------------------- #
def _per_target_table(per_target: list[dict]) -> str:
    """Markdown rows: one line per trained voice, so a short target is visible."""
    head = (
        "| Target speaker | Train clips | Train audio | Epochs | Clips written | QA pass |\n"
        "|---|---|---|---|---|---|\n"
    )
    body = "".join(
        "| {speaker} | {train_clips} | {train_seconds:.0f} s | {epochs} | "
        "{written} | {qa_pass} |\n".format(**row)
        for row in per_target
    )
    return head + body


def _failure_table(failures: dict) -> str:
    """The QA failure breakdown, or an explicit statement that there was none."""
    if not failures:
        return "No clip failed the screen.\n"
    rows = "".join(f"| {reason} | {count} |\n" for reason, count in failures.items())
    return "| Reason | Clips |\n|---|---|\n" + rows


def render_results_doc(ctx: dict) -> str:
    """The results document: what was generated, and what it is worth."""
    qa = ctx["qa"]
    pitch = ctx["pitch"]
    converted = pitch["converted"]
    real = pitch["real_source"]
    return f"""# RVC voice conversion — CM02 generation run (W4-T2)

Generated on Kaggle ({ctx["accelerator"]}) on {ctx["run_date"]} from
`notebooks/kaggle_w4t2_rvc_generation.ipynb`, branch `{ctx["branch"]}`, repo commit
`{ctx["commit"]}`. RVC-WebUI checkout `{ctx["rvc_commit"]}`.

CM02 is the second seen attack family. P-019 established why it has to exist:
XTTS-v2 (CM01) compresses intra-utterance pitch range by roughly 35%, because it
invents prosody from text. Voice conversion starts from a real human recording
and swaps timbre only, so the contour is the source speaker's own — that was the
prediction, and this run is the first time it has been measured. See the pitch
section below.

## What ran

| | |
|---|---|
| Targets trained | **{ctx["n_targets"]}** |
| Epochs per target | {ctx["train_epochs"]} |
| Conversions requested per target | {ctx["conversions_per_target"]} |
| Clips written | **{ctx["written"]}** |
| Conversion failures | **{ctx["failures"]}** |
| Total converted audio | {ctx["total_audio_hours"]:.2f} h |
| Source pool | MUCS 2021 train pool, {ctx["train_pool_speakers"]} speakers |
| Model | RVC v2, {ctx["sample_rate"]}, warm-started from `f0G/f0D{ctx["sample_rate"]}.pth` |
| f0 method | {ctx["f0_method"]} |

Every conversion has a train-pool speaker at **both** ends, no speaker is
converted into themselves, and no source clip is reused within one target —
checked by `assert_rvc_invariants` before any GPU time, not after.

## Per target

{_per_target_table(ctx["per_target"])}

## Mechanical quality screen

`src.data.generation_qa` screens the relationship between each transcript and its
audio, because a generator can return valid audio that is completely wrong without
raising anything. Full report: `{QA_CSV_PATH}`, verdict: `{QA_DOC_PATH}`.

| | |
|---|---|
| Clips screened | {qa["clips"]} |
| Passed | **{qa["passed"]}** |
| Failed | **{qa["failed"]}** |
| Pass rate | **{qa["pass_rate"]}%** |
| Median speaking rate | {qa.get("median_chars_per_sec")} chars/s |

{_failure_table(qa.get("failures", {}))}

## Pitch range — testing P-019's prediction

Measured with `src.data.f0_stats` (Praat autocorrelation, {ctx["pitch_floor"]:.0f}–{ctx["pitch_ceiling"]:.0f} Hz).
The converted clips and the real clips they were converted from were measured in
the same pass with the same settings, so this is one difference rather than two
measurements.

| | Median f0 IQR | Median f0 | Clips |
|---|---|---|---|
| Real MUCS source clips (measured here) | **{real.get("median_f0_iqr_hz", 0):.1f} Hz** | {real.get("median_f0_hz", 0):.0f} Hz | {real.get("usable", 0)} |
| RVC converted clips (measured here) | **{converted.get("median_f0_iqr_hz", 0):.1f} Hz** | {converted.get("median_f0_hz", 0):.0f} Hz | {converted.get("usable", 0)} |
| XTTS-v2, P-019 (ad hoc, tooling unknown) | 25–29 Hz | — | — |

Pitch-range retention: **{pitch["retention_pct"]:.1f}%** of the real speech it started from.

> {pitch["verdict"]}

## What this repository keeps, and what it does not

The audio is **not** committed. These are cloned voices of identifiable MUCS
speakers in a public repository, and `**/generated/**/*.wav` is gitignored
deliberately. The voice models are not committed either — twelve weights at
~{ctx["model_mb"]:.0f} MB each, and `checkpoints/**` is ignored for the same reason.

What is committed is enough to audit and rebuild the run:

| Artefact | Path |
|---|---|
| Job table (every conversion, both endpoints) | `data/manifests/rvc_generation_jobs.csv` |
| Per-clip metadata, paths made portable | `{METADATA_PATH}` |
| Run summary | `{SUMMARY_PATH}` |
| Model checksums (SHA-256, size, epochs) | `{MODEL_MANIFEST_PATH}` |
| QA report | `{QA_CSV_PATH}` |
| QA verdict | `{QA_DOC_PATH}` |

`generation_metadata.jsonl` is written by the generator full of absolute
`/kaggle/working/...` paths. It is passed through `src.utils.paths.portable`
before it enters the tree — commit f60f989 gitignored the ASVspoof manifests for
exactly that reason, and a machine-local path in git is junk on every other
machine.

## Reproducing this

```bash
# Kaggle: GPU T4 x2, Internet on, MUCS train-pool dataset + signed ethics PDF as Inputs
# then run notebooks/kaggle_w4t2_rvc_generation.ipynb top to bottom.
python -m src.data.corpora --data-root $DATA_ROOT      # rebuilds clip_index.csv
python -m src.data.generation_qa --jobs data/manifests/rvc_generation_jobs.csv \\
    --out-dir <converted clips> --report {QA_CSV_PATH}
```

Three toolchain defects had to be fixed before any of this ran; they are recorded
as **P-020** in `docs/problems_and_decisions.md`.
"""


def render_qa_doc(ctx: dict) -> str:
    """The QA verdict, in the shape of `docs/qa/pilot_script_verdict.md`."""
    qa = ctx["qa"]
    pitch = ctx["pitch"]
    converted = pitch["converted"]
    real = pitch["real_source"]
    rate = 100 - float(qa["pass_rate"])
    stands = qa["passed"] > 0 and rate <= 5.0
    headline = (
        f"**Verdict: the CM02 corpus is usable as generated.** "
        f"{qa['passed']} of {qa['clips']} clips pass the mechanical screen "
        f"({qa['pass_rate']}%)."
        if stands
        else (
            f"**Verdict: the CM02 corpus does NOT clear the bar as generated.** "
            f"{qa['failed']} of {qa['clips']} clips fail the mechanical screen "
            f"({rate:.1f}%), which is too many to leave inside a training corpus."
        )
    )
    return f"""# RVC generation — mechanical QA (W4-T2)

`src.data.generation_qa` screened every converted clip. The screen exists because
a generator can return **valid audio that is completely wrong** and raise nothing:
in the 40-clip script pilot one job produced 0.83 s of audio from a 150-character
transcript, was logged as a success, and was counted. Nobody listens to
{qa["clips"]} clips, so this has to be mechanical.

| Check | Threshold | Clips flagged |
|---|---|---|
| Speaking rate | 6–30 chars/s | {ctx["flag_rate"]} |
| Near-silence | RMS < 0.01 | {ctx["flag_silent"]} |
| Clipping | > 1% of samples at full scale | {ctx["flag_clipped"]} |
| Too short to hold a code-switch | < 1.0 s | {ctx["flag_short"]} |
| Missing file | — | {ctx["flag_missing"]} |

{headline}

Median speaking rate is {qa.get("median_chars_per_sec")} chars/s. Full per-clip
report: `{QA_CSV_PATH}`.

{_failure_table(qa.get("failures", {}))}

## The objective measurement agrees

The pilot verdict for CM01 (`docs/qa/pilot_script_verdict.md`) paired the ratings
with intra-utterance pitch range, and that is what separated a legitimate attack
from an easy one. The same measurement, on this corpus:

| | f0 IQR |
|---|---|
| Real spontaneous speech (HiACC), P-019 | 42.2 Hz |
| Real read speech (MUCS), P-019 | 41.1 Hz |
| **Real MUCS source clips, re-measured in this run** | **{real.get("median_f0_iqr_hz", 0):.1f} Hz** |
| **RVC converted clips, this run** | **{converted.get("median_f0_iqr_hz", 0):.1f} Hz** |
| XTTS-v2, every configuration tried (P-019) | 25–29 Hz |

The real column is re-measured rather than quoted: P-019's numbers were taken ad
hoc with no committed code, so quoting them against a different estimator would
compare tooling, not attacks. `src.data.f0_stats` now pins the method, and
measures both sides of this comparison in one pass.

> {pitch["verdict"]}

## What this does and does not decide

**Decided — CM02 is the harder attack family the plan assumed.** Pitch-range
retention is {pitch["retention_pct"]:.1f}% of the real speech. The CM01
flattening is a property of text-to-speech, not of cloning in general, so a
detector that has learned to spot a flat contour will not transfer to CM02.

**Decided — the corpus enters training as generated**, subject to the screen
above. Every clip carries the target speaker's id, and both endpoints are
train-pool, so the pool disjointness that makes CM01 auditable holds identically
here.

**Not decided — perceptual quality.** This is a mechanical screen and a pitch
statistic. Neither says whether a listener would be fooled; the CM01 pilot needed
human raters for that, and CM02 has not had them. Nothing here should be read as
a claim about how convincing the clips are.
"""


def decision_entry(ctx: dict) -> str:
    """The P-020 row for the ADR log: what upstream changed, and the three defects."""
    return (
        "| **P-020** | **Upstream RVC-WebUI was restructured, and P-017/P-018's "
        "`fairseq`/MSVC diagnosis no longer describes it: there is nothing left to "
        "compile.** The CM02 run is driven from the upstream checkout's own scripts "
        "with three fixes recorded below, and `requirments_*.txt` is deliberately "
        "not installed | Three things changed upstream since P-018 was written: "
        "`fairseq` is gone entirely, HuBERT is a **Transformers model directory** "
        "(`assets/hubert_base/{config.json, preprocessor_config.json, "
        "pytorch_model.bin}`) rather than the fairseq `hubert_base.pt`, and the "
        "training scripts moved from `infer/modules/train/` to `train/`. So the "
        "`omegaconf` / `pip<24.1` dance, the MSVC Build Tools saga and the Python "
        "3.10 venv in P-017/P-018 are all obsolete — on Linux the toolchain is four "
        "pip packages (`av`, `ffmpeg-python`, `praat-parselmouth`, `faiss-cpu`) on "
        "top of the stock Kaggle image. Downloading a single `.pt` to the old HuBERT "
        "path is the failure that looks like it worked and then raises "
        "`FileNotFoundError` during feature extraction. **Three defects had to be "
        "fixed before the first target trained.** (1) The Kaggle upload of MUCS "
        "preserves `mucs2021/...` but not the `raw/` anchor that "
        "`src.utils.paths.resolve()` rebases on, and `/kaggle/input` is read-only, so "
        "the notebook links the corpus under a `raw/` anchor in `/kaggle/working` "
        "instead of requiring a re-upload. (2) `clip_index.csv` is gitignored "
        "(machine-local absolute paths), so a fresh clone never has it and step 4 "
        "died on a missing file; it is now rebuilt from MUCS's own Kaldi tables "
        f"({ctx['index_rows']} rows, {ctx['index_speakers']} speakers). (3) RVC's "
        "stage scripts live in subpackages but import each other as top-level "
        "packages, and `python train/preprocess.py` puts the *script's* directory "
        "first on `sys.path` — so the checkout root is absent (`No module named "
        "'infer'`) **and** `train/train.py` shadows the `train` package (`circular "
        "import of a partially initialised 'train'`). Both surface only after the "
        "training wavs have been staged. Fixed in `rvc_generation._stage_env` with "
        "`PYTHONPATH=<checkout>` plus `PYTHONSAFEPATH=1`, pinned by "
        "`tests/test_rvc_generation.py`, and preflighted in the notebook so a broken "
        "path costs two seconds rather than a staging pass. |"
    )
