"""RVC voice conversion (CM02): per-speaker model training + batch conversion.

Week 4 (W4-T2). The second attack family in ``docs/attack_taxonomy.md``: ~12
per-speaker RVC models trained on train-pool voices, then real speech from *other*
train-pool speakers converted into each target's voice.

**Why a second family at all.** P-019 measured that XTTS-v2 (CM01) compresses
intra-utterance pitch range by ~35% against real speech -- it invents prosody from
text and regresses to a flat contour, and no generation parameter fixed it. That
makes CM01 a legitimate but *easy* attack, and a detector scoring near 0% EER on it
has not been tested hard. RVC starts from real speech and keeps the source pitch
contour, so that particular tell cannot appear.

**Why this module is subprocess-driven.** RVC training is not a library call. The
upstream WebUI exposes it as five scripts driven by its Gradio callbacks, and the
step that assembles ``filelist.txt`` and ``config.json`` has no CLI at all -- it
lives inside ``webui.py``'s ``click_train``. :func:`train_speaker_model`
reimplements that one step in Python and shells out for the other four, which is
why the RVC checkout is an argument rather than a dependency.

**What P-017/P-018 recorded is no longer the shape of the problem.** Both entries
are about getting ``fairseq`` to compile so ``hubert_base.pt`` could be loaded.
Upstream has since moved HuBERT to a Transformers model *directory*
(``assets/hubert_base/``, see ``infer/hubert.py``) and restructured the training
scripts from ``infer/modules/train/`` to ``train/``, so fairseq is not in the
picture and neither is the MSVC fight. The assets moved with it --
:func:`download_rvc_assets` and :func:`assert_rvc_assets` encode the current
layout, and the preflight runs before a GPU session is spent rather than after.

**The speaker label on a converted clip is the *target*.** Voice conversion of
source S into target T produces T's voice, so the record carries ``speaker=T``.
That is what makes the pool firewall mean the same thing for CM02 as for CM01
(``docs/attack_taxonomy.md`` rule 1). Both endpoints are held inside the train
pool, checked in :func:`assert_rvc_invariants`.

Heavy imports (torch, soundfile, huggingface_hub) stay lazy so this module imports
cheaply (CI-safe); everything above :func:`download_rvc_assets` is pure pandas and
unit-tested.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from src.data.pilot_jobs import PilotError, train_pool_speakers
from src.data.spoof_generation import read_metadata, reconcile_metadata, rewrite_metadata

__all__ = [
    "TOOL",
    "RVCConfig",
    "RVCError",
    "RVCInferencer",
    "RVCModel",
    "RVCRecord",
    "assert_rvc_assets",
    "assert_rvc_invariants",
    "build_rvc_jobs",
    "convert_batch",
    "download_rvc_assets",
    "gather_training_clips",
    "load_rvc_inferencer",
    "pending_jobs",
    "read_metadata",
    "reconcile_metadata",
    "rewrite_metadata",
    "rvc_generation_stats",
    "rvc_job_summary",
    "select_target_speakers",
    "train_speaker_model",
]

#: Tool tag written into every metadata record. Seen attack: CM02 may enter
#: training, unlike the held-out CM04/CM05 families.
TOOL = "rvc"

#: The CM02 row of the attack taxonomy: ~12 targets x 125 conversions.
DEFAULT_TARGETS = 12
DEFAULT_CONVERSIONS_PER_TARGET = 125

#: How far apart two targets start reading each source speaker's clip list. Coprime
#: with nothing in particular -- it just has to be a stride that does not divide a
#: typical per-speaker clip count, so targets do not collide on the same window.
_TARGET_STRIDE = 7

#: RVC job columns, mirroring the CM01 tables in ``scale_jobs``/``pilot_jobs`` so
#: both attack families can be read by the same downstream code.
JOB_COLUMNS = [
    "job_id",
    "target_speaker",
    "source_speaker",
    "pool",
    "source_utt_id",
    "source_wav",
    "start_seconds",
    "end_seconds",
    "duration_seconds",
    "transcript",
    "language",
    "tool",
    "output_path",
    "seed",
]

#: Assets the training pipeline reads, as ``relative path -> what fetches it``.
#: Kept as data so :func:`assert_rvc_assets` can name the exact missing file and
#: the exact command that supplies it, instead of failing inside a subprocess
#: forty minutes into a session.
_ASSET_DOWNLOADS = (
    (
        "assets/hubert_base/config.json",
        'hf download lj1995/VoiceConversionWebUI --include "hubert_base/*" --local-dir assets',
    ),
    (
        "assets/hubert_base/pytorch_model.bin",
        'hf download lj1995/VoiceConversionWebUI --include "hubert_base/*" --local-dir assets',
    ),
    (
        "assets/rmvpe/rmvpe.pt",
        "hf download lj1995/VoiceConversionWebUI rmvpe.pt --local-dir assets/rmvpe",
    ),
    (
        "logs/mute/0_gt_wavs",
        "hf download lj1995/VoiceConversionWebUI mute.zip --local-dir . "
        "&& python -m zipfile -e mute.zip logs",
    ),
)

#: RVC's sample-rate spelling to hertz, as ``webui.py``'s ``sr_dict`` has it.
_SR_HZ = {"32k": 32000, "40k": 40000, "48k": 48000}


class RVCError(RuntimeError):
    """Raised when the RVC toolchain or its assets are not usable."""


@dataclass(frozen=True)
class RVCConfig:
    """Size, thresholds and RVC hyper-parameters for the CM02 run."""

    n_targets: int = DEFAULT_TARGETS
    conversions_per_target: int = DEFAULT_CONVERSIONS_PER_TARGET
    train_epochs: int = 100
    #: A target needs enough speech to fit a voice model on. RVC guidance is ~10
    #: minutes; the MUCS train pool's thinnest speaker has ~300 s, so this
    #: threshold is what decides how many of the 25 speakers qualify.
    min_training_seconds: float = 240.0
    #: Cap per target: preprocessing scales linearly and the marginal minute buys
    #: very little once the model has a few hundred seconds of the voice.
    max_training_seconds: float = 900.0
    #: Segments shorter than this are mostly boundary, not voice.
    min_clip_seconds: float = 2.0
    #: Bounds on a conversion source. Too short and no code-switch boundary is left
    #: in the clip for a detector to see; too long and one clip eats a target's
    #: whole quota of conversion time.
    min_source_seconds: float = 2.5
    max_source_seconds: float = 15.0

    sample_rate: str = "40k"  # RVC's own spelling; 32k / 40k / 48k
    version: str = "v2"
    batch_size: int = 8
    save_every_epoch: int = 50
    #: ``rmvpe`` needs a GPU. Resolved at call time by :func:`_f0_method`, since
    #: the same config object is used for training and for conversion.
    f0_method: str = "rmvpe"
    n_processes: int = 8
    #: Conversion knobs, RVC-WebUI defaults. ``index_rate`` 0 disables the
    #: retrieval index, which is the fallback when faiss index training fails.
    index_rate: float = 0.75
    protect: float = 0.33
    pitch_shift: int = 0
    language: str = "hi"
    tool: str = TOOL
    seed: int = 4200

    def experiment_name(self, speaker: str) -> str:
        """The RVC experiment id for one target -- also the weight filename stem."""
        return f"{self.tool}_{speaker}"


# --------------------------------------------------------------------------- #
# Target selection + job assembly (pure pandas, testable)
# --------------------------------------------------------------------------- #
def select_target_speakers(
    index: pd.DataFrame, pools: pd.DataFrame, config: RVCConfig | None = None
) -> pd.DataFrame:
    """Train-pool speakers with enough audio to fit a voice model on.

    Returns ``speaker``, ``clips``, ``total_seconds``, longest-first and capped at
    ``config.n_targets``. Ties break on speaker id so two machines pick the same
    targets from the same frozen manifest (P-016).
    """
    cfg = config or RVCConfig()
    allowed = train_pool_speakers(pools)
    frame = index.copy()
    frame["speaker"] = frame["speaker"].astype(str)
    frame = frame[frame["speaker"].isin(allowed)]
    frame = frame[frame["duration_seconds"] >= cfg.min_clip_seconds]
    if frame.empty:
        return pd.DataFrame(columns=["speaker", "clips", "total_seconds"])

    totals = (
        frame.groupby("speaker")["duration_seconds"]
        .agg(clips="count", total_seconds="sum")
        .reset_index()
    )
    totals = totals[totals["total_seconds"] >= cfg.min_training_seconds]
    totals = totals.sort_values(
        ["total_seconds", "speaker"], ascending=[False, True], kind="stable"
    )
    return totals.head(cfg.n_targets).reset_index(drop=True)


def gather_training_clips(
    index: pd.DataFrame, target: str, config: RVCConfig | None = None
) -> pd.DataFrame:
    """The target speaker's own clips -- the material its voice model is fitted on.

    Ordered by ``utt_id`` (unique, so the order is machine-independent) and
    truncated at ``config.max_training_seconds``.
    """
    cfg = config or RVCConfig()
    frame = index.copy()
    frame["speaker"] = frame["speaker"].astype(str)
    frame = frame[frame["speaker"] == str(target)]
    frame = frame[frame["duration_seconds"] >= cfg.min_clip_seconds]
    frame = frame.sort_values("utt_id", kind="stable").reset_index(drop=True)
    if frame.empty:
        return frame
    keep = frame["duration_seconds"].cumsum() <= cfg.max_training_seconds
    # Keep at least one clip, so a mis-set cap surfaces as a training failure with
    # a readable message rather than an empty trainset directory handed to RVC.
    if not keep.any():
        keep.iloc[0] = True
    return frame[keep].reset_index(drop=True)


def _source_pool(index: pd.DataFrame, pools: pd.DataFrame, config: RVCConfig) -> pd.DataFrame:
    """Train-pool clips usable as conversion sources, ordered deterministically."""
    allowed = train_pool_speakers(pools)
    frame = index.copy()
    frame["speaker"] = frame["speaker"].astype(str)
    frame = frame[frame["speaker"].isin(allowed)]
    frame = frame[
        frame["duration_seconds"].between(config.min_source_seconds, config.max_source_seconds)
    ]
    return frame.sort_values("utt_id", kind="stable").reset_index(drop=True)


def build_rvc_jobs(
    index: pd.DataFrame,
    pools: pd.DataFrame,
    out_dir: str = "outputs",
    config: RVCConfig | None = None,
    targets: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Pair source clips with conversion targets, train pool on both ends.

    For each target, sources are drawn **round-robin across the other train-pool
    speakers**, so no single source voice dominates a target's clips -- the
    detector must learn the conversion artefact, not one speaker's channel.

    Each target also starts at a different offset into every source speaker's clip
    list. Without that the twelve targets would be twelve renderings of the same
    125 utterances, which makes the corpus far smaller than its clip count implies.
    """
    cfg = config or RVCConfig()
    chosen = select_target_speakers(index, pools, cfg) if targets is None else targets
    if chosen.empty:
        raise PilotError(
            f"no train-pool speaker has >= {cfg.min_training_seconds:.0f}s of audio "
            f"to fit an RVC model on"
        )

    sources = _source_pool(index, pools, cfg)
    if sources.empty:
        raise PilotError(
            f"no train-pool clip is between {cfg.min_source_seconds:.1f}s and "
            f"{cfg.max_source_seconds:.1f}s -- nothing to convert"
        )

    by_speaker = {
        str(speaker): group.reset_index(drop=True)
        for speaker, group in sources.groupby("speaker", sort=True)
    }
    pool_of = pools.set_index(pools["speaker"].astype(str))["pool"].to_dict()

    rows: list[dict[str, object]] = []
    job_id = 0
    for target_offset, target in enumerate(chosen["speaker"].astype(str)):
        others = sorted(s for s in by_speaker if s != target)
        if not others:
            raise PilotError(f"target {target} is the only train-pool speaker with source clips")

        # One cursor per source speaker, started at this target's offset and
        # wrapped modulo the speaker's clip count. A speaker is exhausted for this
        # target once the cursor has travelled its whole list, which is what stops
        # a short source speaker from being reused within one target.
        start = target_offset * _TARGET_STRIDE
        cursors = {s: start for s in others}
        exhausted: set[str] = set()
        taken = 0
        while taken < cfg.conversions_per_target and len(exhausted) < len(others):
            for source in others:
                if taken >= cfg.conversions_per_target:
                    break
                clips = by_speaker[source]
                if cursors[source] - start >= len(clips):
                    exhausted.add(source)
                    continue
                clip = clips.iloc[cursors[source] % len(clips)]
                cursors[source] += 1
                job_id += 1
                taken += 1
                rows.append(
                    {
                        "job_id": job_id,
                        "target_speaker": target,
                        "source_speaker": source,
                        "pool": pool_of.get(target, "unknown"),
                        "source_utt_id": str(clip["utt_id"]),
                        "source_wav": str(clip["wav_path"]),
                        "start_seconds": float(clip["start_seconds"]),
                        "end_seconds": float(clip["end_seconds"]),
                        "duration_seconds": float(clip["duration_seconds"]),
                        "transcript": str(clip.get("transcript", "")),
                        "language": cfg.language,
                        "tool": cfg.tool,
                        "output_path": f"{out_dir}/{cfg.tool}_{target}_{job_id:05d}.wav",
                        "seed": cfg.seed + job_id,
                    }
                )
        if taken < cfg.conversions_per_target:
            # Reported, never silent: the datasheet states clips per target, and a
            # shortfall means the source pool ran out, not that 125 was the plan.
            print(
                f"  [short] {target}: {taken}/{cfg.conversions_per_target} source clips "
                f"available across {len(others)} other train-pool speaker(s)"
            )

    jobs = pd.DataFrame(rows, columns=JOB_COLUMNS)
    assert_rvc_invariants(jobs, pools)
    return jobs


def assert_rvc_invariants(jobs: pd.DataFrame, pools: pd.DataFrame) -> None:
    """The three properties that make CM02 usable. Checked before any GPU time."""
    if jobs.empty:
        raise PilotError("empty RVC job table")

    # 1. Pool firewall, both endpoints. A converted clip carries the *target's*
    #    speaker id, but the source voice survives in the prosody, so a source
    #    drawn from the eval pool would leak an evaluation voice into training.
    allowed = train_pool_speakers(pools)
    lookup = pools.set_index(pools["speaker"].astype(str))["pool"].to_dict()
    for column in ("target_speaker", "source_speaker"):
        intruders = set(jobs[column].astype(str)) - allowed
        if intruders:
            detail = ", ".join(f"{s} (pool={lookup.get(s, 'unknown')})" for s in sorted(intruders))
            raise PilotError(f"RVC run would use non-train-pool {column}: {detail}")

    # 2. Converting a speaker into themselves is a re-encode, not an attack.
    same = jobs["target_speaker"].astype(str) == jobs["source_speaker"].astype(str)
    if same.any():
        raise PilotError(f"{int(same.sum())} job(s) convert a speaker into themselves")

    # 3. No repeated (target, source clip) pair -- a duplicate teaches the detector
    #    the utterance rather than the conversion artefact.
    pairs = jobs[["target_speaker", "source_utt_id"]].astype(str)
    if pairs.duplicated().any():
        n = int(pairs.duplicated().sum())
        raise PilotError(f"{n} duplicate (target, source clip) pair(s) in the RVC job table")


def rvc_job_summary(jobs: pd.DataFrame) -> str:
    """A human-readable digest of the job table, for the run log and datasheet."""
    if jobs.empty:
        return "0 RVC job(s)"
    per_target = jobs["target_speaker"].value_counts()
    hours = float(jobs["duration_seconds"].sum()) / 3600.0
    return "\n".join(
        [
            f"{len(jobs)} RVC job(s) across {jobs['target_speaker'].nunique()} target(s)",
            f"  per target: min {int(per_target.min())}, max {int(per_target.max())}",
            f"  source speakers: {jobs['source_speaker'].nunique()}",
            f"  unique source clips: {jobs['source_utt_id'].nunique()}",
            f"  source audio: {hours:.2f} h",
            f"  pools: {sorted(jobs['pool'].unique())}",
        ]
    )


# --------------------------------------------------------------------------- #
# Toolchain: assets + subprocess plumbing
# --------------------------------------------------------------------------- #
def _run(cmd: list[str], cwd: str, tail: int = 40) -> None:
    """Run one RVC stage, streaming its log. Raises :class:`RVCError` on failure.

    Output is streamed rather than captured: a training stage runs for minutes and
    a silent notebook cell is indistinguishable from a hung one. Only the tail is
    replayed in the exception, because RVC's per-file progress logs are long and
    the useful part is always at the end.
    """
    print("$", " ".join(cmd))
    lines: list[str] = []
    process = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    for line in process.stdout or ():
        line = line.rstrip()
        lines.append(line)
        print("  ", line)
    if process.wait() != 0:
        raise RVCError(
            f"RVC stage failed (exit {process.returncode}): {' '.join(cmd)}\n"
            + "\n".join(lines[-tail:])
        )


def _f0_method(config: RVCConfig) -> str:
    """``rmvpe`` needs a GPU; fall back to ``pm`` so a CPU smoke test still runs."""
    if config.f0_method == "pm":
        return "pm"
    try:
        import torch

        return "rmvpe" if torch.cuda.is_available() else "pm"
    except ImportError:
        return "pm"


def download_rvc_assets(rvc_repo: str, config: RVCConfig | None = None) -> list[str]:
    """Fetch the HuBERT / RMVPE / pretrained / mute assets into an RVC checkout.

    Upstream ships none of these in git. Two of them are easy to get wrong:

    - HuBERT is a **Transformers model directory** now, not the old fairseq
      ``hubert_base.pt``. A single ``.pt`` downloaded to the old path leaves
      ``infer/hubert.py`` raising ``FileNotFoundError`` at feature extraction,
      after preprocessing has already run.
    - ``logs/mute`` is not optional. Every training run's ``filelist.txt``
      references it, so a missing one fails inside the dataloader.

    Returns the paths fetched; assets already present are skipped, so this is safe
    to call on every run of a resumed notebook.
    """
    import zipfile

    from huggingface_hub import hf_hub_download, snapshot_download

    cfg = config or RVCConfig()
    repo = Path(rvc_repo)
    fetched: list[str] = []

    hubert = repo / "assets" / "hubert_base"
    if not (hubert / "config.json").is_file():
        snapshot_download(
            "lj1995/VoiceConversionWebUI",
            allow_patterns=["hubert_base/*"],
            local_dir=str(repo / "assets"),
        )
        fetched.append(str(hubert))

    rmvpe = repo / "assets" / "rmvpe" / "rmvpe.pt"
    if not rmvpe.is_file():
        rmvpe.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download("lj1995/VoiceConversionWebUI", "rmvpe.pt", local_dir=str(rmvpe.parent))
        fetched.append(str(rmvpe))

    # Only the generator/discriminator pair for the configured rate: the whole
    # pretrained_v2 folder is ~1.3 GB and eleven twelfths of it goes unused.
    pretrained = repo / "assets" / "pretrained_v2"
    pretrained.mkdir(parents=True, exist_ok=True)
    for name in (f"f0G{cfg.sample_rate}.pth", f"f0D{cfg.sample_rate}.pth"):
        if (pretrained / name).is_file():
            continue
        hf_hub_download(
            "lj1995/VoiceConversionWebUI",
            f"pretrained_v2/{name}",
            local_dir=str(repo / "assets"),
        )
        fetched.append(str(pretrained / name))

    mute = repo / "logs" / "mute"
    if not (mute / "0_gt_wavs").is_dir():
        archive = hf_hub_download(
            "lj1995/VoiceConversionWebUI", "mute.zip", local_dir=str(repo / ".model-downloads")
        )
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(repo / "logs")
        fetched.append(str(mute))

    assert_rvc_assets(rvc_repo, cfg)
    return fetched


def assert_rvc_assets(rvc_repo: str, config: RVCConfig | None = None) -> None:
    """Fail now, naming the fix, rather than inside a stage forty minutes in."""
    cfg = config or RVCConfig()
    repo = Path(rvc_repo)
    if not (repo / "train" / "train.py").is_file():
        raise RVCError(
            f"{rvc_repo} does not look like an RVC-WebUI checkout (no train/train.py). "
            f"Clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI"
        )

    required = list(_ASSET_DOWNLOADS)
    for name in (f"f0G{cfg.sample_rate}.pth", f"f0D{cfg.sample_rate}.pth"):
        required.append(
            (
                f"assets/pretrained_v2/{name}",
                'hf download lj1995/VoiceConversionWebUI --include "pretrained_v2/*" '
                "--local-dir assets",
            )
        )

    missing = [(rel, how) for rel, how in required if not (repo / rel).exists()]
    if missing:
        detail = "\n".join(f"  {rel}\n    <- {how}" for rel, how in missing)
        raise RVCError(
            f"RVC assets missing under {rvc_repo}:\n{detail}\n"
            f"src.data.rvc_generation.download_rvc_assets() fetches all of them."
        )


# --------------------------------------------------------------------------- #
# Training one target's voice model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RVCModel:
    """A trained per-speaker voice model and the retrieval index that goes with it."""

    speaker: str
    experiment: str
    model_path: str
    index_path: str
    repo_dir: str
    epochs: int
    n_clips: int
    train_seconds: float
    sample_rate: str
    version: str

    def describe(self) -> str:
        """One-line summary for logs and the run summary JSON."""
        index = Path(self.index_path).name if self.index_path else "no index"
        return (
            f"{self.speaker}: {Path(self.model_path).name} "
            f"({self.epochs} epochs on {self.n_clips} clips / {self.train_seconds:.0f}s, "
            f"{self.sample_rate} {self.version}, {index})"
        )


def _stage_training_wavs(clips: pd.DataFrame, dest: Path) -> int:
    """Cut each indexed segment out into its own wav for RVC's preprocessor.

    A MUCS row names a span inside a long lecture recording, so pointing RVC at
    ``wav_path`` would hand it the whole file -- every other speaker in the lecture
    included, which would poison the voice model with the wrong identity. RVC's
    preprocessor takes a flat directory of standalone files, so the spans are cut
    here first.
    """
    from src.data.corpora import load_clip
    from src.utils.audio_utils import save_wav

    dest.mkdir(parents=True, exist_ok=True)
    written = 0
    for row in clips.to_dict(orient="records"):
        target = dest / f"{row['utt_id']}.wav"
        if not target.is_file():
            audio, sample_rate = load_clip(row)
            save_wav(str(target), audio, sample_rate)
        written += 1
    return written


def _write_filelist(exp_dir: Path, repo: Path, config: RVCConfig, seed: int) -> int:
    """Assemble ``filelist.txt`` and ``config.json`` -- the stage with no CLI.

    Upstream does this inside ``webui.py``'s ``click_train`` callback, so it has to
    be reproduced rather than invoked. Two details are easy to miss and both are
    fatal:

    - the trailing **mute lines** are mandatory. ``logs/mute`` supplies the silence
      example every RVC run trains the speaker embedding against; the filelist
      carries two copies of it, exactly as ``click_train`` writes them.
    - for ``40k`` the config template comes from ``configs/v1/40k.json`` **even for
      v2**, because ``configs/v2/40k.json`` does not exist upstream.
      ``click_train`` encodes the same special case.
    """
    import copy
    import random

    gt_wavs = exp_dir / "0_gt_wavs"
    feature_dim = 256 if config.version == "v1" else 768
    features = exp_dir / f"3_feature{feature_dim}"
    f0_dir = exp_dir / "2a_f0"
    f0nsf_dir = exp_dir / "2b-f0nsf"

    names = (
        {p.name.split(".")[0] for p in gt_wavs.glob("*.wav")}
        & {p.name.split(".")[0] for p in features.glob("*.npy")}
        & {p.name.split(".")[0] for p in f0_dir.glob("*.npy")}
        & {p.name.split(".")[0] for p in f0nsf_dir.glob("*.npy")}
    )
    if not names:
        raise RVCError(
            f"no slice survived all three extraction stages in {exp_dir} -- "
            f"preprocess / f0 / feature extraction produced nothing to train on"
        )

    lines = [
        f"{gt_wavs}/{name}.wav|{features}/{name}.npy|"
        f"{f0_dir}/{name}.wav.npy|{f0nsf_dir}/{name}.wav.npy|0"
        for name in sorted(names)
    ]
    mute = repo / "logs" / "mute"
    mute_line = (
        f"{mute}/0_gt_wavs/mute{config.sample_rate}.wav|"
        f"{mute}/3_feature{feature_dim}/mute.npy|"
        f"{mute}/2a_f0/mute.wav.npy|{mute}/2b-f0nsf/mute.wav.npy|0"
    )
    lines.extend([mute_line, mute_line])
    # Shuffled, but seeded: RVC expects a shuffled filelist, and we expect two runs
    # of the same config to be the same run.
    random.Random(seed).shuffle(lines)
    (exp_dir / "filelist.txt").write_text("\n".join(lines), encoding="utf-8")

    family = "v1" if config.version == "v1" or config.sample_rate == "40k" else config.version
    template = repo / "configs" / family / f"{config.sample_rate}.json"
    if not template.is_file():
        raise RVCError(f"no RVC config template at {template}")
    data = copy.deepcopy(json.loads(template.read_text(encoding="utf-8")))
    data.pop("speaker_info", None)  # single-speaker model
    (exp_dir / "config.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=4, sort_keys=True) + "\n", encoding="utf-8"
    )
    return len(names)


def train_speaker_model(
    target: str,
    clips: pd.DataFrame,
    train_root: str,
    rvc_repo: str,
    config: RVCConfig | None = None,
    python: str | None = None,
) -> RVCModel:
    """Fit one target speaker's RVC voice model. Six stages, five of them subprocesses.

    ``train_root`` holds the cut training wavs. The RVC *experiment* directory has
    to live at ``<rvc_repo>/logs/<experiment>`` regardless of where that is, because
    ``train/train.py`` resolves its ``-e`` argument against a hardcoded ``./logs``
    -- so the experiment path is derived here rather than taken from ``train_root``.

    Resumable: an existing weight file for this experiment is returned as-is. A
    Kaggle session can be reclaimed part-way through twelve targets, and re-running
    the notebook has to continue rather than retrain target one.
    """
    global _LAST_RVC_REPO

    from src.data.ethics_gate import require_signoff

    cfg = config or RVCConfig()
    require_signoff(action=f"RVC voice-model training for speaker {target}")
    assert_rvc_assets(rvc_repo, cfg)

    repo = Path(rvc_repo).resolve()
    _LAST_RVC_REPO = str(repo)
    interpreter = python or sys.executable
    experiment = cfg.experiment_name(str(target))
    exp_dir = repo / "logs" / experiment
    model_path = repo / "assets" / "weights" / f"{experiment}.pth"
    train_seconds = float(clips["duration_seconds"].sum())

    def built() -> RVCModel:
        return RVCModel(
            speaker=str(target),
            experiment=experiment,
            model_path=str(model_path),
            index_path=_find_index(exp_dir, experiment, cfg),
            repo_dir=str(repo),
            epochs=cfg.train_epochs,
            n_clips=len(clips),
            train_seconds=train_seconds,
            sample_rate=cfg.sample_rate,
            version=cfg.version,
        )

    if model_path.is_file():
        print(f"  [skip] {model_path.name} already trained")
        return built()

    trainset = Path(train_root) / experiment / "wavs"
    staged = _stage_training_wavs(clips, trainset)
    print(f"  staged {staged} training wav(s) -> {trainset}")

    exp_dir.mkdir(parents=True, exist_ok=True)
    (repo / "assets" / "weights").mkdir(parents=True, exist_ok=True)
    (repo / "assets" / "indices").mkdir(parents=True, exist_ok=True)
    sr_hz = _SR_HZ[cfg.sample_rate]
    f0_method = _f0_method(cfg)
    on_gpu = f0_method == "rmvpe"

    # 1. slice + resample -> 0_gt_wavs / 1_16k_wavs
    _run(
        [
            interpreter,
            "train/preprocess.py",
            str(trainset),
            str(sr_hz),
            str(cfg.n_processes),
            str(exp_dir),
            "False",
            "3.7",
        ],
        cwd=str(repo),
    )
    # 2. f0 -> 2a_f0 / 2b-f0nsf
    if on_gpu:
        _run(
            [
                interpreter,
                "train/dataset/extract_f0.py",
                "cuda",
                "1",
                "0",
                "0",
                str(exp_dir),
                "True",
            ],
            cwd=str(repo),
        )
    else:
        _run(
            [
                interpreter,
                "train/dataset/extract_f0.py",
                "cpu",
                str(exp_dir),
                str(cfg.n_processes),
                f0_method,
            ],
            cwd=str(repo),
        )
    # 3. HuBERT features -> 3_feature768
    _run(
        [
            interpreter,
            "train/dataset/extract_hubert_feature.py",
            "cuda:0" if on_gpu else "cpu",
            "1",
            "0",
            str(exp_dir),
            cfg.version,
            str(on_gpu),
        ],
        cwd=str(repo),
    )
    # 4. filelist.txt + config.json (no CLI upstream -- see _write_filelist)
    n_slices = _write_filelist(exp_dir, repo, cfg, seed=cfg.seed + int(len(clips)))
    print(f"  filelist: {n_slices} slice(s)")
    # 5. train, warm-started from the pretrained v2 pair. Each target has minutes,
    #    not hours, of audio -- far below a from-scratch budget -- so the warm start
    #    is what makes a per-speaker model viable at all.
    command = [
        interpreter,
        "train/train.py",
        "-e",
        experiment,
        "-sr",
        cfg.sample_rate,
        "-f0",
        "1",
        "-bs",
        str(cfg.batch_size),
        "-te",
        str(cfg.train_epochs),
        "-se",
        str(min(cfg.save_every_epoch, cfg.train_epochs)),
        "-pg",
        f"assets/pretrained_v2/f0G{cfg.sample_rate}.pth",
        "-pd",
        f"assets/pretrained_v2/f0D{cfg.sample_rate}.pth",
        "-l",
        "1",
        "-c",
        "0",
        "-sw",
        "1",
        "-v",
        cfg.version,
    ]
    if on_gpu:
        command += ["-g", "0"]  # omitted entirely on CPU, as webui.py does
    _run(command, cwd=str(repo))
    if not model_path.is_file():
        raise RVCError(f"training finished but no weight file at {model_path}")

    # 6. retrieval index. Optional by design: faiss is the flakiest dependency in
    #    the stack and conversion still runs without it at index_rate 0, so a
    #    failure here costs timbre fidelity rather than the whole target.
    try:
        _run(
            [
                interpreter,
                "train/train_index.py",
                experiment,
                cfg.version,
                "assets/indices",
                str(cfg.n_processes),
                "single",
            ],
            cwd=str(repo),
        )
    except RVCError as exc:
        print(f"  [warn] index training failed, will convert without it: {exc}")

    return built()


def _find_index(exp_dir: Path, experiment: str, config: RVCConfig) -> str:
    """The newest ``added_*`` faiss index for an experiment, or "" if none exists."""
    matches = [
        p for p in exp_dir.glob(f"added_*_{experiment}_{config.version}*.index") if p.is_file()
    ]
    if not matches:
        return ""
    return str(max(matches, key=lambda p: p.stat().st_mtime))


# --------------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RVCInferencer:
    """A bound RVC checkout plus conversion settings, used by :func:`convert_batch`."""

    repo_dir: str
    python: str
    config: RVCConfig

    def describe(self) -> str:
        """One-line summary for logs."""
        return (
            f"RVC CLI at {self.repo_dir} (f0={_f0_method(self.config)}, "
            f"index_rate={self.config.index_rate}, protect={self.config.protect})"
        )


#: Remembered by :func:`train_speaker_model` so that a bare ``load_rvc_inferencer()``
#: -- the form the notebook uses, after training has already located the checkout --
#: does not need the path repeated.
_LAST_RVC_REPO: str | None = None


def load_rvc_inferencer(
    rvc_repo: str | None = None,
    config: RVCConfig | None = None,
    python: str | None = None,
) -> RVCInferencer:
    """Bind the conversion CLI to an RVC checkout.

    Conversion goes through the checkout's own ``infer/cli.py`` rather than a
    separate pip package. The model was produced by *this* tree's ``train.py``, and
    the CLI shares its HuBERT/RMVPE assets and checkpoint format -- a second RVC
    implementation would have to be version-matched by hand on every upgrade, for
    no gain.

    The checkout is taken from the argument, else ``$RVC_REPO``, else whichever one
    the last training call used.
    """
    repo = rvc_repo or os.environ.get("RVC_REPO") or _LAST_RVC_REPO
    if not repo:
        raise RVCError(
            "no RVC checkout known: pass rvc_repo=..., set $RVC_REPO, or call "
            "train_speaker_model() first"
        )
    cfg = config or RVCConfig()
    assert_rvc_assets(repo, cfg)
    if not (Path(repo) / "infer" / "cli.py").is_file():
        raise RVCError(f"{repo}/infer/cli.py not found -- update the RVC checkout")
    return RVCInferencer(
        repo_dir=str(Path(repo).resolve()), python=python or sys.executable, config=cfg
    )


@dataclass
class RVCRecord:
    """Provenance for one converted clip (one JSON line per file).

    Field names match :class:`~src.data.spoof_generation.GenerationRecord` wherever
    they mean the same thing, so ``reconcile_metadata`` and the datasheet builders
    read CM01 and CM02 logs through one code path. ``speaker`` is the **target**:
    the clip carries the target's voice, so that is the identity the split firewall
    has to reason about.
    """

    output_path: str
    tool: str
    speaker: str
    source_speaker: str
    source_utt_id: str
    reference_wav: str
    transcript: str
    language: str
    pool: str
    seed: int
    settings: dict = field(default_factory=dict)


def _append_record(record: RVCRecord, jsonl_path: str) -> None:
    """Append one conversion record as a JSON line."""
    path = Path(jsonl_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def pending_jobs(jobs: pd.DataFrame) -> pd.DataFrame:
    """The rows whose converted clip does not exist yet (makes a run resumable)."""
    if jobs.empty:
        return jobs
    return jobs[~jobs["output_path"].astype(str).map(lambda p: Path(p).is_file())]


def row_to_clip(row: dict) -> dict:
    """A job row reshaped into what ``corpora.load_clip`` expects."""
    return {
        "wav_path": row["source_wav"],
        "start_seconds": row["start_seconds"],
        "end_seconds": row["end_seconds"],
    }


def _convert_command(
    inferencer: RVCInferencer, model: RVCModel, src_dir: Path, out_dir: Path
) -> list[str]:
    """The ``infer/cli.py`` invocation for one target's batch."""
    cfg = inferencer.config
    command = [
        inferencer.python,
        "infer/cli.py",
        "--model",
        model.model_path,
        "--input",
        str(src_dir),
        "--output",
        str(out_dir),
        "--f0-method",
        _f0_method(cfg),
        "--pitch",
        str(cfg.pitch_shift),
        "--protect",
        str(cfg.protect),
        "--format",
        "wav",
        "--overwrite",
    ]
    if model.index_path:
        command += ["--index", model.index_path, "--index-rate", str(cfg.index_rate)]
    else:
        command += ["--index-rate", "0"]
    return command


def _fail(failures: list[dict] | None, row: dict, error: str) -> None:
    """Record one failed job. Never silent -- the failure rate goes in the datasheet."""
    record = {
        "output_path": str(row.get("output_path", "")),
        "target_speaker": str(row.get("target_speaker", "")),
        "source_speaker": str(row.get("source_speaker", "")),
        "source_utt_id": str(row.get("source_utt_id", "")),
        "error": error,
    }
    if failures is not None:
        failures.append(record)
    print(f"    [FAILED] {Path(record['output_path']).name}: {error}")


def convert_batch(
    jobs: pd.DataFrame,
    inferencer: RVCInferencer,
    models: dict[str, RVCModel],
    metadata_path: str,
    skip_existing: bool = True,
    on_progress=None,
    failures: list[dict] | None = None,
) -> list[str]:
    """Convert every job into its target's voice, one metadata record per success.

    Conversion is batched **per target**, not per clip: ``infer/cli.py`` loads the
    voice model, HuBERT and RMVPE at startup, so a process per clip would spend
    almost all of a 1,500-clip run loading models. Source segments are cut into a
    staging directory under the name their output will have, the CLI runs once over
    that directory, and the results are moved into place.

    ``skip_existing`` makes the run resumable, for the same reason the CM01 batch is
    (W4-T1): a Kaggle session can be reclaimed mid-batch, and a re-run must continue
    rather than start over and duplicate metadata records.
    """
    import shutil
    import tempfile

    from src.data.corpora import load_clip
    from src.data.ethics_gate import require_signoff
    from src.utils.audio_utils import save_wav

    require_signoff(action="RVC voice conversion")
    todo = pending_jobs(jobs) if skip_existing else jobs
    written: list[str] = []
    if todo.empty:
        return written

    done = 0
    total = len(todo)
    for target, group in todo.groupby("target_speaker", sort=True):
        target = str(target)
        rows = group.to_dict(orient="records")
        model = models.get(target)
        if model is None:
            for row in rows:
                done += 1
                _fail(failures, row, "no trained model for this target")
            continue

        staging = Path(tempfile.mkdtemp(prefix=f"rvc_{target}_"))
        try:
            src_dir, out_dir = staging / "in", staging / "out"
            src_dir.mkdir()
            staged: dict[str, dict] = {}
            for row in rows:
                done += 1
                if on_progress is not None:
                    on_progress(done, total, row)
                stem = Path(str(row["output_path"])).stem
                try:
                    audio, sample_rate = load_clip(row_to_clip(row))
                    save_wav(str(src_dir / f"{stem}.wav"), audio, sample_rate)
                except Exception as exc:  # noqa: BLE001 - one unreadable source is not the run
                    _fail(failures, row, f"{type(exc).__name__}: {exc}")
                    continue
                staged[stem] = row
            if not staged:
                continue

            try:
                _run(_convert_command(inferencer, model, src_dir, out_dir), cwd=inferencer.repo_dir)
            except RVCError as exc:
                # The CLI exits non-zero if *any* clip in the batch failed, having
                # written the rest, so the batch is reconciled against the disk
                # below rather than abandoned here.
                print(f"  [warn] conversion batch for {target} reported failures: {exc}")

            for stem, row in staged.items():
                produced = out_dir / f"{stem}.wav"
                if not produced.is_file():
                    _fail(failures, row, "conversion produced no output")
                    continue
                final = Path(str(row["output_path"]))
                final.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(produced), str(final))
                _append_record(
                    RVCRecord(
                        output_path=str(final),
                        tool=str(row["tool"]),
                        speaker=target,
                        source_speaker=str(row["source_speaker"]),
                        source_utt_id=str(row["source_utt_id"]),
                        reference_wav=str(row["source_wav"]),
                        transcript=str(row["transcript"]),
                        language=str(row["language"]),
                        pool=str(row["pool"]),
                        seed=int(row["seed"]),
                        settings={
                            "model": Path(model.model_path).name,
                            "epochs": model.epochs,
                            "sample_rate": model.sample_rate,
                            "version": model.version,
                            "f0_method": _f0_method(inferencer.config),
                            "index_rate": (
                                inferencer.config.index_rate if model.index_path else 0.0
                            ),
                            "protect": inferencer.config.protect,
                            "pitch_shift": inferencer.config.pitch_shift,
                        },
                    ),
                    metadata_path,
                )
                written.append(str(final))
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    return written


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def rvc_generation_stats(records: list[dict]) -> dict:
    """Summarise CM02 metadata for the audit report and the datasheet."""
    from collections import Counter

    per_target = Counter(r["speaker"] for r in records)
    per_source = Counter(r.get("source_speaker", "") for r in records)
    return {
        "total": len(records),
        "by_tool": dict(Counter(r["tool"] for r in records)),
        "by_pool": dict(Counter(r["pool"] for r in records)),
        "by_language": dict(Counter(r["language"] for r in records)),
        "n_targets": len(per_target),
        "n_source_speakers": len([s for s in per_source if s]),
        "per_target_min": min(per_target.values()) if per_target else 0,
        "per_target_max": max(per_target.values()) if per_target else 0,
        "unique_source_clips": len({r.get("source_utt_id", "") for r in records}),
    }


def main() -> None:
    """CLI: ``python -m src.data.rvc_generation --jobs-only`` (or with ``--rvc-repo``).

    The notebook is the primary runner -- RVC wants a GPU and a Linux container --
    so the job-table half is what this is mostly for: it is pure pandas, and being
    able to check the firewall and the per-target counts on the laptop is worth
    more than a session spent finding out.
    """
    import argparse

    from src.data.ethics_gate import signoff_status
    from src.utils.paths import data_root, resolve_column

    parser = argparse.ArgumentParser(description="Assemble / run the W4-T2 RVC attack (CM02)")
    parser.add_argument("--index", default="data/manifests/clip_index.csv")
    parser.add_argument("--pools", default="data/manifests/speaker_pools.csv")
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "data"))
    parser.add_argument("--jobs-out", default="data/manifests/rvc_generation_jobs.csv")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--n-targets", type=int, default=DEFAULT_TARGETS)
    parser.add_argument("--per-target", type=int, default=DEFAULT_CONVERSIONS_PER_TARGET)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--rvc-repo", default=os.environ.get("RVC_REPO"))
    parser.add_argument("--train-root", default=None, help="where training wavs are cut to")
    parser.add_argument("--jobs-only", action="store_true", help="build the table, do not train")
    args = parser.parse_args()

    config = RVCConfig(
        n_targets=args.n_targets,
        conversions_per_target=args.per_target,
        train_epochs=args.epochs,
    )
    index = resolve_column(pd.read_csv(args.index), "wav_path", root=args.data_root)
    pools = pd.read_csv(args.pools)

    targets = select_target_speakers(index, pools, config)
    print(f"{len(targets)} target(s) qualify (>= {config.min_training_seconds:.0f}s each):")
    print(targets.to_string(index=False))

    jobs = build_rvc_jobs(index, pools, out_dir=args.out_dir, config=config, targets=targets)
    Path(args.jobs_out).parent.mkdir(parents=True, exist_ok=True)
    jobs.to_csv(args.jobs_out, index=False)
    print(rvc_job_summary(jobs))
    print(f"\nwrote {args.jobs_out}  (DATA_ROOT={data_root(args.data_root)})")

    if args.jobs_only:
        return
    if not args.rvc_repo:
        raise SystemExit("--rvc-repo (or $RVC_REPO) is required unless --jobs-only")

    status = signoff_status()
    print(status.describe())
    if not status.signed:
        raise SystemExit(1)

    train_root = args.train_root or str(Path(args.data_root) / "interim" / "rvc_train")
    models: dict[str, RVCModel] = {}
    for target in targets["speaker"].astype(str):
        clips = gather_training_clips(index, target, config)
        print(f"training {target} ({len(clips)} clips, {clips['duration_seconds'].sum():.0f}s)...")
        models[target] = train_speaker_model(target, clips, train_root, args.rvc_repo, config)
        print("  ", models[target].describe())

    metadata_path = str(Path(args.out_dir) / "generation_metadata.jsonl")
    failures: list[dict] = []
    written = convert_batch(
        jobs,
        load_rvc_inferencer(args.rvc_repo, config),
        models,
        metadata_path=metadata_path,
        failures=failures,
    )
    print(f"\nconverted {len(written)} clip(s) -> {args.out_dir}")
    if failures:
        log = Path(args.out_dir) / "generation_failures.jsonl"
        with log.open("a", encoding="utf-8") as fh:
            for record in failures:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{len(failures)} job(s) failed -> {log}")

    kept = rewrite_metadata(metadata_path)
    print(f"metadata reconciled: {kept} record(s) matching files on disk")
    print(json.dumps(rvc_generation_stats(read_metadata(metadata_path)), indent=2))


if __name__ == "__main__":
    main()
