"""Check a training config against the dataset rule before it is launched (P-028).

``tests/test_splits.py`` proves the leakage checks work on synthetic manifests. It
never looks at what a real config actually trains on. That gap mattered less while
only two kinds of run existed; S3 native training adds a third, and it is the first
run that fine-tunes the whole encoder on code-mixed audio. So this module reads
each ``configs/train_*.yaml``, opens the manifests it names, and applies the rule
to the rows it would really train on:

1. no held-out tool (Tortoise), no evaluation-only corpus (IndicSynth,
   IndicTTS-Deepfake, IndicVoices, HiACC), no child audio;
2. no eval-pool speaker in training -- measured on the frozen pools, not on the
   config's own split column, which a mistake could set either way;
3. code-mixed rows only in a Stage-3 run: either LoRA adaptation or a config that
   declares ``system: s3_native``;
4. an S3 native run starts from the pretrained encoder (no ``init_from``), because
   the point of S3 is that it has never seen English spoofing data;
5. RVC conversions only from speakers absent from the RVC test half (P-025).

Pure pandas/yaml, so CI runs it on every committed config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.data import build_manifests as bm

POOLS = "data/manifests/speaker_pools.csv"
RVC_TEST = "data/manifests/rvc_holdout_test.csv"
EVAL_ONLY = bm.EVAL_ONLY_SOURCES | {"hiacc"}
S3_NATIVE = "s3_native"


@dataclass
class ConfigReport:
    """What a config trains on, and every rule it breaks."""

    config: str
    rows: int = 0
    codemix_rows: int = 0
    problems: list[str] = field(default_factory=list)
    unverifiable: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def _training_rows(cfg: dict, root: Path, report: ConfigReport) -> pd.DataFrame | None:
    frames = []
    for key in ("train_manifest", "dev_manifest"):
        path = cfg.get(key)
        if not path:
            continue
        if not (root / path).is_file():
            report.unverifiable.append(f"{key} not in the repository: {path}")
            continue
        frames.append(pd.read_csv(root / path))
    return pd.concat(frames, ignore_index=True) if frames else None


def check_rows(
    rows: pd.DataFrame,
    cfg: dict,
    eval_speakers: set[str],
    rvc_test_speakers: set[str],
) -> list[str]:
    """Every rule the rows of one config break. Empty means clean."""
    problems = []
    tool = rows["tool"].astype(str).str.lower()
    source = rows["source"].astype(str).str.lower()
    speaker = rows["speaker"].astype(str)

    if (tool == bm.HELD_OUT_TOOL).any():
        problems.append(f"{int((tool == bm.HELD_OUT_TOOL).sum())} rows of the held-out tool")
    bad_sources = sorted(set(source) & EVAL_ONLY)
    if bad_sources:
        problems.append(f"evaluation-only corpus in training: {bad_sources}")
    if (source.str.contains("child") | speaker.str.lower().str.contains("child")).any():
        problems.append("child audio in training")

    leaked = sorted(set(speaker) & eval_speakers)
    if leaked:
        problems.append(f"{len(leaked)} eval-pool speakers in training, e.g. {leaked[:3]}")

    codemix = rows["language"].astype(str).str.lower().eq("hi-en")
    if codemix.any():
        is_lora = bool(cfg.get("lora"))
        is_s3 = cfg.get("system") == S3_NATIVE
        if not (is_lora or is_s3):
            problems.append(
                "code-mixed rows in a run that is neither LoRA adaptation nor system: s3_native"
            )
        if is_s3 and cfg.get("init_from"):
            problems.append("s3_native must start from the pretrained encoder, not init_from")

    rvc_speakers = set(speaker[tool == "rvc"])
    shared = sorted(rvc_speakers & rvc_test_speakers)
    if shared:
        problems.append(f"RVC training speakers also in the RVC test half: {shared[:3]}")
    return problems


def check_config(path: str, root: str = ".") -> ConfigReport:
    """Open one config's manifests and check them against the dataset rule."""
    import yaml

    base = Path(root)
    cfg = yaml.safe_load((base / path).read_text(encoding="utf-8")) or {}
    report = ConfigReport(config=path)
    rows = _training_rows(cfg, base, report)
    if rows is None:
        return report

    pools = pd.read_csv(base / POOLS)
    eval_speakers = set(pools.loc[pools["pool"] == "eval", "speaker"].astype(str))
    rvc_test = pd.read_csv(base / RVC_TEST) if (base / RVC_TEST).is_file() else None
    rvc_test_speakers = set(rvc_test["speaker"].astype(str)) if rvc_test is not None else set()

    report.rows = len(rows)
    report.codemix_rows = int(rows["language"].astype(str).str.lower().eq("hi-en").sum())
    report.problems = check_rows(rows, cfg, eval_speakers, rvc_test_speakers)
    return report


def main() -> None:
    """CLI: ``python -m src.training.config_guard [configs ...]`` -- exits 1 on any breach."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Check training configs against the data rule")
    parser.add_argument("configs", nargs="*", help="default: every configs/train_*.yaml")
    args = parser.parse_args()
    paths = args.configs or sorted(
        str(p).replace("\\", "/") for p in Path("configs").glob("train_*.yaml")
    )

    failed = False
    for path in paths:
        report = check_config(path)
        if report.problems:
            failed = True
            status = "FAIL"
        elif report.unverifiable and not report.rows:
            status = "skip"
        else:
            status = "ok"
        print(f"{status:4s} {path}  rows={report.rows} code-mixed={report.codemix_rows}")
        for problem in report.problems:
            print(f"     - {problem}")
        for note in report.unverifiable:
            print(f"     . {note}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
