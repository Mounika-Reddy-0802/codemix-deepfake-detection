"""Automatic quality screen for generated clips (W4-T6).

The failure this exists for: XTTS can return a clip that is **valid audio and
completely wrong** without raising anything. In the 40-clip script pilot, one job
produced 0.83 s of audio from a 150-character transcript -- roughly 180 characters
per second, which is not speech. ``generate_batch`` catches exceptions, and no
exception was raised, so the clip was written, logged as a success, and counted.

At the Week-4 scale of ~4,000 clips a 2.5% silent-failure rate is ~100 dead files
sitting inside the training corpus, each one teaching the detector that a spoof is
a 0.8-second noise. Nobody listens to 4,000 clips, so this has to be mechanical.

Three checks, all cheap and all on the *relationship* between the transcript and
the audio rather than on the audio alone:

- **speaking rate** -- characters per second. Hindi/English speech sits around
  10-20; far outside that means the model either stopped early or stalled.
- **near-silence** -- RMS below a floor, i.e. the file has audio but no speech.
- **clipping** -- a large fraction of samples at full scale, which is distortion
  rather than voice.

Pure numpy plus the standard library ``wave`` module, so it runs anywhere the
corpus does and needs no torch.
"""

from __future__ import annotations

import contextlib
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

#: Plausible speaking rate for the code-mixed material in this project, in
#: characters per second of the romanised transcript. The pilot measured a median
#: of 13.9 (Devanagari) and 14.4 (romanised); these bounds sit well outside both
#: so that natural variation is never flagged, only genuine failures.
MIN_CHARS_PER_SEC = 6.0
MAX_CHARS_PER_SEC = 30.0

#: Below this RMS the file carries no usable speech.
SILENCE_RMS = 0.01
#: Fraction of samples at full scale above which the clip is distorted.
MAX_CLIPPED_FRACTION = 0.01
#: A clip shorter than this cannot hold a code-switch boundary.
MIN_DURATION_SEC = 1.0


@dataclass(frozen=True)
class QAConfig:
    """Thresholds for the screen."""

    min_chars_per_sec: float = MIN_CHARS_PER_SEC
    max_chars_per_sec: float = MAX_CHARS_PER_SEC
    silence_rms: float = SILENCE_RMS
    max_clipped_fraction: float = MAX_CLIPPED_FRACTION
    min_duration_sec: float = MIN_DURATION_SEC


def probe(path: str) -> dict:
    """Duration, RMS and clipped fraction for one 16-bit wav."""
    with contextlib.closing(wave.open(str(path))) as handle:
        frames, rate = handle.getnframes(), handle.getframerate()
        raw = handle.readframes(frames)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if audio.size == 0:
        return {"duration_sec": 0.0, "rms": 0.0, "clipped_fraction": 0.0}
    return {
        "duration_sec": frames / rate if rate else 0.0,
        "rms": float(np.sqrt(np.mean(audio**2))),
        "clipped_fraction": float(np.mean(np.abs(audio) >= 0.999)),
    }


def reasons(row: dict, config: QAConfig | None = None) -> list[str]:
    """Why a clip fails, or an empty list. Every failure names itself."""
    cfg = config or QAConfig()
    out: list[str] = []
    duration = float(row.get("duration_sec", 0.0))
    if duration < cfg.min_duration_sec:
        out.append(f"too short ({duration:.2f}s)")
    if float(row.get("rms", 0.0)) < cfg.silence_rms:
        out.append(f"near-silent (rms {row.get('rms', 0.0):.4f})")
    if float(row.get("clipped_fraction", 0.0)) > cfg.max_clipped_fraction:
        out.append(f"clipped ({100 * row['clipped_fraction']:.1f}% of samples)")
    if duration > 0:
        rate = len(str(row.get("transcript", ""))) / duration
        if rate > cfg.max_chars_per_sec:
            out.append(f"speech too fast / truncated ({rate:.0f} chars/s)")
        elif rate < cfg.min_chars_per_sec:
            out.append(f"speech too slow / stalled ({rate:.0f} chars/s)")
    return out


def screen(jobs: pd.DataFrame, out_dir: str, config: QAConfig | None = None) -> pd.DataFrame:
    """Probe every generated clip and mark each pass or fail with a reason."""
    rows: list[dict] = []
    for job in jobs.itertuples(index=False):
        name = str(job.output_path).replace("\\", "/").rsplit("/", 1)[-1]
        path = Path(out_dir) / name
        record: dict = {
            "clip": name,
            "speaker": str(job.speaker),
            "language": str(getattr(job, "language", "")),
            "transcript": str(job.transcript),
            "chars": len(str(job.transcript)),
        }
        if not path.is_file():
            record.update(
                duration_sec=0.0, rms=0.0, clipped_fraction=0.0, ok=False, reason="missing"
            )
            rows.append(record)
            continue
        record.update(probe(str(path)))
        why = reasons(record, config)
        record["chars_per_sec"] = (
            round(record["chars"] / record["duration_sec"], 1) if record["duration_sec"] else 0.0
        )
        record["ok"] = not why
        record["reason"] = "; ".join(why)
        rows.append(record)
    return pd.DataFrame(rows)


def summarise(report: pd.DataFrame) -> dict:
    """Pass rate and the failure breakdown, for the datasheet."""
    if report.empty:
        return {"clips": 0}
    failed = report[~report["ok"]]
    return {
        "clips": int(len(report)),
        "passed": int(report["ok"].sum()),
        "failed": int(len(failed)),
        "pass_rate": round(100 * float(report["ok"].mean()), 2),
        "median_chars_per_sec": float(report.loc[report["ok"], "chars_per_sec"].median())
        if report["ok"].any()
        else None,
        "failures": failed["reason"].value_counts().to_dict(),
    }


def main() -> None:
    """CLI: ``python -m src.data.generation_qa --jobs JOBS.csv --out-dir DIR``."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Quality-screen generated clips")
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--report", default=None, help="default: <out-dir>/qa_report.csv")
    parser.add_argument(
        "--fail-over",
        type=float,
        default=None,
        help="exit 1 if the failure rate exceeds this percentage",
    )
    args = parser.parse_args()

    jobs = pd.read_csv(args.jobs)
    report = screen(jobs, args.out_dir)
    destination = args.report or str(Path(args.out_dir) / "qa_report.csv")
    report.to_csv(destination, index=False)
    stats = summarise(report)
    print(json.dumps(stats, indent=2))
    print(f"\nwrote {destination}")
    for row in report[~report["ok"]].itertuples(index=False):
        print(f"  [FAIL] {row.clip}: {row.reason}")
    if args.fail_over is not None and stats.get("clips"):
        rate = 100 - stats["pass_rate"]
        if rate > args.fail_over:
            raise SystemExit(f"failure rate {rate:.1f}% exceeds --fail-over {args.fail_over}%")


if __name__ == "__main__":
    main()
