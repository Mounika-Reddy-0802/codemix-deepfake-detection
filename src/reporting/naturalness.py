"""How natural are the clones? Predicted MOS for every attack set and its genuine set.

A detector result means more when the fakes are good. Listening tests do not scale
to thousands of clips, so naturalness is estimated with UTMOS (``utmos22_strong``), a
predictor of the 1-5 mean opinion score trained on listening-test data, run on the
clean, level-normalised clips. A fixed random sample per set (seeded) keeps the cost
small; the report gives each set's mean with a bootstrap interval, beside the genuine
speech of the same speakers, so a clone set can be read as "x below the real thing".

UTMOS was trained on English synthetic speech, so absolute values on Hindi-English
audio are indicative only; the comparison between sets of the same speakers is the
reading this report supports.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

SETS = {  # name -> (manifest stem, label kept, tool kept or None)
    "genuine, new test speakers": ("eval_ext_genuine", "bonafide", None),
    "CM05 XTTS-v2 (new speakers)": ("cm05", "spoof", "xtts_v2"),
    "CM06 Tortoise (new speakers)": ("cm06", "spoof", "tortoise"),
    "genuine, IndicF5 speakers (all Hindi)": ("indicf5_genuine", "bonafide", None),
    "CM07 IndicF5": ("cm07", "spoof", "indicf5"),
    "CM08 XTTS-v2 (adaptation speakers)": ("cm08", "spoof", "xtts_v2"),
}


def sample(frame: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """At most ``n`` rows, drawn without replacement with a fixed seed."""
    if len(frame) <= n:
        return frame.reset_index(drop=True)
    return frame.sample(n=n, random_state=seed).reset_index(drop=True)


def summarise(scores: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    boot = rng.choice(scores, size=(n_boot, len(scores)), replace=True).mean(axis=1)
    return {
        "clips": int(len(scores)),
        "mos_mean": float(scores.mean()),
        "mos_sd": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
    }


def utmos() -> Callable[[np.ndarray, int], float]:
    """The UTMOS22 strong predictor as ``(waveform, sample_rate) -> MOS``."""
    import torch

    model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).eval()

    def predict(wave: np.ndarray, sr: int) -> float:
        with torch.no_grad():
            return float(model(torch.from_numpy(wave.astype(np.float32))[None], sr))

    return predict


def run(manifests: str, data_root: str, n: int, predict, seed: int = 0) -> dict:
    import soundfile as sf

    report = {}
    for name, (stem, label, tool) in SETS.items():
        path = Path(manifests) / f"{stem}.csv"
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        frame = frame[frame["label"] == label]
        if tool is not None:
            frame = frame[frame["tool"] == tool]
        chosen = sample(frame, n, seed)
        values = []
        for fp in chosen["filepath"]:
            wave, sr = sf.read(str(fp).replace("${DATA_ROOT}", data_root), dtype="float32")
            values.append(predict(wave, sr))
        report[name] = summarise(np.asarray(values))
        print(f"{name:42s} MOS {report[name]['mos_mean']:.2f} (n={len(values)})", flush=True)
    return report


def markdown(report: dict) -> str:
    lines = [
        "# How natural are the clones? Predicted MOS (v1)",
        "",
        "UTMOS22 (strong) predicted mean opinion score, 1-5, on a seeded random sample of",
        "clean, level-normalised clips per set; 95% bootstrap interval of the mean. UTMOS",
        "was trained on English, so compare sets of the same speakers rather than reading",
        "absolute values. Built by `src/reporting/naturalness.py`.",
        "",
        "| Set | Clips | MOS mean [95% CI] | SD |",
        "|---|---|---|---|",
    ]
    for name, r in report.items():
        lines.append(
            f"| {name} | {r['clips']} | {r['mos_mean']:.2f} "
            f"[{r['ci_low']:.2f}, {r['ci_high']:.2f}] | {r['mos_sd']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Predicted MOS per attack set")
    parser.add_argument("--manifests", default="data/manifests")
    parser.add_argument("--data-root", required=True, help="clean bundle holding every set")
    parser.add_argument("--per-set", type=int, default=200)
    parser.add_argument("--out", default="experiments/results/naturalness_v1.json")
    parser.add_argument("--doc", default="docs/results/naturalness_v1.md")
    args = parser.parse_args()
    report = run(args.manifests, args.data_root, args.per_set, utmos())
    Path(args.out).write_text(json.dumps(report, indent=2))
    Path(args.doc).write_text(markdown(report), encoding="utf-8")
    print(markdown(report))


if __name__ == "__main__":
    main()
