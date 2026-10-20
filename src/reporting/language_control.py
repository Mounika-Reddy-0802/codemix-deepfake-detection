"""Code-mixing or recording domain? The language-control false-alarm report.

The control set (``src.data.codemixing.control_manifest``) holds genuine lecture clips
of 183 speakers who were never trained or tested on, each speaker contributing
all-English, all-Hindi and mixed clips from the same recordings. If code-mixing
itself fooled a detector, its false alarms would be highest on the mixed clips; if the
recording domain is what matters, all three conditions would look alike.

Two views per model:

* the false-alarm rate in each condition, with a 95% interval from resampling whole
  speakers (clips of one speaker are not independent);
* the paired, within-speaker difference to all-English (mixed - English and Hindi -
  English), averaged over the speakers who have both, with a speaker-bootstrap
  interval. Pairing removes the speaker and the recording from the comparison.

A false alarm is a genuine clip whose P(bonafide) falls below 0.5, the deployed
operating point (``src.training.false_alarm``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

THRESHOLD = 0.5
CONDITIONS = ("all English", "all Hindi", "mixed")
MODELS = {  # file stem -> how the report names it
    "s1__clean": "S1 (English only), clean audio",
    "s1__channel": "S1 (English only), phone line",
    "s2_rvc_clean__clean": "S2 trained clean, clean audio",
    "deployed_s2__channel": "deployed S2, phone line",
}


def load(results_dir: str) -> dict[str, pd.DataFrame]:
    """Per-clip scores of every model in ``MODELS`` found in ``results_dir``."""
    out = {}
    for stem in MODELS:
        path = Path(results_dir) / f"{stem}_scores.csv"
        if path.is_file():
            df = pd.read_csv(path, dtype={"speaker": str})
            df["false_alarm"] = df["score"] < THRESHOLD
            out[stem] = df
    return out


def _speaker_boot(per_speaker: pd.Series, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    values = per_speaker.to_numpy(dtype=float)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    return values[idx].mean(axis=1)


def condition_rates(df: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """False-alarm rate per condition, with a speaker-bootstrap interval."""
    rng = np.random.default_rng(seed)
    rows = []
    for cond in CONDITIONS:
        part = df[df["language_condition"] == cond]
        if part.empty:
            continue
        # Clip-weighted rate; the interval resamples speakers with their clip counts.
        flags = part.groupby("speaker")["false_alarm"].agg(["sum", "size"])
        idx = rng.integers(0, len(flags), size=(n_boot, len(flags)))
        sums, sizes = flags["sum"].to_numpy()[idx].sum(1), flags["size"].to_numpy()[idx].sum(1)
        boot = sums / sizes
        rows.append(
            {
                "condition": cond,
                "clips": int(len(part)),
                "speakers": int(part["speaker"].nunique()),
                "false_alarm_rate": float(part["false_alarm"].mean()),
                "ci_low": float(np.percentile(boot, 2.5)),
                "ci_high": float(np.percentile(boot, 97.5)),
            }
        )
    return pd.DataFrame(rows)


def paired_differences(df: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Within-speaker difference of each condition to all-English."""
    rng = np.random.default_rng(seed)
    per = df.pivot_table(
        index="speaker", columns="language_condition", values="false_alarm", aggfunc="mean"
    )
    rows = []
    for cond in ("mixed", "all Hindi"):
        if cond not in per or "all English" not in per:
            continue
        both = (per[cond] - per["all English"]).dropna()
        if both.empty:
            continue
        boot = _speaker_boot(both, n_boot, rng)
        rows.append(
            {
                "comparison": f"{cond} - all English",
                "speakers": int(len(both)),
                "mean_difference": float(both.mean()),
                "ci_low": float(np.percentile(boot, 2.5)),
                "ci_high": float(np.percentile(boot, 97.5)),
            }
        )
    return pd.DataFrame(rows)


def analyse(results_dir: str) -> dict:
    """Both views for every model found."""
    report = {}
    for stem, df in load(results_dir).items():
        report[stem] = {
            "name": MODELS[stem],
            "conditions": condition_rates(df).to_dict("records"),
            "paired": paired_differences(df).to_dict("records"),
        }
    return report


def markdown(report: dict) -> str:
    lines = [
        "# Code-mixing or recording domain? Language-control false alarms (v1)",
        "",
        "Genuine clips of 183 speakers who were never trained or tested on; every speaker",
        "contributes all-English, all-Hindi and mixed clips from the same lecture recordings.",
        "A false alarm is a genuine clip scored below P(bonafide) = 0.5. Intervals are 95%",
        "speaker-bootstrap intervals. Built by `src/reporting/language_control.py` from",
        "`experiments/results/language_control_v1/`.",
        "",
        "## False-alarm rate by language condition",
        "",
        "| Model | all English | all Hindi | mixed |",
        "|---|---|---|---|",
    ]
    for entry in report.values():
        cells = {
            c["condition"]: f"{100 * c['false_alarm_rate']:.1f}% "
            f"[{100 * c['ci_low']:.1f}, {100 * c['ci_high']:.1f}]"
            for c in entry["conditions"]
        }
        lines.append(
            f"| {entry['name']} | " + " | ".join(cells.get(c, "-") for c in CONDITIONS) + " |"
        )
    lines += [
        "",
        "## Within-speaker difference to all-English (percentage points)",
        "",
        "| Model | mixed - English | Hindi - English |",
        "|---|---|---|",
    ]
    for entry in report.values():
        cells = {
            p["comparison"].split(" - ")[0]: f"{100 * p['mean_difference']:+.1f} "
            f"[{100 * p['ci_low']:+.1f}, {100 * p['ci_high']:+.1f}] (n={p['speakers']})"
            for p in entry["paired"]
        }
        lines.append(
            f"| {entry['name']} | {cells.get('mixed', '-')} | {cells.get('all Hindi', '-')} |"
        )
    lines += [
        "",
        "## Reading",
        "",
        "If code-mixing drove the false alarms, the mixed column would stand out and the",
        "mixed - English difference would be clearly positive. An interval that contains",
        "zero, or a negative difference, means the model treats mixed speech no worse than",
        "the same speaker's single-language speech: the rejections come from the",
        "recording domain, not from the language mix.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Language-control false-alarm report")
    parser.add_argument("--results", default="experiments/results/language_control_v1")
    parser.add_argument("--doc", default="docs/results/language_control_v1.md")
    args = parser.parse_args()
    report = analyse(args.results)
    Path(args.results, "summary.json").write_text(json.dumps(report, indent=2))
    Path(args.doc).write_text(markdown(report), encoding="utf-8")
    print(markdown(report))


if __name__ == "__main__":
    main()
