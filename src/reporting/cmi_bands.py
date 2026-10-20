"""False alarms by Code-Mixing Index on the original test pool.

The language-control report compares conditions within the same speakers. This one
asks the same question on the evaluation pool the paper's tables use: does the share
of genuine clips a model rejects rise with the amount of code-mixing in the clip?

Each genuine clip gets its CMI (Das and Gambaeck, 2014; ``src.data.codemixing.cmi``)
from its transcript in the clip index, and the clips are grouped into four bands
(0-10, 10-25, 25-40, 40-50). Reported per band: clips, speakers, false-alarm rate at
P(bonafide) < 0.5 with a speaker-bootstrap 95% interval; and per model the Spearman
correlation between CMI and score over all genuine clips.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.codemixing import cmi

THRESHOLD = 0.5
BANDS = [-0.1, 10, 25, 40, 50.1]
LABELS = ["0-10", "10-25", "25-40", "40-50"]
SCORES = {  # name -> per-clip scores on the evaluation pool
    "S1 (English only), clean audio": "results/s1_norm/stage1__eval_pool_scores.csv",
    "S1 (English only), phone line": "results/s1_norm/stage1__eval_pool_channel_scores.csv",
    "deployed S2, phone line": "results/w10/lora_norm_rvc_channel__eval_pool_scores.csv",
}


def with_cmi(scores: pd.DataFrame, clip_index: pd.DataFrame) -> pd.DataFrame:
    """Genuine rows with ``cmi`` and ``band``; rows without a transcript are dropped."""
    genuine = scores[scores["label"].str.lower() == "bonafide"]
    text = clip_index[["utt_id", "transcript"]].dropna().drop_duplicates("utt_id")
    out = genuine.merge(text, on="utt_id", how="inner")
    out["cmi"] = out["transcript"].map(cmi)
    out["band"] = pd.cut(out["cmi"], BANDS, labels=LABELS)
    out["false_alarm"] = out["score"] < THRESHOLD
    return out


def band_table(df: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for band in LABELS:
        part = df[df["band"] == band]
        if part.empty:
            continue
        flags = part.groupby("speaker")["false_alarm"].agg(["sum", "size"])
        idx = rng.integers(0, len(flags), size=(n_boot, len(flags)))
        boot = flags["sum"].to_numpy()[idx].sum(1) / flags["size"].to_numpy()[idx].sum(1)
        rows.append(
            {
                "band": band,
                "clips": int(len(part)),
                "speakers": int(part["speaker"].nunique()),
                "false_alarm_rate": float(part["false_alarm"].mean()),
                "ci_low": float(np.percentile(boot, 2.5)),
                "ci_high": float(np.percentile(boot, 97.5)),
            }
        )
    return pd.DataFrame(rows)


def analyse(repo: str, clip_index: pd.DataFrame) -> dict:
    report = {}
    for name, rel in SCORES.items():
        path = Path(repo) / rel
        if not path.is_file():
            continue
        df = with_cmi(pd.read_csv(path, dtype={"speaker": str}), clip_index)
        report[name] = {
            "genuine_clips": int(len(df)),
            "spearman_cmi_score": float(df[["cmi", "score"]].corr(method="spearman").iloc[0, 1]),
            "bands": band_table(df).to_dict("records"),
        }
    return report


def markdown(report: dict) -> str:
    lines = [
        "# False alarms by Code-Mixing Index on the evaluation pool (v1)",
        "",
        "Genuine clips of the 15 evaluation-pool speakers, grouped by the CMI of their",
        "transcript (0 = one language, 50 = an even mix). A false alarm is P(bonafide) < 0.5;",
        "intervals are 95% speaker-bootstrap intervals. Built by",
        "`src/reporting/cmi_bands.py`.",
        "",
        "| Model | " + " | ".join(f"CMI {b}" for b in LABELS) + " | Spearman(CMI, score) |",
        "|---|" + "---|" * (len(LABELS) + 1),
    ]
    for name, entry in report.items():
        cells = {
            b["band"]: f"{100 * b['false_alarm_rate']:.1f}% "
            f"[{100 * b['ci_low']:.1f}, {100 * b['ci_high']:.1f}] (n={b['clips']})"
            for b in entry["bands"]
        }
        lines.append(
            f"| {name} | "
            + " | ".join(cells.get(b, "-") for b in LABELS)
            + f" | {entry['spearman_cmi_score']:+.3f} |"
        )
    lines += [
        "",
        "A detector that confused code-mixing with spoofing would show false alarms",
        "rising across the bands and a negative correlation between CMI and its",
        "P(bonafide). Flat bands and a correlation near zero point to the recording",
        "domain instead (see `language_control_v1.md`).",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="False alarms by CMI band")
    parser.add_argument("--clip-index", default="data/manifests/clip_index.csv")
    parser.add_argument("--out", default="experiments/results/cmi_bands_v1.json")
    parser.add_argument("--doc", default="docs/results/cmi_bands_v1.md")
    args = parser.parse_args()
    index = pd.read_csv(args.clip_index, usecols=["utt_id", "transcript"])
    report = analyse(".", index)
    Path(args.out).write_text(json.dumps(report, indent=2))
    Path(args.doc).write_text(markdown(report), encoding="utf-8")
    print(markdown(report))


if __name__ == "__main__":
    main()
