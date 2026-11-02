"""DET curves and calibration of the final systems, from their per-clip scores.

Every result document so far quotes the equal error rate, which hides two things a
deployment depends on: how errors trade off away from the equal-error point (a DET
curve), and whether a score can be read as a probability (calibration). This module
adds both for the deployed adapter, the final adapter and SSL-AASIST over every
training seed and test set, and for S1 against the adapters on English.

Per system, seed and set it reports:

- the EER, recomputed with :func:`src.training.metrics.eer` (identical to the
  committed summaries; the tests and the CLI check this);
- the Brier score and the expected calibration error (10 equal-width bins) of
  P(genuine);
- accuracy, false-alarm rate (genuine called fake) and miss rate (fake called genuine)
  at the fixed clip rule every result document uses: fake when P(genuine) < 0.5. No
  threshold is chosen here.

The adapters emit P(genuine); SSL-AASIST emits log-odds, which are mapped through the
logistic function, so the rule is the same (log-odds below 0). The EER does not depend
on that mapping.

The per-clip scores are the unzipped Kaggle outputs (not committed: tens of
megabytes, one file per seed and set), so the CLI takes their root::

    python -m src.reporting.det_curves --scores-root D:/dfdata/gen_results

and writes ``experiments/results/calibration_v1.json`` and
``experiments/figures/det_final_systems.{png,pdf}``. Matplotlib and scipy are imported
lazily so the module stays importable in CI.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.reporting.figures import FIGURES_DIR, det_points
from src.training.metrics import eer

OUT_JSON = "experiments/results/calibration_v1.json"
ENGLISH_DIR = "experiments/results/english_channel_v1"
RULE = 0.5  # fake when P(genuine) < RULE: the clip rule of every result document
BINS = 10

#: test set key -> label, in the order of the results tables
SETS = {
    "eval_pool": "Original test pool (XTTS-v2), G.711",
    "new_speakers": "New speakers (XTTS-v2, Tortoise), G.711",
    "new_speakers_amr": "New speakers (XTTS-v2, Tortoise), AMR-NB",
    "cm04": "CM04 Tortoise, G.711",
    "cm04_amr": "CM04 Tortoise, AMR-NB",
    "cm07": "IndicF5, G.711",
    "cm07_amr": "IndicF5, AMR-NB",
    "cm09": "Chatterbox, G.711",
    "rvc_test": "RVC test half, G.711",
    "control": "Language control (genuine only), G.711",
    "english_clean": "English (ASVspoof sample), clean",
    "english_channel": "English (ASVspoof sample), G.711",
}


@dataclass(frozen=True)
class Source:
    """Where one system's per-clip score files are, relative to the scores root."""

    system: str
    folder: str
    prefix: str  # file names are <prefix><seed>__<set>_scores.csv


SOURCES = (
    Source("Deployed adapter", "final_x/final_results", "deployed_s"),
    Source("Final adapter", "final_x/final_results", "final_s"),
    Source("SSL-AASIST", "final_ssl_x/final_ssl_results", "final_s"),
    Source("SSL-AASIST", "run8_x/final_ssl_seeds2_results", "final_s"),
)

#: English scores are committed (one training each): system -> file prefix
ENGLISH = {
    "S1 (English only)": "s1",
    "Deployed adapter (published)": "deployed_s2",
    "Final adapter": "final_s1234",
}


def probability(scores: np.ndarray) -> np.ndarray:
    """P(genuine): scores already in [0, 1] are kept, log-odds go through the logistic."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.min() >= 0.0 and scores.max() <= 1.0:
        return scores
    return 1.0 / (1.0 + np.exp(-scores))


def calibration_error(p: np.ndarray, labels: np.ndarray, bins: int = BINS) -> float:
    """Expected calibration error of P(genuine) over equal-width bins."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        inside = which == b
        if inside.any():
            total += inside.mean() * abs(p[inside].mean() - labels[inside].mean())
    return float(total)


def clip_metrics(scores: np.ndarray, labels: np.ndarray, rule: float = RULE) -> dict:
    """All metrics for one score file; ``labels`` is 1 for genuine."""
    labels = np.asarray(labels).astype(int)
    p = probability(scores)
    called_genuine = p >= rule
    genuine, fake = labels == 1, labels == 0
    both = genuine.any() and fake.any()
    return {
        "clips": int(len(labels)),
        "genuine": int(genuine.sum()),
        "fake": int(fake.sum()),
        "eer": float(eer(np.asarray(scores, dtype=np.float64), labels)) * 100 if both else None,
        "brier": float(np.mean((p - labels) ** 2)),
        "ece": calibration_error(p, labels),
        "accuracy": float(np.mean(called_genuine == genuine)) * 100,
        "false_alarm": float(np.mean(~called_genuine[genuine])) * 100 if genuine.any() else None,
        "miss": float(np.mean(called_genuine[fake])) * 100 if fake.any() else None,
    }


def read_scores(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Scores and labels (1 = genuine) of one per-clip score CSV."""
    frame = pd.read_csv(path, usecols=["label", "score"])
    labels = (frame["label"].astype(str) == "bonafide").to_numpy().astype(int)
    return frame["score"].to_numpy(dtype=np.float64), labels


def score_files(root: str | Path, sources=SOURCES) -> pd.DataFrame:
    """One row per (system, seed, set) score file found under ``root``."""
    rows = []
    for src in sources:
        folder = Path(root) / src.folder
        pattern = re.compile(rf"^{re.escape(src.prefix)}(\d+)__(\w+?)_scores\.csv$")
        for path in sorted(folder.glob(f"{src.prefix}*__*_scores.csv")):
            match = pattern.match(path.name)
            if match and match.group(2) in SETS:
                rows.append(
                    {
                        "system": src.system,
                        "seed": int(match.group(1)),
                        "set": match.group(2),
                        "path": str(path),
                    }
                )
    return pd.DataFrame(rows, columns=["system", "seed", "set", "path"])


def english_files(folder: str | Path = ENGLISH_DIR) -> pd.DataFrame:
    """The committed English score files (one training per system)."""
    rows = []
    for system, prefix in ENGLISH.items():
        for condition in ("english_clean", "english_channel"):
            path = Path(folder) / f"{prefix}__{condition}_scores.csv"
            if path.is_file():
                seed = int(prefix.rsplit("_s", 1)[1]) if "_s" in prefix else 0
                rows.append({"system": system, "seed": seed, "set": condition, "path": str(path)})
    return pd.DataFrame(rows, columns=["system", "seed", "set", "path"])


def measure(files: pd.DataFrame) -> pd.DataFrame:
    """Add the metrics of every file in ``files`` as columns."""
    metrics = [clip_metrics(*read_scores(path)) for path in files["path"]]
    return pd.concat([files.reset_index(drop=True), pd.DataFrame(metrics)], axis=1)


def summarise(per_file: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation over seeds, per system and set."""
    values = ["eer", "brier", "ece", "accuracy", "false_alarm", "miss"]
    grouped = per_file.groupby(["system", "set"], sort=False)
    table = grouped[values].agg(["mean", "std"])
    table.columns = [f"{name}_{stat}" for name, stat in table.columns]
    table["seeds"] = grouped["seed"].nunique()
    return table.reset_index()


def markdown(summary: pd.DataFrame) -> str:
    """The summary as a markdown table (mean ± sd over seeds)."""

    def cell(row: pd.Series, name: str, digits: int) -> str:
        mean, sd = row[f"{name}_mean"], row[f"{name}_std"]
        if pd.isna(mean):
            return "-"
        return f"{mean:.{digits}f}" + (f" ± {sd:.{digits}f}" if pd.notna(sd) else "")

    lines = [
        "| Set | System | Seeds | EER (%) | Brier | ECE | Accuracy (%) | False alarm (%) | Miss (%) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = {key: i for i, key in enumerate(SETS)}
    for _, row in summary.sort_values("set", key=lambda s: s.map(order), kind="stable").iterrows():
        lines.append(
            f"| {SETS[row['set']]} | {row['system']} | {int(row['seeds'])} | "
            f"{cell(row, 'eer', 2)} | {cell(row, 'brier', 3)} | {cell(row, 'ece', 3)} | "
            f"{cell(row, 'accuracy', 1)} | {cell(row, 'false_alarm', 1)} | {cell(row, 'miss', 1)} |"
        )
    return "\n".join(lines)


#: the three panels of the figure: (title, set key, files to draw)
PANELS = (
    ("(a) New speakers, AMR-NB", "new_speakers_amr"),
    ("(b) IndicF5, G.711", "cm07"),
    ("(c) English, G.711", "english_channel"),
)

#: one colour per system in every panel; the published deployed checkpoint shares the
#: deployed adapter's colour
COLOURS = {
    "S1 (English only)": "#7f7f7f",
    "Deployed adapter": "#1f77b4",
    "Deployed adapter (published)": "#1f77b4",
    "Final adapter": "#ff7f0e",
    "SSL-AASIST": "#2ca02c",
}


def det_figure(files: pd.DataFrame, out_dir: str = FIGURES_DIR, seed: int = 1234) -> str:
    """DET curves on normal-deviate axes, one panel per set, one curve per system.

    The Kaggle panels use one seed (``seed``) of each system; the English panel uses the
    single committed training of each system.
    """
    from scipy.stats import norm

    from src.reporting.figures import _style

    plt = _style()
    ticks = [0.1, 0.5, 2, 5, 10, 20, 40]
    low, high = 0.05, 60.0  # axis range in %, so a 0.2% EER curve is still visible
    fig, axes = plt.subplots(1, len(PANELS), figsize=(7.2, 2.9), sharey=True)
    handles = {}
    for ax, (title, key) in zip(axes, PANELS, strict=True):
        part = files[files["set"] == key]
        if key != "english_channel":
            part = part[part["seed"] == seed]
        for _, row in part.drop_duplicates("system").iterrows():
            scores, labels = read_scores(row["path"])
            fa, miss = det_points(scores, labels)
            # zero rates have no normal deviate: thresholds below every genuine clip are
            # left out, and a zero miss rate is drawn on the axis floor
            keep = (fa > 0) & (fa < high) & (miss < high)
            miss = np.clip(miss, low, None)
            (line,) = ax.plot(
                norm.ppf(fa[keep] / 100),
                norm.ppf(miss[keep] / 100),
                lw=1.3,
                color=COLOURS.get(row["system"]),
                label=row["system"].replace(" (published)", ""),
            )
            handles.setdefault(line.get_label(), line)
        ax.plot(norm.ppf([low / 100, 0.6]), norm.ppf([low / 100, 0.6]), ls=":", lw=0.8, c="grey")
        ax.set_xticks(norm.ppf(np.array(ticks) / 100), [str(t) for t in ticks], fontsize=7)
        ax.set_yticks(norm.ppf(np.array(ticks) / 100), [str(t) for t in ticks], fontsize=7)
        ax.set_xlim(norm.ppf(low / 100), norm.ppf(0.5))
        ax.set_ylim(norm.ppf(low / 100), norm.ppf(0.5))
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("False alarm (%)", fontsize=8)
    axes[0].set_ylabel("Miss (%)", fontsize=8)
    order = ["S1 (English only)", "Deployed adapter", "Final adapter", "SSL-AASIST"]
    names = [n for n in order if n in handles]
    fig.legend(
        [handles[n] for n in names],
        names,
        loc="lower center",
        ncol=len(names),
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, -0.12),
    )
    out = Path(out_dir) / "det_final_systems.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return str(out)


def main() -> None:
    """CLI: measure every score file, write the JSON and the figure, print the table."""
    import argparse

    parser = argparse.ArgumentParser(description="DET curves and calibration")
    parser.add_argument("--scores-root", required=True, help="where the Kaggle zips are unpacked")
    parser.add_argument("--out", default=OUT_JSON)
    parser.add_argument("--figures", default=FIGURES_DIR)
    args = parser.parse_args()

    files = pd.concat([score_files(args.scores_root), english_files()], ignore_index=True)
    if files.empty:
        raise SystemExit(f"no score files under {args.scores_root}")
    per_file = measure(files)
    summary = summarise(per_file)
    payload = {
        "rule": f"fake when P(genuine) < {RULE}",
        "ece_bins": BINS,
        "summary": json.loads(summary.to_json(orient="records")),
        "per_file": json.loads(per_file.drop(columns="path").to_json(orient="records")),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(markdown(summary))
    print(det_figure(files, args.figures))


if __name__ == "__main__":
    main()
