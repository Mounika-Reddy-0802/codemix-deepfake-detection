"""Confusion matrices and threshold metrics for every scored run.

The project reports EER and AUC because they are threshold-free, and a viva
audience asks for accuracy, precision and recall instead. Those four numbers only
mean something once three things are stated, so this module always states them:

1. **Which class is positive.** Here it is *spoof*, because the question the system
   answers is "did we catch the clone?". Recall is therefore the fraction of cloned
   clips flagged, and specificity is the fraction of real speech left alone.
2. **The threshold.** A confusion matrix is a slice through the score distribution.
   Every run is reported twice: at the deployed operating point from
   ``configs/threshold.yaml``, and at that run's own EER threshold.
3. **The class balance.** Accuracy on a set that is 80% spoof is not comparable to
   accuracy on a balanced one, so the counts are always carried alongside.

Scores are ``P(bonafide)``: a clip is called spoof when its score is *below* the
threshold. Nothing here re-scores audio; it reads the per-clip CSVs written by
``src.training.evaluate``.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

#: Directories holding ``<run>__<set>_scores.csv`` files, in reporting order.
SCORE_DIRS: tuple[str, ...] = (
    "results/w10",
    "docs/results/s3_native",
    "results/s1_norm",
)

OUT_JSON = "experiments/results/classification_metrics.json"

#: run key -> how it is named in the write-up. Runs not listed are reported under
#: their own key rather than dropped, so a new run never goes missing silently.
RUN_LABELS: dict[str, str] = {
    "stage1": "S1 baseline (English-only)",
    "lora_norm_clean": "S2 LoRA, XTTS, clean",
    "lora_norm_channel": "S2 LoRA, XTTS, channel",
    "lora_norm_rvc_clean": "S2 LoRA, XTTS+RVC, clean",
    "lora_norm_rvc_channel": "S2 LoRA, XTTS+RVC, channel (deployed)",
    "s3_native_clean": "S3 native, XTTS, clean",
    "s3_native_channel": "S3 native, XTTS, channel",
    "s3_native_rvc_clean": "S3 native, XTTS+RVC, clean",
    "s3_native_rvc_channel": "S3 native, XTTS+RVC, channel",
}

SET_LABELS: dict[str, str] = {
    "eval_pool": "seen tool (XTTS)",
    "rvc_test": "RVC, unseen speakers",
    "cm04": "unseen tool (Tortoise)",
    "english": "English (ASVspoof LA)",
    "asvspoof_eval": "English (ASVspoof LA)",
    "eval_pool_channel": "seen tool (XTTS), channel",
    "rvc_test_channel": "RVC, unseen speakers, channel",
    "cm04_channel": "unseen tool (Tortoise), channel",
}


@dataclass(frozen=True)
class Confusion:
    """Counts with *spoof* as the positive class."""

    tp: int  # cloned, flagged
    fn: int  # cloned, missed
    tn: int  # real, passed
    fp: int  # real, wrongly flagged

    @property
    def total(self) -> int:
        return self.tp + self.fn + self.tn + self.fp

    @property
    def spoof(self) -> int:
        return self.tp + self.fn

    @property
    def bonafide(self) -> int:
        return self.tn + self.fp


def confusion_at(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Confusion:
    """Counts at one operating point. ``labels``: 1 bonafide, 0 spoof."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.size != scores.size:
        raise ValueError("labels and scores must be the same length")
    called_spoof = scores < threshold
    is_spoof = labels == 0
    return Confusion(
        tp=int(np.sum(called_spoof & is_spoof)),
        fn=int(np.sum(~called_spoof & is_spoof)),
        tn=int(np.sum(~called_spoof & ~is_spoof)),
        fp=int(np.sum(called_spoof & ~is_spoof)),
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    """A rate, or ``None`` when the denominator is empty -- never a silent zero."""
    return float(numerator) / denominator if denominator else None


def metrics_from(counts: Confusion) -> dict[str, float | None]:
    """Accuracy, precision, recall, specificity, F1 and balanced accuracy."""
    precision = _ratio(counts.tp, counts.tp + counts.fp)
    recall = _ratio(counts.tp, counts.spoof)
    specificity = _ratio(counts.tn, counts.bonafide)
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    balanced = None
    if recall is not None and specificity is not None:
        balanced = (recall + specificity) / 2
    return {
        "accuracy": _ratio(counts.tp + counts.tn, counts.total),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced,
        "false_alarm_rate": None if specificity is None else 1.0 - specificity,
        "miss_rate": None if recall is None else 1.0 - recall,
    }


def read_scores(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """``(labels, scores)`` from a scored CSV. 1 = bonafide, 0 = spoof."""
    labels: list[int] = []
    scores: list[float] = []
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            raw = (row.get("score") or "").strip()
            if not raw:  # a clip that produced no window is not a prediction
                continue
            labels.append(1 if row["label"].strip().lower() == "bonafide" else 0)
            scores.append(float(raw))
    return np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=np.float64)


def split_name(filename: str) -> tuple[str, str]:
    """``lora_norm_channel__cm04_scores.csv`` -> ``("lora_norm_channel", "cm04")``."""
    stem = Path(filename).name
    for suffix in ("_scores.csv", ".csv"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    run, _, evaluation = stem.partition("__")
    return run, (evaluation or "unknown")


def find_runs(dirs: tuple[str, ...] = SCORE_DIRS, root: str | Path = ".") -> list[Path]:
    """Every scored CSV under the reporting directories, in a stable order."""
    base = Path(root)
    found: list[Path] = []
    for directory in dirs:
        folder = base / directory
        if folder.is_dir():
            found.extend(sorted(folder.glob("*_scores.csv")))
    return found


def report_for(path: str | Path, threshold: float) -> dict:
    """Both operating points for one scored run."""
    from src.training.metrics import compute_eer, roc_auc

    run, evaluation = split_name(path)
    labels, scores = read_scores(path)
    entry: dict = {
        "run": run,
        "run_label": RUN_LABELS.get(run, run),
        "set": evaluation,
        "set_label": SET_LABELS.get(evaluation, evaluation),
        "source": str(Path(path).as_posix()),
        "clips": int(labels.size),
        "bonafide": int(np.sum(labels == 1)),
        "spoof": int(np.sum(labels == 0)),
    }
    if entry["bonafide"] == 0 or entry["spoof"] == 0:
        entry["error"] = "one class only: a confusion matrix would be meaningless"
        return entry

    eer_value, eer_threshold = compute_eer(scores, labels)
    entry["eer"] = float(eer_value)
    entry["auc"] = float(roc_auc(scores, labels))
    entry["eer_threshold"] = float(eer_threshold)
    # "eer" already holds the EER value, so the operating point gets its own key.
    for name, point in (("deployed", threshold), ("eer_point", eer_threshold)):
        counts = confusion_at(labels, scores, point)
        entry[name] = {
            "threshold": float(point),
            "confusion": asdict(counts),
            **metrics_from(counts),
        }
    return entry


def build(threshold: float | None = None, root: str | Path = ".") -> dict:
    """Every scored run, at the deployed point and at its own EER point."""
    if threshold is None:
        from src.inference.predict import resolve_threshold

        threshold = resolve_threshold(
            threshold_file=str(Path(root) / "configs" / "threshold.yaml")
        ).value
    runs = [report_for(path, threshold) for path in find_runs(root=root)]
    return {
        "positive_class": "spoof",
        "rule": "a clip is called spoof when P(bonafide) < threshold",
        "deployed_threshold": float(threshold),
        "note": (
            "accuracy depends on the class balance of each set, which is given per "
            "row; EER and AUC are the threshold-free numbers the paper reports"
        ),
        "runs": runs,
    }


def markdown(report: dict, point: str = "deployed") -> str:
    """The table as markdown, one row per run and set."""
    header = (
        "| Model | Test set | Real | Cloned | TP | FN | TN | FP | "
        "Accuracy | Precision | Recall | Specificity | F1 | EER |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    rows = []
    for entry in report["runs"]:
        if "error" in entry:
            continue
        block = entry[point]
        counts = block["confusion"]

        def pct(value: float | None) -> str:
            return "—" if value is None else f"{value * 100:.1f}%"

        rows.append(
            f"| {entry['run_label']} | {entry['set_label']} | {entry['bonafide']} | "
            f"{entry['spoof']} | {counts['tp']} | {counts['fn']} | {counts['tn']} | "
            f"{counts['fp']} | {pct(block['accuracy'])} | {pct(block['precision'])} | "
            f"{pct(block['recall'])} | {pct(block['specificity'])} | "
            f"{pct(block['f1'])} | {entry['eer'] * 100:.2f}% |"
        )
    return header + "\n".join(rows) + "\n"


def main() -> None:
    """Write the metrics JSON and print the deployed-point table."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=OUT_JSON)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--point", choices=["deployed", "eer_point"], default="deployed")
    args = parser.parse_args()

    report = build(args.threshold)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(markdown(report, args.point))
    print(f"wrote {out}  ({len(report['runs'])} runs)")


if __name__ == "__main__":
    main()
