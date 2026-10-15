"""Seed stability of the phone-line systems (external review comment 1).

Every published number came from one training run with seed 1234, and the only
seed repeat ever made was of a different model. The review asked for the
channel-matched results across seeds, with intervals. This module reads the seed
repeats trained by ``notebooks/kaggle_w12_review_runs.ipynb`` and reports, for each
system and test set:

- the EER of every seed, their mean, standard deviation and range, and a 95%
  t-interval for the mean (the uncertainty that comes from training, which the
  clip-level bootstrap intervals in the paper never covered);
- for every seed, the paired, speaker-level comparison with the matched shortcut
  floor (``src.training.floor_compare``), so a claim such as "beats its floor" can be
  stated as "in k of n seeds" rather than resting on one run.

The matched floor depends only on the training data and the test clips, so every
seed of a system is compared with the same floor scores
(``experiments/results/floor_v2/<run>__<set>_paired.csv``).

Pure numpy/pandas, so CI runs it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.training.floor_compare import binary_labels, speaker_bootstrap, verdict

SETS = ("eval_pool", "cm04", "rvc_test")
FLOOR_DIR = "experiments/results/floor_v2"

#: Two-sided 97.5% Student-t quantiles by degrees of freedom (no scipy in CI).
T975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}


@dataclass(frozen=True)
class SeededSystem:
    """A system repeated over seeds: where its published run and floor live."""

    key: str  # the ``system`` field in runs.json
    label: str
    published_prefix: str  # "<prefix>__<set>.json" / "_scores.csv" of the seed-1234 run
    floor_run: str  # "<floor_run>__<set>_paired.csv" in FLOOR_DIR
    rerun_1234: bool  # True when the notebook re-trained seed 1234 itself


SYSTEMS: tuple[SeededSystem, ...] = (
    SeededSystem(
        "deployed",
        "S2 LoRA, XTTS+RVC, channel (deployed)",
        "results/w10/lora_norm_rvc_channel",
        "lora_norm_rvc_channel",
        True,
    ),
    SeededSystem(
        "xtts_channel",
        "S2 LoRA, XTTS, channel",
        "results/w10/lora_norm_channel",
        "lora_norm_channel",
        False,
    ),
    SeededSystem(
        "s3_xtts_channel",
        "S3 native, XTTS, channel",
        "docs/results/s3_native/s3_native_channel",
        "s3_native_channel",
        False,
    ),
)


def mean_interval(values: list[float]) -> dict:
    """Mean, sample sd, range and 95% t-interval of the mean over seeds."""
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    if n == 0:
        raise ValueError("no values")
    out = {"n": n, "mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}
    if n >= 2:
        sd = float(v.std(ddof=1))
        half = T975.get(n - 1, 1.96) * sd / np.sqrt(n)
        out.update({"sd": sd, "mean_ci": [out["mean"] - half, out["mean"] + half]})
    else:
        out.update({"sd": None, "mean_ci": None})
    return out


def floor_test(
    scores: pd.DataFrame, floor: pd.DataFrame, n_boot: int = 1000, seed: int = 0
) -> dict:
    """Paired, speaker-level detector-minus-floor test on the same clips."""
    joined = floor[["filepath", "label", "speaker", "floor"]].merge(
        scores[["filepath", "score"]], on="filepath", how="inner", validate="one_to_one"
    )
    if len(joined) != len(floor):
        raise ValueError(f"scores cover {len(joined)} of the floor's {len(floor)} clips")
    stats = speaker_bootstrap(
        binary_labels(joined["label"]),
        joined["speaker"].to_numpy(),
        joined["score"].to_numpy(),
        joined["floor"].to_numpy(),
        n_boot=n_boot,
        seed=seed,
    )
    return {
        "eer": stats["eer_a"],
        "floor": stats["eer_b"],
        "diff": stats["diff"],
        "diff_ci": stats["diff_ci"],
        "verdict": verdict(stats["diff_ci"]),
    }


def seed_scores(system: SeededSystem, runs: list[dict], results_dir: str) -> dict[int, dict]:
    """seed -> {set: scores DataFrame} for the published-recipe runs of one system."""
    out: dict[int, dict] = {}
    if not system.rerun_1234:  # the published seed-1234 run is one of the seeds
        out[1234] = {ev: pd.read_csv(f"{system.published_prefix}__{ev}_scores.csv") for ev in SETS}
    for r in runs:
        if r.get("system") != system.key or r.get("options") or r.get("status") != "ok":
            continue
        if r.get("block") not in ("deployed_seeds", "s3_seeds", "xtts_channel_seeds"):
            continue
        out[int(r["seed"])] = {
            ev: pd.read_csv(Path(results_dir) / f"{r['name']}__{ev}_scores.csv") for ev in SETS
        }
    return out


def analyse(results_dir: str, floor_dir: str = FLOOR_DIR, n_boot: int = 1000) -> dict:
    """Seed stability and per-seed floor verdicts for every system with results."""
    runs = json.loads((Path(results_dir) / "runs.json").read_text())
    report = {}
    for system in SYSTEMS:
        by_seed = seed_scores(system, runs, results_dir)
        if len(by_seed) < 2:
            continue
        sets = {}
        for ev in SETS:
            floor = pd.read_csv(Path(floor_dir) / f"{system.floor_run}__{ev}_paired.csv")
            per_seed = {
                s: floor_test(scores[ev], floor, n_boot=n_boot)
                for s, scores in sorted(by_seed.items())
            }
            eers = [100 * r["eer"] for r in per_seed.values()]
            sets[ev] = {
                "seeds": {
                    str(s): {
                        "eer": 100 * r["eer"],
                        "diff": 100 * r["diff"],
                        "diff_ci": [100 * x for x in r["diff_ci"]],
                        "verdict": r["verdict"],
                    }
                    for s, r in per_seed.items()
                },
                "eer": mean_interval(eers),
                "floor": 100 * next(iter(per_seed.values()))["floor"],
                "seeds_beating_floor": sum(
                    r["verdict"] == "beats floor" for r in per_seed.values()
                ),
                "seeds_worse_than_floor": sum(
                    r["verdict"] == "worse than floor" for r in per_seed.values()
                ),
            }
        published = {
            ev: 100
            * json.loads(Path(f"{system.published_prefix}__{ev}.json").read_text())["pooled"]["eer"]
            for ev in SETS
        }
        report[system.key] = {
            "label": system.label,
            "seeds": sorted(by_seed),
            "published_1234": published,
            "sets": sets,
        }
    return report


def markdown(report: dict) -> str:
    """The stability table, one row per system and set."""
    lines = [
        "| System | Set | Seeds | EER mean ± sd | 95% CI of mean | Range | Floor | "
        "Seeds beating floor |",
        "|---|---|---:|---|---|---|---:|---:|",
    ]
    for sysrep in report.values():
        for ev, s in sysrep["sets"].items():
            e = s["eer"]
            sd = "-" if e["sd"] is None else f"{e['sd']:.2f}"
            ci = "-" if e["mean_ci"] is None else f"[{e['mean_ci'][0]:.2f}, {e['mean_ci'][1]:.2f}]"
            lines.append(
                f"| {sysrep['label']} | {ev} | {e['n']} | {e['mean']:.2f} ± {sd} | {ci} | "
                f"{e['min']:.2f}-{e['max']:.2f} | {s['floor']:.2f} | "
                f"{s['seeds_beating_floor']} of {e['n']} |"
            )
    return "\n".join(lines)


def main() -> None:
    """CLI: ``python -m src.reporting.seeds --results <unzipped review_results>``."""
    import argparse

    parser = argparse.ArgumentParser(description="Seed stability of the phone-line systems")
    parser.add_argument("--results", required=True, help="unzipped review_results/ folder")
    parser.add_argument("--floor-dir", default=FLOOR_DIR)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--out", default="experiments/results/seed_stability_v1.json")
    args = parser.parse_args()

    report = analyse(args.results, args.floor_dir, args.n_boot)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(markdown(report))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
