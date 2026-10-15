"""Training changes aimed at the unseen tool (external review comment 2).

Three opt-in changes were tried on the deployed recipe: a fresh random crop each
epoch, augmentation that randomises the recording cues the shortcut floor found, and
a learned mix of all encoder layers. This module reads the Kaggle runs and reports
two separate things, kept apart on purpose:

- **the selection**, replayed with the rule fixed before any result was seen: among
  the published recipe and the four variants (seed 1234), the lowest *dev* EER wins,
  dev EERs within ``TIE`` count as a tie, and a tie goes to the candidate with fewer
  changes. Test sets are reported but never used to choose;
- **an exploratory check** of the one variant singled out *after* seeing a test score
  (crop + augment + layer mix, lowest single-run CM04 EER): five seeds of it against
  the published recipe's five, an exact two-sided Mann-Whitney test across seeds
  (also on the four seeds trained after the choice, which it cannot have influenced),
  and the paired speaker-level floor test of ``src.reporting.seeds`` for every seed.

Pure numpy/pandas, so CI runs it.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.reporting.seeds import floor_test

SETS = ("eval_pool", "cm04", "rvc_test")
TIE = 0.005  # half a point of dev EER, as in the notebook
RUN1_DIR = "experiments/results/review_runs_v1"
LAYERS_DIR = "experiments/results/review_runs_v1_layers"
EXPLORE_DIR = "experiments/results/review_runs_v1_explore"
FLOOR_DIR = "experiments/results/floor_v2"
FLOOR_RUN = "lora_norm_rvc_channel"  # every candidate trains on the deployed recipe's data
EXPLORE = "crop_aug_layers"
NEW_SEEDS = (2025, 7, 42, 99)


def candidates(run1: list[dict], layers: list[dict]) -> list[dict]:
    """The published recipe (seed 1234) and the four variants, from both runs."""
    pool = [r for r in run1 if r["name"] == "deployed_s1234"]
    pool += [r for r in run1 if r["block"] == "variants" and r["status"] == "ok"]
    pool += [r for r in layers if r["block"] == "variants" and r["status"] == "ok"]
    return pool


def pick(pool: list[dict], tie: float = TIE) -> dict | None:
    """The pre-registered rule: lowest dev EER; ties within ``tie`` go to fewer options."""
    eligible = [r for r in pool if r["status"] == "ok" and r.get("dev_eer") is not None]
    if not eligible:
        return None
    best = min(r["dev_eer"] for r in eligible)
    tied = [r for r in eligible if r["dev_eer"] <= best + tie]
    return min(tied, key=lambda r: (len(r["options"]), r["dev_eer"]))


def mann_whitney_exact(a: list[float], b: list[float]) -> float:
    """Two-sided exact Mann-Whitney p-value by enumerating every split of the pooled values.

    Fine for the five-against-five comparisons here (252 splits). Ties count half.
    """
    if not a or not b:
        raise ValueError("both groups need at least one value")
    pooled = list(a) + list(b)
    n, m = len(a), len(b)

    def u_stat(x, y):
        return sum((xi < yi) + 0.5 * (xi == yi) for xi in x for yi in y)

    centre = n * m / 2
    observed = abs(u_stat(a, b) - centre)
    extreme = total = 0
    for idx in itertools.combinations(range(n + m), n):
        chosen = set(idx)
        x = [pooled[i] for i in idx]
        y = [pooled[i] for i in range(n + m) if i not in chosen]
        total += 1
        extreme += abs(u_stat(x, y) - centre) >= observed - 1e-12
    return extreme / total


def describe(values: list[float]) -> dict:
    v = np.asarray(values, dtype=float)
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "min": float(v.min()),
        "max": float(v.max()),
    }


def _scores(results_dir: Path, name: str, ev: str) -> pd.DataFrame:
    return pd.read_csv(results_dir / f"{name}__{ev}_scores.csv")


def analyse(
    run1_dir: str = RUN1_DIR,
    layers_dir: str = LAYERS_DIR,
    explore_dir: str = EXPLORE_DIR,
    floor_dir: str = FLOOR_DIR,
    n_boot: int = 1000,
) -> dict:
    """The selection replay and the exploratory five-seed check."""
    run1_dir, layers_dir, explore_dir = Path(run1_dir), Path(layers_dir), Path(explore_dir)
    run1 = json.loads((run1_dir / "runs.json").read_text())
    layers = json.loads((layers_dir / "runs.json").read_text())
    explore = json.loads((explore_dir / "runs.json").read_text())

    pool = candidates(run1, layers)
    chosen = pick(pool)
    selection = [
        {
            "name": r["name"],
            "options": sorted(r["options"]),
            "dev_eer": 100 * r["dev_eer"],
            **{ev: 100 * r[ev] for ev in SETS},
            "chosen": chosen is not None and r["name"] == chosen["name"],
        }
        for r in pool
    ]

    published = {
        r["seed"]: r for r in run1 if r["block"] == "deployed_seeds" and r["status"] == "ok"
    }
    variant = {
        r["seed"]: (layers_dir, r) for r in layers if r["name"] == f"deployed_{EXPLORE}_s1234"
    }
    variant |= {r["seed"]: (explore_dir, r) for r in explore if r["status"] == "ok"}

    floors = {ev: pd.read_csv(Path(floor_dir) / f"{FLOOR_RUN}__{ev}_paired.csv") for ev in SETS}
    sets = {}
    for ev in SETS:
        pub = [100 * r[ev] for r in published.values()]
        var = [100 * r[ev] for _, r in variant.values()]
        var_new = [100 * r[ev] for s, (_, r) in variant.items() if s in NEW_SEEDS]
        per_seed = {}
        for seed, (d, r) in sorted(variant.items()):
            t = floor_test(_scores(d, r["name"], ev), floors[ev], n_boot=n_boot)
            per_seed[seed] = {
                "eer": 100 * t["eer"],
                "floor": 100 * t["floor"],
                "diff": 100 * t["diff"],
                "diff_ci": [100 * x for x in t["diff_ci"]],
                "verdict": t["verdict"],
            }
        sets[ev] = {
            "published": describe(pub),
            "variant": describe(var),
            "difference_of_means": float(np.mean(var) - np.mean(pub)),
            "mann_whitney_p": mann_whitney_exact(var, pub),
            "mann_whitney_p_new_seeds": mann_whitney_exact(var_new, pub),
            "floor": next(iter(per_seed.values()))["floor"],
            "variant_seeds_beating_floor": sum(
                v["verdict"] == "beats floor" for v in per_seed.values()
            ),
            "per_seed": per_seed,
        }
    dev = {
        "published": describe([100 * r["dev_eer"] for r in published.values()]),
        "variant": describe([100 * r["dev_eer"] for _, r in variant.values()]),
    }
    return {
        "rule": f"lowest dev EER; ties within {100 * TIE:.1f} points go to fewer changes",
        "selection": selection,
        "chosen": None if chosen is None else chosen["name"],
        "explore": {"variant": EXPLORE, "seeds": sorted(variant), "dev": dev, "sets": sets},
    }


def markdown(result: dict) -> str:
    """The selection table and the exploratory comparison."""
    lines = [
        f"Selection ({result['rule']}):",
        "",
        "| Candidate (seed 1234) | Dev | Eval pool | CM04 | RVC test |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in result["selection"]:
        mark = " (chosen)" if r["chosen"] else ""
        lines.append(
            f"| {r['name']}{mark} | {r['dev_eer']:.2f} | {r['eval_pool']:.2f} | "
            f"{r['cm04']:.2f} | {r['rvc_test']:.2f} |"
        )
    ex = result["explore"]
    lines += [
        "",
        f"Exploratory, {ex['variant']} against the published recipe, five seeds each:",
        "",
        "| Set | Published | Variant | Difference | Exact MWU p (all / new seeds) | "
        "Variant seeds beating floor |",
        "|---|---|---|---:|---|---:|",
    ]
    for ev, s in ex["sets"].items():
        p, v = s["published"], s["variant"]
        lines.append(
            f"| {ev} | {p['mean']:.2f} ± {p['sd']:.2f} | {v['mean']:.2f} ± {v['sd']:.2f} | "
            f"{s['difference_of_means']:+.2f} | {s['mann_whitney_p']:.4f} / "
            f"{s['mann_whitney_p_new_seeds']:.4f} | {s['variant_seeds_beating_floor']} of "
            f"{v['n']} (floor {s['floor']:.2f}) |"
        )
    d = ex["dev"]
    lines.append(
        f"| dev | {d['published']['mean']:.2f} ± {d['published']['sd']:.2f} | "
        f"{d['variant']['mean']:.2f} ± {d['variant']['sd']:.2f} | "
        f"{d['variant']['mean'] - d['published']['mean']:+.2f} | | |"
    )
    return "\n".join(lines)


def main() -> None:
    """CLI: ``python -m src.reporting.variants``."""
    import argparse

    parser = argparse.ArgumentParser(description="Training changes aimed at the unseen tool")
    parser.add_argument("--out", default="experiments/results/variants_v1.json")
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()
    result = analyse(n_boot=args.n_boot)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(markdown(result))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
