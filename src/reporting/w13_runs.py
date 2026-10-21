"""The three week-13 Kaggle runs, compared seed for seed.

Each run's ``runs.json`` holds one row per trained model: its system or training set,
its seed, and one EER (or false-alarm rate) per evaluation set. Every comparison here
pairs two systems trained with the **same seeds** and reports, per metric:

* each system's mean, sd and 95% t-interval over its seeds (``seeds.mean_interval``);
* the paired difference (first minus second) averaged over the seeds both have, the
  number of seeds where the first is better, and an exact two-sided sign-flip
  permutation p-value over those paired differences (2^n flips, n = 5 gives 32).

The comparisons:

* **Run 1 (corpus-overfit fix):** the extended recipe (248 genuine speakers) against the
  deployed recipe (21);
* **Run 2 (AASIST):** AASIST fine-tuned on the deployed training set against the
  deployed wav2vec2 S2 from Run 1 with the same seeds, and the same for the extended
  set; the zero-shot AASIST row is reported beside S1;
* **Run 3 (two codecs):** the G.711 + AMR-NB recipe against the deployed recipe.

Nothing here chooses anything: the pairs are fixed by design and every metric is shown.
"""

from __future__ import annotations

import argparse
import itertools
import json
import zipfile
from pathlib import Path

import numpy as np

from src.reporting.seeds import mean_interval
from src.reporting.variants import mann_whitney_exact

FIELDS = {
    "eval_pool": "Original test pool (XTTS)",
    "cm04": "CM04 Tortoise",
    "rvc_test": "RVC test half",
    "new_xtts": "New speakers, XTTS",
    "new_tortoise": "New speakers, Tortoise",
    "cm07": "IndicF5",
    "control_fa": "False alarms, language control",
    "new_xtts_amr": "New speakers, XTTS, AMR-NB",
    "new_tortoise_amr": "New speakers, Tortoise, AMR-NB",
    "cm07_amr": "IndicF5, AMR-NB",
    "cm04_amr": "CM04, AMR-NB",
}


def load_runs(path: str) -> list[dict]:
    """``runs.json`` from a results zip or folder (any depth)."""
    p = Path(path)
    if p.is_dir():
        found = sorted(p.rglob("runs.json"))
        if not found:
            raise FileNotFoundError(f"no runs.json under {path}")
        return json.loads(found[0].read_text(encoding="utf-8"))
    with zipfile.ZipFile(p) as zf:
        name = next((n for n in zf.namelist() if n.endswith("runs.json")), None)
        if name is None:
            raise FileNotFoundError(f"no runs.json in {path}")
        return json.loads(zf.read(name))


def sign_flip_p(diffs: list[float]) -> float:
    """Exact two-sided p-value of the mean paired difference under random sign flips."""
    d = np.asarray(diffs, dtype=float)
    if d.size == 0:
        raise ValueError("no paired differences")
    observed = abs(d.mean())
    extreme = total = 0
    for signs in itertools.product((1.0, -1.0), repeat=d.size):
        total += 1
        extreme += abs((d * np.asarray(signs)).mean()) >= observed - 1e-12
    return extreme / total


def by_seed(rows: list[dict], field: str) -> dict[int, float]:
    return {
        int(r["seed"]): float(r[field])
        for r in rows
        if r.get("seed") is not None and r.get(field) is not None
    }


def compare(rows_a: list[dict], rows_b: list[dict], fields=FIELDS) -> dict:
    """Per metric: both systems over seeds and their paired difference (a - b)."""
    out = {}
    for field in fields:
        a, b = by_seed(rows_a, field), by_seed(rows_b, field)
        if not a or not b:
            continue
        common = sorted(set(a) & set(b))
        diffs = [a[s] - b[s] for s in common]
        entry = {
            "a": mean_interval(list(a.values())),
            "b": mean_interval(list(b.values())),
            "paired_seeds": common,
        }
        if len(a) > 1 and len(b) > 1:
            entry["p_mann_whitney"] = mann_whitney_exact(list(a.values()), list(b.values()))
        if diffs:
            entry.update(
                diff_mean=float(np.mean(diffs)),
                a_better=int(sum(d < 0 for d in diffs)),
                p_sign_flip=sign_flip_p(diffs),
            )
        out[field] = entry
    return out


def select(runs: list[dict], key: str, value) -> list[dict]:
    return [r for r in runs if r.get(key) == value and r.get("status") in ("ok", "partial")]


def analyse(run1: str | None = None, run2: str | None = None, run3: str | None = None) -> dict:
    report: dict = {}
    r1 = load_runs(run1) if run1 else []
    deployed = select(r1, "system", "deployed")
    if r1:
        report["run1_ext_vs_deployed"] = compare(select(r1, "system", "ext"), deployed)
    if run2:
        r2 = load_runs(run2)
        zero = [r for r in r2 if r.get("name") == "zero_shot"]
        report["run2_zero_shot"] = {
            f: zero[0].get(f) for f in FIELDS if zero and zero[0].get(f) is not None
        }
        if deployed:
            report["run2_aasist_s2data_vs_deployed"] = compare(
                select(r2, "data", "s2data"), deployed
            )
            ext = select(r1, "system", "ext")
            if ext:
                report["run2_aasist_extdata_vs_ext"] = compare(select(r2, "data", "extdata"), ext)
    if run3:
        r3 = load_runs(run3)
        report["run3_mixcodec_vs_deployed"] = compare(
            select(r3, "system", "mixcodec"), select(r3, "system", "deployed")
        )
    return report


TITLES = {
    "run1_ext_vs_deployed": (
        "Run 1: extended recipe (248 speakers) vs deployed (21)",
        "extended",
        "deployed",
    ),
    "run2_aasist_s2data_vs_deployed": (
        "Run 2: AASIST vs wav2vec2 S2, same training data and seeds",
        "AASIST",
        "S2",
    ),
    "run2_aasist_extdata_vs_ext": (
        "Run 2: AASIST vs wav2vec2 S2, both on the extended set",
        "AASIST",
        "S2 ext",
    ),
    "run3_mixcodec_vs_deployed": (
        "Run 3: G.711 + AMR-NB training vs deployed (G.711)",
        "two-codec",
        "deployed",
    ),
}


def _cell(stats: dict) -> str:
    if stats.get("sd") is None:
        return f"{100 * stats['mean']:.2f} (n=1)"
    return f"{100 * stats['mean']:.2f} ± {100 * stats['sd']:.2f} (n={stats['n']})"


def markdown(report: dict) -> str:
    lines = [
        "# The week-13 Kaggle runs, seed for seed (v1)",
        "",
        "EER (%) mean ± sd over training seeds (false alarms: % of genuine clips called fake).",
        "Difference: first minus second over the seeds both have; negative favours the first.",
        "p paired: exact sign-flip test over paired seeds (minimum 0.0625 with five seeds);",
        "p MWU: exact Mann-Whitney test, unpaired (minimum 0.008 with five each). Built by",
        "`src/reporting/w13_runs.py`.",
        "",
    ]
    if report.get("run2_zero_shot"):
        lines += [
            "## AASIST zero-shot (published ASVspoof 2019 LA weights)",
            "",
            "| Set | EER (%) |",
            "|---|---|",
        ]
        lines += [f"| {FIELDS[f]} | {100 * v:.2f} |" for f, v in report["run2_zero_shot"].items()]
        lines.append("")
    for key, (title, a_name, b_name) in TITLES.items():
        table = report.get(key)
        if not table:
            continue
        lines += [
            f"## {title}",
            "",
            f"| Set | {a_name} | {b_name} | Difference | {a_name} better | p paired | p MWU |",
            "|---|---|---|---|---|---|---|",
        ]
        for field, e in table.items():
            diff = "-" if "diff_mean" not in e else f"{100 * e['diff_mean']:+.2f}"
            better = "-" if "a_better" not in e else f"{e['a_better']}/{len(e['paired_seeds'])}"
            p = "-" if "p_sign_flip" not in e else f"{e['p_sign_flip']:.3f}"
            mwu = "-" if "p_mann_whitney" not in e else f"{e['p_mann_whitney']:.3f}"
            lines.append(
                f"| {FIELDS[field]} | {_cell(e['a'])} | {_cell(e['b'])} | {diff} | {better} "
                f"| {p} | {mwu} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed-for-seed comparison of the week-13 runs")
    parser.add_argument("--run1", help="overfit_fix_results.zip or its folder")
    parser.add_argument("--run2", help="aasist_results.zip or its folder")
    parser.add_argument("--run3", help="mixcodec_results.zip or its folder")
    parser.add_argument("--out", default="experiments/results/w13_runs_v1.json")
    parser.add_argument("--doc", default="docs/results/w13_runs_v1.md")
    args = parser.parse_args()
    report = analyse(args.run1, args.run2, args.run3)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    Path(args.doc).write_text(markdown(report), encoding="utf-8")
    print(markdown(report))


if __name__ == "__main__":
    main()
