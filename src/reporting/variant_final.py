"""Plan task A5: the week-12 variant on the final recipe, judged by its registered rule.

The hypotheses and the decision rule were written before the run
(``docs/results/variants_final_v1.md``, committed while the run was in progress) and are coded
here unchanged. The variant is **confirmed** only if all three hold:

1. lower IndicF5 (CM07, G.711) EER than the final recipe in at least 4 of 5 paired seeds;
2. exact two-sided Mann-Whitney p <= 0.05 on the five-against-five IndicF5 EERs;
3. H2: new-speaker Tortoise (G.711) mean EER at most 1.0 point above the final recipe's.

Otherwise it is **not confirmed**; with fewer than five seeds finished it is **incomplete**.
The comparison is the final recipe's own five seeds (``experiments/results/final_v1``)::

    python -m src.reporting.variant_final --runs variant_results/runs.json
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from src.reporting.variants import mann_whitney_exact

FINAL_RUNS = "experiments/results/final_v1/runs.json"
SEEDS = (1234, 2025, 7, 42, 99)
PRIMARY = "cm07"  # IndicF5 over G.711
GUARD = "new_tortoise"  # new-speaker Tortoise over G.711
MIN_LOWER = 4  # of 5 paired seeds
MAX_P = 0.05
MAX_RISE = 0.010  # 1.0 point, as a fraction


def by_seed(rows: list[dict], system: str) -> dict[int, dict]:
    """Seed -> run row of one system, finished runs only."""
    return {
        int(r["seed"]): r
        for r in rows
        if r.get("system") == system and r.get("status") == "ok" and r.get(PRIMARY) is not None
    }


def judge(variant: dict[int, dict], final: dict[int, dict]) -> dict:
    """The registered decision on the variant's and the final recipe's runs."""
    seeds = [s for s in SEEDS if s in variant and s in final]
    out = {"seeds": seeds, "planned_seeds": list(SEEDS)}
    for key in (PRIMARY, GUARD):
        out[key] = {s: {"variant": variant[s].get(key), "final": final[s].get(key)} for s in seeds}
    if len(seeds) < len(SEEDS):
        out["decision"] = "incomplete"
        return out
    var = [variant[s][PRIMARY] for s in seeds]
    fin = [final[s][PRIMARY] for s in seeds]
    lower = sum(v < f for v, f in zip(var, fin, strict=True))
    p = mann_whitney_exact(var, fin)
    rise = mean(variant[s][GUARD] for s in seeds) - mean(final[s][GUARD] for s in seeds)
    out.update(
        lower_in_seeds=lower,
        mann_whitney_p=p,
        tortoise_mean_rise=rise,
        conditions={
            "1_lower_in_4_of_5": lower >= MIN_LOWER,
            "2_mann_whitney": p <= MAX_P,
            "3_tortoise_within_1_point": rise <= MAX_RISE,
        },
    )
    out["decision"] = "confirmed" if all(out["conditions"].values()) else "not confirmed"
    return out


def main() -> None:
    """CLI: judge the variant's runs.json against the final recipe; write one JSON."""
    import argparse

    parser = argparse.ArgumentParser(description="A5: the variant on the final recipe")
    parser.add_argument("--runs", required=True, help="the variant notebook's runs.json")
    parser.add_argument("--final-runs", default=FINAL_RUNS)
    parser.add_argument("--out", default="experiments/results/variant_final_v1.json")
    args = parser.parse_args()
    variant = by_seed(json.loads(Path(args.runs).read_text(encoding="utf-8")), "variant")
    final = by_seed(json.loads(Path(args.final_runs).read_text(encoding="utf-8")), "final")
    report = judge(variant, final)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("seeds:", report["seeds"])
    for key in (PRIMARY, GUARD):
        cells = "  ".join(
            f"s{s} {100 * v['variant']:.2f} vs {100 * v['final']:.2f}"
            for s, v in report[key].items()
        )
        print(f"{key:13s} {cells}")
    if "conditions" in report:
        print("conditions:", report["conditions"], " p =", round(report["mann_whitney_p"], 4))
    print("DECISION:", report["decision"])


if __name__ == "__main__":
    main()
