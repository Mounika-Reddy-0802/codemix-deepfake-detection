"""Markdown tables of the generator study, written from the rule's own output file.

The results section of ``docs/results/generator_study_v1.md`` quotes these tables, so no number
is copied by hand from a terminal. Reads ``experiments/results/generator_study_v1.json``
(``src.reporting.generator_study``) and, when present, ``floor_genstudy_v1.json``::

    python -m src.reporting.genstudy_tables > tables.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPORT = "experiments/results/generator_study_v1.json"
FLOORS = "experiments/results/floor_genstudy_v1.json"
NAMES = {
    "chatterbox": "Chatterbox, G.711",
    "indicf5": "IndicF5, G.711",
    "indicf5_amr": "IndicF5, AMR-NB",
    "new_tortoise": "Tortoise, new speakers, G.711",
    "new_tortoise_amr": "Tortoise, new speakers, AMR-NB",
    "new_xtts": "XTTS-v2, new speakers, G.711",
    "cm04": "CM04 Tortoise, G.711",
    "cm04_amr": "CM04 Tortoise, AMR-NB",
    "eval_pool": "Code-mixed test pool (XTTS-v2)",
    "rvc_test": "RVC test half",
    "control_fa": "Language-control genuine called fake",
}
SYSTEMS = ("control", "fold_a", "fold_b", "fold_c")


def pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.2f}"


def decision_tables(report: dict) -> str:
    """Per fold: the per-seed table of its unseen tool, then the four conditions."""
    if "folds" not in report:
        return f"The study is {report.get('answer')}: seeds {report.get('seeds')}.\n"
    seeds = report["seeds"]
    lines = [f"**Answer: {report['answer']}.**", ""]
    for fold, f in report["folds"].items():
        lines += [
            f"**{fold}** (never heard {NAMES[f['unseen_tool']]}): **{f['decision']}**",
            "",
            "| Seed | Control | " + fold + " |",
            "|---|---|---|",
        ]
        for s, c, v in zip(seeds, f["control"], f["fold"], strict=True):
            lines.append(f"| {s} | {pct(c)} | {pct(v)} |")
        conds = f["conditions"]
        lines += [
            "",
            f"1. lower in at least 4 of 5 seeds: {f['lower_in_seeds']} of 5 "
            f"({'met' if conds['1_lower_in_4_of_5'] else 'not met'});",
            f"2. Mann-Whitney p <= 0.025: p = {f['mann_whitney_p']:.4f} "
            f"({'met' if conds['2_mann_whitney'] else 'not met'}; two-sided, condition 1 sets the "
            "direction);",
            f"3. Tortoise within 1 point: {100 * f['tortoise_mean_rise']:+.2f} points "
            f"({'met' if conds['3_tortoise_within_1_point'] else 'not met'});",
            f"4. false alarms within 0.5 points: {100 * f['false_alarm_mean_rise']:+.2f} points "
            f"({'met' if conds['4_false_alarms_within_half_point'] else 'not met'}).",
            "",
        ]
    return "\n".join(lines)


def means_table(report: dict) -> str:
    """Mean ± SD of every measure, every system (held-out speakers)."""
    means = report.get("means", {})
    lines = [
        "| Measure | " + " | ".join(SYSTEMS) + " |",
        "|---|" + "---|" * len(SYSTEMS),
    ]
    for key, name in NAMES.items():
        cells = []
        for system in SYSTEMS:
            m = means.get(system, {}).get(key)
            cells.append("-" if m is None else f"{100 * m['mean']:.2f} ± {100 * m['sd']:.2f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def floors_table(floors: dict) -> str:
    lines = [
        "| System | Set | Detector EER | Floor EER | Beats floor |",
        "|---|---|---|---|---|",
    ]
    for r in floors.get("summary", []):
        lines.append(
            f"| {r['detector']} | {r['test_set']} | {100 * r['detector_eer']:.2f} "
            f"| {100 * r['floor_eer']:.2f} | {r['beats_floor']}/{r['seeds']} |"
        )
    return "\n".join(lines) + "\n"


def render(report: dict, floors: dict | None = None) -> str:
    parts = [decision_tables(report), "EER (%) on held-out speakers, mean ± SD over seeds:", "",
             means_table(report)]  # fmt: skip
    if floors:
        parts += ["", "Matched shortcut floors (held-out speakers):", "", floors_table(floors)]
    return "\n".join(parts)


def main() -> None:
    # the tables are redirected into markdown files; Windows would write them in its code page
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    report = json.loads(Path(REPORT).read_text(encoding="utf-8"))
    floors = (
        json.loads(Path(FLOORS).read_text(encoding="utf-8")) if Path(FLOORS).is_file() else None
    )
    print(render(report, floors))


if __name__ == "__main__":
    main()
