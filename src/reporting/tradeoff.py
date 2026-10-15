"""The Hinglish phone-line versus English trade-off (external review comment 4).

The review asks us to quantify what channel-matched adaptation gains on Hinglish over
the phone line against what it costs on clean English, and what that means for a
deployment. The pieces were measured in different places; this module puts them in
one table, read from the result files rather than retyped:

- each S2 adapter on Hinglish in the condition it was trained in
  (``systems_s1_s2_s3.json``) and in the condition it was *not* trained in
  (``cross_condition_v1/``), so the cost of choosing a condition is visible;
- each adapter on clean English, all 71,237 ASVspoof LA evaluation clips, as EER and
  as the share of English fakes caught at the deployed threshold
  (``classification_metrics.json``);
- the deployed adapter and the English-only S1 on a fixed 6,004-clip English sample,
  clean and over the same phone line (``english_channel_v1/``), so the English cost
  is also measured in the deployment condition.

A file not yet produced is reported as ``None`` ("not measured"), never estimated.
Pure json/pandas, so CI runs it.
"""

from __future__ import annotations

import json
from pathlib import Path

SYSTEMS = "experiments/results/systems_s1_s2_s3.json"
CLASSIFICATION = "experiments/results/classification_metrics.json"
ENGLISH_DIR = "experiments/results/english_channel_v1"
CROSS_DIR = "experiments/results/cross_condition_v1"
SETS = ("eval_pool", "cm04", "rvc_test")

#: S2 run -> (training attacks, training condition).
S2_RUNS: dict[str, tuple[str, str]] = {
    "lora_norm_clean": ("XTTS", "clean"),
    "lora_norm_channel": ("XTTS", "channel"),
    "lora_norm_rvc_clean": ("XTTS + RVC", "clean"),
    "lora_norm_rvc_channel": ("XTTS + RVC", "channel"),
}
#: Models scored on the English sample, and their file stems.
ENGLISH_MODELS = {"s1": "S1 (English only)", "deployed_s2": "S2 XTTS + RVC, channel (deployed)"}


def _pooled(path: Path) -> dict | None:
    """EER (%) and its interval from an evaluate JSON, or None if it does not exist."""
    if not path.is_file():
        return None
    p = json.loads(path.read_text())["pooled"]
    return {
        "eer": round(100 * p["eer"], 2),
        "ci": [
            round(100 * p.get("eer_ci_low", float("nan")), 2),
            round(100 * p.get("eer_ci_high", float("nan")), 2),
        ],
    }


def build(
    systems: str = SYSTEMS,
    classification: str = CLASSIFICATION,
    english_dir: str = ENGLISH_DIR,
    cross_dir: str = CROSS_DIR,
) -> dict:
    """The trade-off, one entry per S2 adapter plus the English sample comparison."""
    cells = json.loads(Path(systems).read_text())["cells"]
    runs = json.loads(Path(classification).read_text())["runs"]

    def system_cell(run: str, ev: str) -> dict | None:
        hit = [c for c in cells if c.get("run") == run and c.get("set") == ev]
        if not hit:
            return None
        c = hit[0]
        return {"eer": c["eer"], "ci": c.get("eer_ci")}

    def deployed_point(run: str, ev: str) -> dict | None:
        hit = [r for r in runs if r["run"] == run and r["set"] == ev]
        if not hit:
            return None
        d = hit[0]["deployed"]
        return {
            "fakes_caught": round(100 * d["recall"], 1),
            "genuine_passed": round(100 * d["specificity"], 1),
        }

    adapters = []
    for run, (attacks, trained) in S2_RUNS.items():
        other = "clean" if trained == "channel" else "channel"
        adapters.append(
            {
                "run": run,
                "attacks": attacks,
                "trained_condition": trained,
                "hinglish_own_condition": {ev: system_cell(run, ev) for ev in SETS},
                "hinglish_other_condition": {
                    ev: _pooled(Path(cross_dir) / f"{run}__{ev}_{other}.json") for ev in SETS
                },
                "other_condition": other,
                "english_clean_full": system_cell(run, "english"),
                "english_at_deployed_threshold": deployed_point(run, "english"),
            }
        )

    english_sample = {
        key: {
            "label": label,
            "clean": _pooled(Path(english_dir) / f"{key}__english_clean.json"),
            "channel": _pooled(Path(english_dir) / f"{key}__english_channel.json"),
        }
        for key, label in ENGLISH_MODELS.items()
    }
    return {
        "adapters": adapters,
        "english_sample": english_sample,
        "s1_english_at_deployed_threshold": deployed_point("stage1", "asvspoof_eval"),
    }


def _fmt(cell: dict | None) -> str:
    return "not measured" if cell is None else f"{cell['eer']:.2f}"


def markdown(result: dict) -> str:
    """Two tables: Hinglish own vs other condition with English, and the English sample."""
    lines = [
        "| S2 adapter | Trained on | Hinglish, own condition (eval / CM04 / RVC) | "
        "Hinglish, other condition (eval / CM04 / RVC) | English, clean | "
        "English fakes caught at 0.5 |",
        "|---|---|---|---|---:|---:|",
    ]
    for a in result["adapters"]:
        own = " / ".join(_fmt(a["hinglish_own_condition"][ev]) for ev in SETS)
        other = " / ".join(_fmt(a["hinglish_other_condition"][ev]) for ev in SETS)
        pt = a["english_at_deployed_threshold"]
        caught = "-" if pt is None else f"{pt['fakes_caught']:.1f}%"
        lines.append(
            f"| {a['attacks']} | {a['trained_condition']} | {own} | {other} "
            f"({a['other_condition']}) | {_fmt(a['english_clean_full'])} | {caught} |"
        )
    lines += [
        "",
        "| Model | English sample, clean | English sample, phone line |",
        "|---|---:|---:|",
    ]
    for m in result["english_sample"].values():
        lines.append(f"| {m['label']} | {_fmt(m['clean'])} | {_fmt(m['channel'])} |")
    return "\n".join(lines)


def main() -> None:
    """CLI: ``python -m src.reporting.tradeoff``."""
    import argparse

    parser = argparse.ArgumentParser(description="Hinglish phone-line vs English trade-off")
    parser.add_argument("--out", default="experiments/results/tradeoff_v1.json")
    args = parser.parse_args()
    result = build()
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(markdown(result))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
