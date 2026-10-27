"""A stricter speech gate for the first seconds of a call: set on dev, measured on real calls.

The live path treats a window as silence only below -50 dBFS, so on a real call the quiet
before the caller's first word is normalised and scored as speech, and the final recipe's
false alarms on genuine live calls start there (``docs/results/real_calls_v1.md``). This
study sets a stricter level for the first seconds of a call on the development clips only
(the level below which a small share of dev speech windows falls), replays every scored
model's real-call windows with it, and reports what it changes: warnings on genuine calls,
clone calls caught, and how soon. Nothing is chosen on the real calls.

    python -m src.inference.early_warning \
        --dev experiments/calibration_final/dev_windows.csv \
        --windows experiments/results/real_calls_v1 --out experiments/results/early_warning_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from live_call.verdict_engine import load_config
from src.inference.real_calls import CALL_PLAN, CONDITIONS, MODELS, as_silence, call_verdicts

#: Windows ending this many seconds into a call or earlier are held to the stricter gate.
FIRST_SECONDS = 10.0
#: Share of dev speech windows the stricter gate may treat as silence.
DEV_SHARE = 0.01


def gate_level(dev_windows: pd.DataFrame, share: float = DEV_SHARE) -> float:
    """The level (dBFS) below which ``share`` of the dev windows fall; every dev window is speech."""
    return float(dev_windows["level_dbfs"].quantile(share))


def quiet_start(
    windows: pd.DataFrame, level: float, first_seconds: float = FIRST_SECONDS
) -> pd.DataFrame:
    """Windows that end in the first ``first_seconds`` and are quieter than ``level`` become silence."""
    mask = (windows["end_seconds"] <= first_seconds) & (windows["level_dbfs"] < level)
    return as_silence(windows, mask)


def outcome(calls: pd.DataFrame) -> list[dict]:
    """Per kind of call: how many were warned and taken to "likely cloned", and how soon."""
    rows = []
    for (kind, tool), group in calls.groupby(["kind", "tool"]):
        warned = group[group["caution"]]
        likely = group[group["likely_cloned"]]
        rows.append(
            {
                "kind": kind,
                "tool": tool,
                "calls": int(len(group)),
                "warned": int(len(warned)),
                "likely_cloned": int(len(likely)),
                "median_first_caution_s": float(warned["first_caution_s"].median())
                if len(warned)
                else None,
                "median_first_likely_cloned_s": float(likely["first_likely_cloned_s"].median())
                if len(likely)
                else None,
                "warned_calls": sorted(warned["call_id"]),
            }
        )
    return rows


def study(
    dev_windows: pd.DataFrame,
    windows_dir: Path,
    share: float = DEV_SHARE,
    first_seconds: float = FIRST_SECONDS,
    plan: pd.DataFrame = CALL_PLAN,
) -> dict:
    """Every scored model and condition replayed without and with the dev-set gate."""
    level = gate_level(dev_windows, share)
    result: dict = {
        "gate_level_dbfs": round(level, 2),
        "dev_share": share,
        "first_seconds": first_seconds,
        "models": {},
    }
    for name, threshold_file in MODELS.items():
        for condition in CONDITIONS:
            path = windows_dir / f"windows_{name}_{condition}.csv"
            if not path.is_file():
                continue
            config = load_config(threshold_file)
            windows = pd.read_csv(path)
            early = windows["end_seconds"] <= first_seconds
            before = plan.merge(call_verdicts(windows, config), on="call_id", how="inner")
            after = plan.merge(
                call_verdicts(quiet_start(windows, level, first_seconds), config),
                on="call_id",
                how="inner",
            )
            result["models"][f"{name}_{condition}"] = {
                "early_windows": int(early.sum()),
                "early_windows_gated": int((early & (windows["level_dbfs"] < level)).sum()),
                "without_gate": outcome(before),
                "with_gate": outcome(after),
            }
    return result


def report_markdown(result: dict) -> str:
    """One table per model and condition: the calls without and with the gate."""
    out = [
        "# Early-warning study: a stricter speech gate for the first seconds of a call",
        "",
        f"Gate: windows ending in the first {result['first_seconds']:g} s of a call and quieter "
        f"than {result['gate_level_dbfs']} dBFS are treated as silence (the level below which "
        f"{100 * result['dev_share']:g}% of the dev speech windows fall; chosen on dev only).",
    ]
    for key, entry in result["models"].items():
        out += [
            "",
            f"## {key} ({entry['early_windows_gated']} of {entry['early_windows']} windows "
            "in the first seconds gated)",
            "",
            "| Calls | Tool | n | Warned | Likely cloned | Median first caution (s) | Warned calls |",
            "|---|---|---|---|---|---|---|",
        ]
        for label, rows in (("without", entry["without_gate"]), ("with", entry["with_gate"])):
            for row in rows:
                first = row["median_first_caution_s"]
                out.append(
                    f"| {row['kind']} ({label} gate) | {row['tool']} | {row['calls']} | "
                    f"{row['warned']} | {row['likely_cloned']} | "
                    f"{'-' if first is None else f'{first:g}'} | "
                    f"{', '.join(row['warned_calls']) or '-'} |"
                )
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dev", required=True, help="dev windows CSV (levels of speech windows)")
    parser.add_argument("--windows", required=True, help="folder of real-call windows_*.csv")
    parser.add_argument("--out", required=True, help="output folder")
    parser.add_argument("--share", type=float, default=DEV_SHARE)
    parser.add_argument("--first-seconds", type=float, default=FIRST_SECONDS)
    args = parser.parse_args()
    result = study(pd.read_csv(args.dev), Path(args.windows), args.share, args.first_seconds)
    result["chosen_on"] = Path(args.dev).as_posix()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (out / "report.md").write_text(report_markdown(result), encoding="utf-8")
    print(f"gate {result['gate_level_dbfs']} dBFS; wrote {out}")


if __name__ == "__main__":
    main()
