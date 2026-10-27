"""Spontaneous speech against the operating point: a trade-off that chooses nothing.

At its live operating point the final recipe warns every genuine spontaneous (HiACC) 60 s
call (``docs/results/live_final_recipe_v1.md``). Could a different threshold have avoided
that? This sweeps the threshold of the final recipe's verdict ladder (caution after 3
fake-leaning windows, "likely cloned" after 5, as live) over the cached live-path window
scores in ``experiments/calibration_final/``, stitched into 60 s calls exactly as the call
study stitches them, and reports at each value the share of genuine HiACC calls warned, of
genuine read-speech calls warned, and of clone calls caught in each test set.

HiACC is evaluation-only (P-036 and the golden dataset rule), so nothing is chosen here: the
table and figure describe the curve, and the live threshold stays the one chosen on dev. At
the live threshold the sweep must reproduce the call study's table, which it checks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from live_call.verdict_engine import EngineConfig, load_config
from src.inference.calibrate import replay_calls, stitch_calls

ROOT = Path("experiments/calibration_final")
THRESHOLD_FILE = "configs/threshold_final.yaml"
#: window file stem -> (label for tables, whether its clones are from an unseen tool)
SETS = {
    "eval_pool": ("Evaluation pool (XTTS-v2)", False),
    "cm04": ("CM04 (Tortoise, unseen)", True),
    "rvc_test": ("RVC test half", False),
    "new_speakers": ("60 new speakers (XTTS-v2 + Tortoise)", False),
    "cm07": ("IndicF5 (unseen)", True),
    "cm09": ("Chatterbox (unseen)", True),
}
SPONTANEOUS = "hiacc"
GRID = tuple(round(float(t), 2) for t in np.arange(0.05, 1.0, 0.05))


def stitched_sets(root: str | Path = ROOT, seconds: float = 60) -> dict[str, pd.DataFrame]:
    """Every set's windows stitched into calls once; the threshold does not change them."""
    root = Path(root)
    out = {}
    for stem in (*SETS, SPONTANEOUS):
        path = root / f"{stem}_windows.csv"
        if path.is_file():
            out[stem] = stitch_calls(pd.read_csv(path), seconds)
    return out


def sweep(
    calls: dict[str, pd.DataFrame], thresholds, suspicious_after: int = 3, fake_after: int = 5
) -> pd.DataFrame:
    """One row per threshold, set and label: calls, and the share warned or "likely cloned"."""
    rows = []
    for threshold in thresholds:
        config = EngineConfig(
            threshold=float(threshold), suspicious_after=suspicious_after, fake_after=fake_after
        )
        for stem, stitched in calls.items():
            verdicts = replay_calls(stitched, config)
            for label, group in verdicts.groupby("label"):
                rows.append(
                    {
                        "threshold": float(threshold),
                        "set": stem,
                        "label": label,
                        "calls": int(len(group)),
                        "alerted": int(group["alerted"].sum()),
                        "likely_fake": int(group["likely_fake"].sum()),
                    }
                )
    return pd.DataFrame(rows)


def summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Per threshold: genuine HiACC and read-speech calls warned, clone calls caught (%).

    Several read-speech sets share their genuine side (CM07 and CM09 are scored against the
    same genuine clips), so pooling them would count a call more than once; the read-speech
    figure is the worst of the six sets instead.
    """

    def pct(part: pd.DataFrame) -> float:
        return round(100 * part["alerted"].sum() / part["calls"].sum(), 1) if len(part) else np.nan

    unseen = {stem for stem, (_, is_unseen) in SETS.items() if is_unseen}
    out = []
    for threshold, at in rows.groupby("threshold"):
        genuine, fake = at[at["label"] == "bonafide"], at[at["label"] == "spoof"]
        read = genuine[genuine["set"] != SPONTANEOUS]
        entry = {
            "threshold": threshold,
            "hiacc_genuine_warned_pct": pct(genuine[genuine["set"] == SPONTANEOUS]),
            "read_genuine_warned_worst_set_pct": max(
                (pct(group) for _, group in read.groupby("set")), default=np.nan
            ),
            "clones_caught_pct": pct(fake),
            "unseen_clones_caught_pct": pct(fake[fake["set"].isin(unseen)]),
        }
        for stem in SETS:
            entry[f"{stem}_caught_pct"] = pct(fake[fake["set"] == stem])
        out.append(entry)
    return pd.DataFrame(out)


def figure(table: pd.DataFrame, live_threshold: float, out_dir: str | None = None) -> str:
    """Write ``spontaneous_tradeoff.png`` and ``.pdf``: one axis, percent of calls."""
    from src.reporting.figures import DPI, FIGURES_DIR, _style

    plt = _style()
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    grid = table[table["threshold"].isin(GRID)].sort_values("threshold")
    lines = (
        ("hiacc_genuine_warned_pct", "Genuine spontaneous calls warned", "#c0392b", "o", "-"),
        (
            "read_genuine_warned_worst_set_pct",
            "Genuine read-speech calls warned (worst set)",
            "#7f8c8d",
            "^",
            "-",
        ),
        ("clones_caught_pct", "Clone calls caught, all six sets", "#1baf7a", "s", "-"),
        ("unseen_clones_caught_pct", "Clone calls caught, unseen tools", "#2e86c1", "D", "--"),
    )
    for column, name, colour, marker, style in lines:
        ax.plot(
            grid["threshold"],
            grid[column],
            color=colour,
            marker=marker,
            markersize=3,
            lw=1.4,
            ls=style,
            label=name,
        )
    ax.axvline(live_threshold, ls=":", lw=0.8, color="#333333")
    ax.text(
        live_threshold + 0.01,
        3,
        f"live threshold {live_threshold:.2f}",
        fontsize=5.5,
        color="#333333",
        rotation=90,
        va="bottom",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(-3, 103)
    ax.set_xlabel("Threshold on the smoothed window score", fontsize=7)
    ax.set_ylabel("Share of 60 s calls (%)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=5.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    out = Path(out_dir or FIGURES_DIR) / "spontaneous_tradeoff.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return str(out)


def main() -> None:
    live = load_config(THRESHOLD_FILE)
    calls = stitched_sets()
    rows = sweep(calls, sorted({*GRID, live.threshold}), live.suspicious_after, live.fake_after)
    table = summary(rows)
    out = Path("experiments/results/spontaneous_tradeoff_v1")
    out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out / "calls_by_threshold.csv", index=False)
    table.to_csv(out / "tradeoff.csv", index=False)
    print(table.to_string(index=False))
    at_live = table[np.isclose(table["threshold"], live.threshold)].iloc[0]
    print("\nat the live threshold:", at_live.to_dict())
    print("wrote", figure(table, live.threshold))


if __name__ == "__main__":
    main()
