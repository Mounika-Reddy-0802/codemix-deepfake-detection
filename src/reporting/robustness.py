"""The robustness sweep's report: each transform against no transform, by the rules fixed first.

``docs/results/robustness_v1.md`` fixed the measures and the reading rules on 1 November 2026,
before any clip was transformed or scored; this module is those rules as code, written before
the scores exist. For every model, set and transform it compares the transformed clips with
the same clips under ``none`` (the channel bundle's clip):

- the EER and its change, by a paired bootstrap over whole speakers
  (``floor_compare.speaker_bootstrap``);
- at the operating rule, the share of genuine clips called fake (false alarms) and of clones
  passed as genuine, and their changes, paired the same way.

A transform **helps an attacker** when the clones-passed change has its whole 95% interval
above zero, and **hurts a genuine caller** when the false-alarm change does; otherwise it is
reported as no measurable effect, with its interval.

Run, once the sweep is scored::

    python -m src.reporting.robustness --scores adapter=SCORES.csv --scores ssl=SCORES.csv \\
        --out experiments/results/robustness_v1/summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.training.floor_compare import binary_labels, speaker_bootstrap

#: The operating rule, a clip is called fake below this score: P(genuine) 0.5 for the adapter,
#: log-odds 0 for SSL-AASIST, whose scorer writes log-odds. The same rule on both scales.
CALLED_FAKE_BELOW = {"adapter": 0.5, "ssl": 0.0}
N_BOOT = 1000
SEED = 1234


def clip_key(filepath: str) -> str:
    """The clip's name: the same under every transform folder."""
    return Path(str(filepath).replace("\\", "/")).name


def paired_rate_change(
    speakers: np.ndarray,
    flags_t: np.ndarray,
    flags_none: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict:
    """Share of flagged clips with a transform minus without, resampling whole speakers."""
    speakers = np.asarray(speakers).astype(str)
    t = np.asarray(flags_t, dtype=np.float64)
    n = np.asarray(flags_none, dtype=np.float64)
    if not (len(speakers) == len(t) == len(n)) or not len(t):
        raise ValueError("speakers and both flag arrays must be the same, non-zero length")
    unique = np.unique(speakers)
    members = [np.flatnonzero(speakers == s) for s in unique]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(unique), len(unique))])
        diffs.append(t[idx].mean() - n[idx].mean())
    return {
        "with": float(t.mean()),
        "without": float(n.mean()),
        "change": float(t.mean() - n.mean()),
        "change_ci": [float(v) for v in np.percentile(diffs, [2.5, 97.5])],
        "clips": int(len(t)),
        "speakers": int(len(unique)),
    }


def reading(passed: dict, alarms: dict) -> str:
    """The pre-registered reading of one transform on one set."""
    said = []
    if passed["change"] > 0 and passed["change_ci"][0] > 0:
        said.append("helps an attacker")
    if alarms["change"] > 0 and alarms["change_ci"][0] > 0:
        said.append("hurts a genuine caller")
    return " and ".join(said) or "no measurable effect"


def compare(scored: pd.DataFrame, model: str, n_boot: int = N_BOOT) -> list[dict]:
    """Every transform of every set against ``none`` on the same clips, for one model."""
    cut = CALLED_FAKE_BELOW[model]
    df = scored.assign(key=scored["filepath"].map(clip_key))
    rows = []
    for set_name, part in df.groupby("set", sort=True):
        base = part[part["transform"] == "none"].set_index("key")
        if base.empty:
            raise ValueError(f"{model}/{set_name}: no untransformed clips to compare with")
        for transform, tpart in part[part["transform"] != "none"].groupby("transform", sort=True):
            t = tpart.set_index("key")
            missing = base.index.difference(t.index)
            if len(missing):
                raise ValueError(f"{model}/{set_name}/{transform}: {len(missing)} clips not scored")
            t = t.loc[base.index]
            labels = binary_labels(base["label"])
            speakers = base["speaker"].astype(str).to_numpy()
            eer = speaker_bootstrap(
                labels, speakers, t["score"].to_numpy(), base["score"].to_numpy(), n_boot, SEED
            )
            genuine, spoof = labels == 1, labels == 0
            alarms = paired_rate_change(
                speakers[genuine],
                t["score"].to_numpy()[genuine] < cut,
                base["score"].to_numpy()[genuine] < cut,
                n_boot,
            )
            passed = paired_rate_change(
                speakers[spoof],
                t["score"].to_numpy()[spoof] >= cut,
                base["score"].to_numpy()[spoof] >= cut,
                n_boot,
            )
            rows.append(
                {
                    "model": model,
                    "set": set_name,
                    "transform": transform,
                    "eer": eer["eer_a"],
                    "eer_none": eer["eer_b"],
                    "eer_change": eer["diff"],
                    "eer_change_ci": eer["diff_ci"],
                    "false_alarms": alarms,
                    "clones_passed": passed,
                    "reading": reading(passed, alarms),
                }
            )
    return rows


def markdown(rows: list[dict]) -> str:
    """The results table for ``robustness_v1.md``: one row per model, set and transform."""
    pct = lambda v: f"{100 * v:.1f}"  # noqa: E731
    ci = lambda c: f"[{100 * c[0]:+.1f}, {100 * c[1]:+.1f}]"  # noqa: E731
    lines = [
        "| Model | Set | Transform | EER (%) | Change | Clones passed (%) | Change | "
        "False alarms (%) | Change | Reading |",
        "|---|---|---|---:|---|---:|---|---:|---|---|",
    ]
    for r in rows:
        p, a = r["clones_passed"], r["false_alarms"]
        lines.append(
            f"| {r['model']} | {r['set']} | `{r['transform']}` | {pct(r['eer'])} | "
            f"{100 * r['eer_change']:+.1f} {ci(r['eer_change_ci'])} | {pct(p['with'])} | "
            f"{100 * p['change']:+.1f} {ci(p['change_ci'])} | {pct(a['with'])} | "
            f"{100 * a['change']:+.1f} {ci(a['change_ci'])} | {r['reading']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    """CLI: every scored model -> the summary JSON and the markdown table beside it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scores", action="append", required=True, help="MODEL=scores.csv")
    parser.add_argument("--out", default="experiments/results/robustness_v1/summary.json")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args()
    rows = []
    for spec in args.scores:
        model, path = spec.split("=", 1)
        rows += compare(pd.read_csv(path, dtype={"speaker": str}), model, args.n_boot)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n_boot": args.n_boot, "rows": rows}, indent=2), encoding="utf-8")
    out.with_suffix(".md").write_text(markdown(rows), encoding="utf-8")
    print(f"{len(rows)} rows -> {out} and {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
