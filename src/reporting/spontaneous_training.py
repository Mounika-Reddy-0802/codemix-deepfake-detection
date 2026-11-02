"""Decision D1 part (b): the spontaneous recipe judged against its hypotheses.

The hypotheses were written before launch (``docs/results/spontaneous_training_v1.md``) and
are coded here unchanged:

- **H1** (HiACC, never trained on): genuine HiACC clips called fake at the fixed rule fall by
  at least 10 points on the mean over seeds; for the adapter, seed 1234 is also below its
  frozen 43.9%; for SSL-AASIST, neither seed is more than 2 points above its own frozen seed.
- **H2** (unseen generators): paired by seed, the mean G.711 EER rises by at most 2.0 points on
  CM04 (Tortoise) and CM07 (IndicF5), and for SSL-AASIST also on CM09 (Chatterbox).

Verdict per family: *helps* (H1 and H2), *trade-off* (H1 only), *does not help* (H1 fails),
*incomplete* (a planned seed has no result). Every value is a fraction, as the notebooks'
``runs.json`` and the frozen runs store them; the frozen adapter's HiACC value exists for seed
1234 only (``hiacc_false_alarm_v1/final_s1234_channel.json``)::

    python -m src.reporting.spontaneous_training --adapter-runs a/runs.json b/runs.json \\
        --ssl-runs s1234/runs.json s2025/runs.json
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

FROZEN_ADAPTER = "experiments/results/final_v1/runs.json"
FROZEN_SSL = "experiments/results/encoder_spontaneous_v1/runs.json"
FROZEN_ADAPTER_HIACC = "experiments/results/hiacc_false_alarm_v1/final_s1234_channel.json"

FAMILIES = {
    "adapter": {"seeds": (1234, 2025, 7), "unseen": ("cm04", "cm07"), "h1_drop": 0.10},
    "ssl": {"seeds": (1234, 2025), "unseen": ("cm04", "cm07", "cm09"), "h1_drop": 0.10},
}
H1_SEED_SLACK = 0.02  # SSL-AASIST: no seed more than 2 points above its frozen seed
H2_MAX_RISE = 0.02  # mean paired EER rise allowed on each unseen-generator set
#: reported, not judged
REPORTED = (
    "eval_pool",
    "rvc_test",
    "new_xtts",
    "new_tortoise",
    "control_fa",
    "iv_test",
    "iv_test_fa",
    "iv_test_amr",
    "cm04_amr",
    "cm07_amr",
)


def load_runs(paths: list[str]) -> dict[int, dict]:
    """Seed -> run row, from one or more ``runs.json`` files (later files win)."""
    rows = {}
    for path in paths:
        for row in json.loads(Path(path).read_text(encoding="utf-8")):
            rows[int(row["seed"])] = row
    return rows


def frozen_runs(family: str, root: str = ".") -> dict[int, dict]:
    """The frozen final-recipe runs of one family, by seed, with the adapter's HiACC value."""
    base = Path(root)
    if family == "adapter":
        rows = {
            int(r["seed"]): dict(r)
            for r in json.loads((base / FROZEN_ADAPTER).read_text(encoding="utf-8"))
            if r.get("system") == "final"
        }
        hiacc = json.loads((base / FROZEN_ADAPTER_HIACC).read_text(encoding="utf-8"))
        rows[1234]["hiacc_fa"] = hiacc["summary"]["false_alarm_rate"]
        return rows
    return {
        int(r["seed"]): dict(r)
        for r in json.loads((base / FROZEN_SSL).read_text(encoding="utf-8"))
        if r.get("data") == "final"
    }


def _value(row: dict | None, key: str) -> float | None:
    v = (row or {}).get(key)
    return None if v is None else float(v)


def judge(family: str, new: dict[int, dict], frozen: dict[int, dict]) -> dict:
    """H1, H2 and the verdict of one family; ``new`` and ``frozen`` map seed -> run row."""
    spec = FAMILIES[family]
    seeds = spec["seeds"]
    have = [s for s in seeds if _value(new.get(s), "hiacc_fa") is not None]
    out = {"family": family, "planned_seeds": list(seeds), "seeds_with_results": have, "sets": {}}
    for key in ("hiacc_fa", *spec["unseen"], *REPORTED):
        per_seed = {
            s: {"new": _value(new.get(s), key), "frozen": _value(frozen.get(s), key)} for s in seeds
        }
        pairs = [(v["new"], v["frozen"]) for v in per_seed.values() if None not in v.values()]
        news = [v["new"] for v in per_seed.values() if v["new"] is not None]
        out["sets"][key] = {
            "per_seed": per_seed,
            "new_mean": mean(news) if news else None,
            "paired_mean_change": mean(n - f for n, f in pairs) if pairs else None,
            "paired_seeds": len(pairs),
        }
    if len(have) < len(seeds):
        out.update(h1=None, h2=None, verdict="incomplete")
        return out

    hiacc = out["sets"]["hiacc_fa"]
    frozen_hiacc = [_value(frozen.get(s), "hiacc_fa") for s in seeds]
    frozen_mean = mean(v for v in frozen_hiacc if v is not None)
    h1 = hiacc["new_mean"] <= frozen_mean - spec["h1_drop"]
    if family == "adapter":
        h1 = h1 and hiacc["per_seed"][1234]["new"] < hiacc["per_seed"][1234]["frozen"]
    else:
        h1 = h1 and all(v["new"] <= v["frozen"] + H1_SEED_SLACK for v in hiacc["per_seed"].values())
    rises = {k: out["sets"][k]["paired_mean_change"] for k in spec["unseen"]}
    h2 = all(r is not None and r <= H2_MAX_RISE for r in rises.values())
    out.update(
        h1=bool(h1),
        h1_frozen_mean=frozen_mean,
        h2=bool(h2),
        h2_rises=rises,
        verdict="helps" if h1 and h2 else ("trade-off" if h1 else "does not help"),
    )
    return out


def main() -> None:
    """CLI: judge whichever families have run files; write one JSON."""
    import argparse

    parser = argparse.ArgumentParser(description="Spontaneous recipe against its hypotheses")
    parser.add_argument("--adapter-runs", nargs="*", default=[])
    parser.add_argument("--ssl-runs", nargs="*", default=[])
    parser.add_argument("--out", default="experiments/results/spontaneous_training_v1.json")
    args = parser.parse_args()
    report = {}
    for family, paths in (("adapter", args.adapter_runs), ("ssl", args.ssl_runs)):
        if paths:
            report[family] = judge(family, load_runs(paths), frozen_runs(family))
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for family, r in report.items():
        print(f"\n{family}: verdict {r['verdict']} (seeds {r['seeds_with_results']})")
        for key, v in r["sets"].items():
            cells = "  ".join(
                f"s{s} {100 * p['new']:.2f}"
                + ("" if p["frozen"] is None else f" (frozen {100 * p['frozen']:.2f})")
                for s, p in v["per_seed"].items()
                if p["new"] is not None
            )
            change = v["paired_mean_change"]
            change = "" if change is None else f"  paired change {100 * change:+.2f}"
            print(f"  {key:12s} {cells}{change}")


if __name__ == "__main__":
    main()
