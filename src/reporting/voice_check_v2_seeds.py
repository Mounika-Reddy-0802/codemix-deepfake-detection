"""Voice check v2 over four seeds (P-045): reported, not judged.

Seeds 1234 and 2025 decided the adoption (P-043). Seeds 7 and 42 were added afterwards with the
same notebook and the same pinned commit, only the seed changed, so that the numbers carried into
the paper rest on four trainings rather than two. Nothing here can change the adoption, the voice
check's checkpoint (seed 1234) or its threshold. Every seed found is reported and none is dropped:
per measure each seed's value, the mean, the standard deviation and the range.

Every EER is on the G.711 line or the real AMR-NB codec and on speakers outside ``gen_train``
(``generator_study.held_out_eer``), the same code the adoption and the generator studies use. The
false-alarm rates are the notebook's own (genuine clips called fake at its fixed score 0)::

    python -m src.reporting.voice_check_v2_seeds --results <run folders> \\
        --out experiments/results/voice_check_v2_seeds.md
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import numpy as np

from src.data import generator_study as gs
from src.reporting import generator_study as adapter_study

RUN = "vcv2"
SEEDS = (1234, 2025, 7, 42)
#: measure -> row label, in table order; EERs first, then the notebook's false-alarm rates
ROWS = {
    "indicf5": "IndicF5 clones (CM07), G.711",
    "indicf5_amr": "IndicF5 clones (CM07), AMR-NB",
    "chatterbox": "Chatterbox clones (CM09), G.711",
    "new_tortoise": "Tortoise, new speakers, G.711",
    "new_tortoise_amr": "Tortoise, new speakers, AMR-NB",
    "cm04": "Tortoise, CM04, G.711",
    "cm04_amr": "Tortoise, CM04, AMR-NB",
    "rvc_test": "RVC, held-out half, G.711",
    "new_xtts": "XTTS-v2, new speakers, G.711",
    "eval_pool": "XTTS-v2, evaluation pool, G.711",
    "control_fa": "Language-control genuine called fake",
    "hiacc_fa": "HiACC spontaneous genuine called fake",
    "iv_test_fa": "IndicVoices spontaneous genuine called fake",
}
RATES = ("control_fa", "hiacc_fa", "iv_test_fa")


def collect(*folders: Path, train: set[str]) -> dict[int, dict]:
    """``table[seed]`` for every scored v2 run in the folders; a seed found twice stops."""
    table: dict[int, dict] = {}
    for folder in folders:
        for r in json.loads((folder / "runs.json").read_text(encoding="utf-8")):
            if r.get("data") != RUN or r.get("status") not in adapter_study.SCORED:
                continue
            seed = int(r["seed"])
            if seed in table:
                raise ValueError(f"seed {seed} appears in two results folders")
            row = adapter_study.measure_run(folder, f"{RUN}_s{seed}", train)
            row.update({k: r.get(k) for k in RATES})
            table[seed] = row
    return table


def summary(table: dict[int, dict]) -> dict[str, dict]:
    """Per measure: the seeds' values, mean, SD (n - 1) and range; measures with no value skipped."""
    out: dict[str, dict] = {}
    for key in ROWS:
        vals = {s: table[s][key] for s in sorted(table) if table[s].get(key) is not None}
        if not vals:
            continue
        v = list(vals.values())
        out[key] = {
            "seeds": vals,
            "mean": mean(v),
            "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
            "min": min(v),
            "max": max(v),
            "n": len(v),
        }
    return out


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.2f}"


def markdown(table: dict[int, dict], summ: dict[str, dict], excluded: int) -> str:
    """The four-seed table, in percent."""
    seeds = [s for s in SEEDS if s in table] + sorted(set(table) - set(SEEDS))
    missing = [s for s in SEEDS if s not in table]
    lines = [
        "# Voice check v2 over four seeds",
        "",
        f"EER (%) on speakers outside `gen_train` ({excluded} left out), and genuine clips "
        "called fake (%) at the notebook's fixed score. Seeds 1234 and 2025 decided the "
        "adoption (P-043); 7 and 42 were added after it (P-045). Reported, not judged.",
        "",
    ]
    if missing:
        lines += [f"**Missing seeds: {', '.join(map(str, missing))}.**", ""]
    lines += [
        "| Measure | " + " | ".join(f"s{s}" for s in seeds) + " | mean ± SD | range |",
        "|---|" + "---|" * (len(seeds) + 2),
    ]
    for key, label in ROWS.items():
        if key not in summ:
            continue
        m = summ[key]
        cells = [_pct(table[s].get(key)) for s in seeds]
        lines.append(
            f"| {label} | "
            + " | ".join(cells)
            + f" | {_pct(m['mean'])} ± {_pct(m['sd'])} | {_pct(m['min'])}-{_pct(m['max'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    """CLI: read the run folders, write the table (and a JSON beside it)."""
    import argparse

    parser = argparse.ArgumentParser(description="Voice check v2 over four seeds (P-045)")
    parser.add_argument("--results", nargs="+", required=True, help="run folders")
    parser.add_argument("--pools", default=f"data/manifests/{gs.POOLS_OUT}")
    parser.add_argument("--out", default="experiments/results/voice_check_v2_seeds.md")
    args = parser.parse_args()
    train = gs.train_speakers(args.pools)
    table = collect(*map(Path, args.results), train=train)
    summ = summary(table)
    text = markdown(table, summ, len(train))
    print(text)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    payload = {"seeds": sorted(table), "per_seed": table, "summary": summ}
    out.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
