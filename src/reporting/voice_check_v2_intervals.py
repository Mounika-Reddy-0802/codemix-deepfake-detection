"""Voice check v2's headline percentages with 95% intervals (reported, not judged).

The adoption rule (``voice_check_v2``) reads each percentage as a point. This adds how much
each could move with other test speakers: a speaker-level bootstrap (10,000 resamples of the
set's speakers with replacement, generator seed 20261108, percentile interval), the scheme of
the SSL generator study. A set with no speaker column (the reference's HiACC rows) gets a
Wilson interval over clips instead, marked as such; it treats clips as independent and is
narrower than a speaker interval would be. A set judged entirely right (or wrong) leaves the
bootstrap nothing to resample, so it gets an exact (Clopper-Pearson) interval over clips, and the
genuine calls get one over calls.

Every point is recomputed from the rows the rule used, the ``gen_train`` speakers left out of
the same sets, and checked against ``any_audio_accuracy.accuracy``::

    python -m src.reporting.voice_check_v2_intervals \\
        --out experiments/results/voice_check_v2_intervals.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.reporting import voice_check_v2 as rule

BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20261108
INTERVAL = (2.5, 97.5)
#: (name, label, set key, rows, expected) in table order, as ``voice_check_v2.measures``
MEASURES = (
    ("hiacc", "Spontaneous genuine (HiACC) judged real", "hiacc", ("bonafide", None), "bonafide"),
    (
        "read_new_speakers",
        "Read genuine, new speakers",
        "new_speakers",
        ("bonafide", None),
        "bonafide",
    ),
    ("read_control", "Read genuine, language control", "control", ("bonafide", None), "bonafide"),
    ("indicf5", "IndicF5 clones caught", "cm07", ("spoof", None), "spoof"),
    ("chatterbox", "Chatterbox clones caught", "cm09", ("spoof", None), "spoof"),
    ("tortoise_new", "Tortoise, new speakers", "new_speakers", ("spoof", "tortoise"), "spoof"),
    ("tortoise_cm04", "Tortoise, CM04", "cm04", ("spoof", None), "spoof"),
    ("rvc", "RVC conversions caught", "rvc_test", ("spoof", None), "spoof"),
    ("xtts", "XTTS-v2 clones caught", "new_speakers", ("spoof", "xtts_v2"), "spoof"),
)
CALLS = (("genuine_live", "Genuine live calls judged real"), ("genuine_replay", "Genuine replays"))


def selected(model: str, key: str, rows, exclude: frozenset[str]) -> pd.DataFrame | None:
    """The rows ``any_audio_accuracy.accuracy`` judges for one set (same filters)."""
    from src.reporting import any_audio_accuracy as acc

    d = acc.get(model, key)
    if d is None:
        return None
    lab, tool = rows
    d = d[d["label"].str.lower() == lab]
    if tool:
        d = d[d["tool"] == tool]
    if exclude and key in rule.POOL_SETS:
        d = d[~d["speaker"].astype(str).isin(exclude)]
    return None if d.empty else d


def speaker_interval(
    correct: np.ndarray, speakers: np.ndarray, n: int = BOOTSTRAP, seed: int = BOOTSTRAP_SEED
) -> tuple[float, float]:
    """Percentile interval of the percent correct over resampled speakers."""
    codes, inverse = np.unique(speakers, return_inverse=True)
    hits = np.bincount(inverse, weights=correct.astype(float), minlength=len(codes))
    sizes = np.bincount(inverse, minlength=len(codes)).astype(float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(codes), size=(n, len(codes)))
    pcts = 100 * hits[pick].sum(axis=1) / sizes[pick].sum(axis=1)
    low, high = np.percentile(pcts, INTERVAL)
    return float(low), float(high)


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for k of n, in percent."""
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(100 * (centre - half)), float(100 * (centre + half))


def exact(k: int, n: int) -> tuple[float, float]:
    """Clopper-Pearson interval for k of n, in percent."""
    from scipy.stats import beta

    low = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    high = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return float(100 * low), float(100 * high)


def measure(model: str, key: str, rows, expected: str, exclude: frozenset[str]) -> dict | None:
    """Point, interval and its method for one model and set."""
    from src.reporting import any_audio_accuracy as acc

    d = selected(model, key, rows, exclude)
    if d is None:
        return None
    thr = acc.threshold(model)
    correct = ((d["p"] >= thr) if expected == "bonafide" else (d["p"] < thr)).to_numpy()
    point = 100 * float(correct.mean())
    keep_out = exclude if key in rule.POOL_SETS else frozenset()
    check = acc.accuracy(model, key, rows, expected, keep_out)
    if check is None or abs(check[0] - point) > 1e-9 or check[1] != len(d):
        raise ValueError(f"{model} {key}: {point} does not match the rule's {check}")
    out = {"point": point, "n": len(d)}
    if correct.all() or not correct.any():
        out.update(method="clips (exact)", interval=list(exact(int(correct.sum()), len(d))))
    elif "speaker" in d.columns and d["speaker"].notna().all():
        speakers = d["speaker"].astype(str).to_numpy()
        out.update(method="speakers", speakers=len(set(speakers)))
        out["interval"] = list(speaker_interval(correct, speakers))
    else:
        out.update(method="clips (Wilson)", interval=list(wilson(int(correct.sum()), len(d))))
    return out


def call_measures(model: str) -> dict:
    """The genuine calls judged real, with exact intervals."""
    from src.reporting import any_audio_accuracy as acc

    counts = acc.call_counts(model) or {}
    out = {}
    for name, _ in CALLS:
        k, n = counts.get((name, "none"), (None, None))
        if k is not None and n:
            out[name] = {"point": 100 * k / n, "k": k, "n": n, "method": "calls (exact)"}
            out[name]["interval"] = list(exact(k, n))
    return out


def collect(models, exclude: frozenset[str]) -> dict:
    """``out[model][measure]`` for every model and measure present."""
    out = {}
    for m in models:
        out[m] = {}
        for name, _, key, rows, expected in MEASURES:
            r = measure(m, key, rows, expected, exclude)
            if r is not None:
                out[m][name] = r
        out[m].update(call_measures(m))
    return out


def _cell(r: dict | None) -> str:
    if r is None:
        return "-"
    lo, hi = r["interval"]
    mark = {"speakers": "", "clips (Wilson)": " w", "clips (exact)": " e", "calls (exact)": " c"}
    return f"{r['point']:.1f} [{lo:.1f}, {hi:.1f}]{mark[r['method']]}"


def markdown(results: dict, excluded: int) -> str:
    models = list(results)
    lines = [
        "# Voice check v2: headline percentages with 95% intervals",
        "",
        "Percent judged correctly at each model's frozen dev-only point, on the simulated G.711 "
        f"line, the {excluded} `gen_train` speakers left out of the sets that hold them. Brackets: "
        "95% speaker-bootstrap interval (10,000 resamples, seed 20261108); w: Wilson interval "
        "over clips (no speaker column; narrower than a speaker interval); e: exact interval over "
        "clips (no error to resample); c: exact interval over calls.",
        "",
        "| Measure | " + " | ".join(models) + " |",
        "|---|" + "---|" * len(models),
    ]
    for name, label, *_ in MEASURES:
        lines.append(
            f"| {label} | " + " | ".join(_cell(results[m].get(name)) for m in models) + " |"
        )
    for name, label in CALLS:
        lines.append(
            f"| {label} | " + " | ".join(_cell(results[m].get(name)) for m in models) + " |"
        )
    speakers = {
        name: results[models[-1]][name].get("speakers")
        for name, *_ in MEASURES
        if name in results[models[-1]]
    }
    lines += [
        "",
        "Speakers per set (last column): "
        + ", ".join(f"{k} {v}" for k, v in speakers.items() if v is not None)
        + ".",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    """CLI: the reference and both v2 seeds, with intervals; writes the table and a JSON."""
    import argparse

    from src.data import generator_study as gs

    parser = argparse.ArgumentParser(description="Voice check v2 with 95% intervals")
    parser.add_argument("--pools", default=f"data/manifests/{gs.POOLS_OUT}")
    parser.add_argument("--out", default="experiments/results/voice_check_v2_intervals.md")
    args = parser.parse_args()
    exclude = frozenset(gs.train_speakers(args.pools))
    results = collect((rule.REFERENCE, *rule.CANDIDATES), exclude)
    text = markdown(results, len(exclude))
    print(text)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
