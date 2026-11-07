"""Shortcut floors for the generator study (P-041), on the held-out speakers only.

Every claim in the project sits beside a matched floor: eight low-level statistics
(``src.training.floor_compare``) fitted on exactly the detector's own training data and scored
on exactly its test clips, the difference given a 95% interval from a bootstrap over whole
speakers. The study's four systems each train on a different manifest
(``genstudy_train_<system>.csv``), so each gets its own floor; the test clips are those of
the study's measures, **without the ``gen_train`` speakers**, as in the study's rule. The
procedure and the functions are ``src.training.floor_final``'s, unchanged; the features of
every clip the study adds are already in that module's cache.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import generator_study as gs
from src.training import floor_final as fl

#: the study's tool measures: Chatterbox, IndicF5 and Tortoise, both codecs where they exist
TEST_SETS = {
    k: fl.TEST_SETS[k] for k in ("cm09", "cm07", "new_tortoise", "cm07_amr", "new_tortoise_amr")
}


def held_out(frame: pd.DataFrame, train: set[str]) -> pd.DataFrame:
    """Rows of speakers outside the study's training half."""
    return frame[gs.test_speakers(frame["speaker"], train)].reset_index(drop=True)


def run(manifest_dir: str, index: dict[str, str], results: Path, n_boot: int = 1000) -> list[dict]:
    """Every system's floor against every seed of that system, on every study set."""
    from src.data import lowlevel_cue as lc

    mdir = Path(manifest_dir)
    train_half = gs.train_speakers(str(mdir / gs.POOLS_OUT))
    tests = {}
    for test_set, (manifest, tools, _) in TEST_SETS.items():
        frame = held_out(fl.select(pd.read_csv(mdir / manifest), tools), train_half)
        print(f"featurising {test_set} ({len(frame)} clips, held-out speakers)", flush=True)
        tests[test_set] = fl.featurise(frame, index)
    out = []
    for system in gs.SYSTEMS:
        train = pd.read_csv(mdir / f"genstudy_train_{system}.csv")
        print(f"featurising genstudy_train_{system} ({len(train)} clips)", flush=True)
        _, x_train, y_train = fl.featurise(train, index)
        scaler = lc.Standardiser().fit(x_train)
        w, b = lc.fit_logistic(scaler.transform(x_train), y_train)
        for test_set, (_, _, key) in TEST_SETS.items():
            rows, x_test, _ = tests[test_set]
            floor_scores = lc.predict(scaler.transform(x_test), w, b)
            for csv in sorted(results.glob(f"{system}_s*__{key}_scores.csv")):
                stats = fl.compare_one(rows, floor_scores, csv, n_boot, seed=0)
                stats.update(detector=system, run=csv.name.split("__")[0], test_set=test_set)
                out.append(stats)
                print(
                    f"  {stats['run']:14s} {test_set:17s} det {100 * stats['eer_a']:6.2f} "
                    f"floor {100 * stats['eer_b']:6.2f} diff {100 * stats['diff']:+6.2f} "
                    f"{stats['verdict']}",
                    flush=True,
                )
    return out


def markdown(summary: list[dict]) -> str:
    lines = [
        "# Shortcut floors for the generator study (v1)",
        "",
        "Each system's floor (eight low-level statistics) is fitted on that system's own training",
        "manifest and scored on the study's test clips without the `gen_train` speakers. Per seed,",
        "the detector-minus-floor difference has a 95% interval from a bootstrap over whole",
        "speakers; 'beats floor' counts the seeds whose whole interval is below zero. EER (%) are",
        "means over seeds. Built by `src/training/floor_genstudy.py`.",
        "",
        "| System | Test set | Seeds | Detector EER | Floor EER | Difference | Beats floor | Worse |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in summary:
        lines.append(
            f"| {r['detector']} | {r['test_set']} | {r['seeds']} | {100 * r['detector_eer']:.2f} "
            f"| {100 * r['floor_eer']:.2f} | {100 * r['diff']:+.2f} | {r['beats_floor']}/{r['seeds']} "
            f"| {r['worse_than_floor']}/{r['seeds']} |"
        )
    above = sorted({r["test_set"] for r in summary if r["floor_eer"] > 0.5})
    if above:
        lines += [
            "",
            f"The floor is above chance on {', '.join(above)}, so beating it there is a weak bar.",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Shortcut floors for the generator study")
    parser.add_argument("--manifests", default="data/manifests")
    parser.add_argument(
        "--phone-dirs", nargs="+", required=True, help="G.711 clip folders, in order"
    )
    parser.add_argument("--amr-dirs", nargs="+", required=True, help="AMR-NB clip folders")
    parser.add_argument("--results", required=True, help="folder of the study's score files")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--cache", help="per-clip feature cache folder (floor_final's)")
    parser.add_argument("--out", default="experiments/results/floor_genstudy_v1.json")
    parser.add_argument("--doc", default="docs/results/floor_genstudy_v1.md")
    args = parser.parse_args()

    fl.CACHE = Path(args.cache) if args.cache else None
    index = fl.clip_index(args.phone_dirs, args.amr_dirs)
    print(f"{len(index)} clips indexed", flush=True)
    results = run(args.manifests, index, Path(args.results), args.n_boot)
    summary = fl.summarise(results) if results else []
    Path(args.out).write_text(
        json.dumps(
            {"n_boot": args.n_boot, "results": results, "summary": summary},
            indent=2,
            default=lambda v: v.tolist() if isinstance(v, np.ndarray) else str(v),
        ),
        encoding="utf-8",
    )
    Path(args.doc).write_text(markdown(summary), encoding="utf-8")
    print(markdown(summary))


if __name__ == "__main__":
    main()
