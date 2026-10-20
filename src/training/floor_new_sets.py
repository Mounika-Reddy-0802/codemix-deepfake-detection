"""Detector against shortcut floor on the week-13 sets (new speakers, Tortoise, IndicF5).

``src.training.floor_compare`` answered review comment 5 on the original test sets: each
detector gets a floor of eight low-level statistics fitted on exactly its own training
manifest, the two are scored on exactly the same clips, and the difference gets an
interval from a bootstrap that resamples whole speakers. This module applies the same
procedure, with the same functions, to the sets made in week 13:

* ``new_xtts`` - the 60 new test speakers' genuine clips with CM05 (XTTS);
* ``new_tortoise`` - the same genuine clips with CM06 (Tortoise, unseen tool);
* ``cm07`` - the IndicF5 speakers' all-Hindi genuine clips with CM07 (IndicF5, unseen).

Two differences from the original run, both forced by where the clips live: the floor
is fitted on the training bundle and applied to the extension bundle (the clips are
the same kind, normalised and rendered identically), and the detector scores come from
``experiments/results/ext_scoring_v1`` (laptop scoring of the four S2 systems).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.training import floor_compare as fc

SCORES = "experiments/results/ext_scoring_v1"


@dataclass(frozen=True)
class System:
    name: str
    train_manifest: str
    condition: str  # "clean" or "channel"
    tag: str  # prefix of "<tag>__<scoring set>_<condition>_scores.csv"


SYSTEMS: tuple[System, ...] = (
    System("S2 LoRA, XTTS, clean", "codemix_adapt_train.csv", "clean", "s2_clean"),
    System("S2 LoRA, XTTS, channel", "codemix_adapt_train_channel20.csv", "channel", "s2_channel"),
    System("S2 LoRA, XTTS+RVC, clean", "codemix_adapt_train_rvc.csv", "clean", "s2_rvc_clean"),
    System(
        "S2 LoRA, XTTS+RVC, channel (deployed)",
        "codemix_adapt_train_rvc_channel20.csv",
        "channel",
        "deployed_s2",
    ),
)

#: Test set -> (scoring set, spoof tools kept); genuine rows are always kept.
TEST_SETS: dict[str, tuple[str, tuple[str, ...]]] = {
    "new_xtts": ("score_ext", ("xtts_v2",)),
    "new_tortoise": ("score_ext", ("tortoise",)),
    "cm07": ("score_cm07", ("indicf5",)),
}


def select(frame: pd.DataFrame, tools: tuple[str, ...]) -> pd.DataFrame:
    """Genuine rows plus the spoof rows of ``tools``."""
    genuine = frame["label"].str.lower() == "bonafide"
    return frame[genuine | frame["tool"].isin(tools)].reset_index(drop=True)


def run(
    manifest_dir: str,
    train_roots: dict,
    test_roots: dict,
    scores_dir: str = SCORES,
    systems: tuple[System, ...] = SYSTEMS,
    n_boot: int = 1000,
    seed: int = 0,
) -> list[dict]:
    from src.data import lowlevel_cue as lc

    cache: dict = {}

    def featurise(frame: pd.DataFrame, key: tuple, root: str):
        if key not in cache:
            print(f"featurising {key[0]} ({len(frame)} clips) from {root}", flush=True)
            x, y, kept = lc.extract(frame, root, return_index=True)
            cache[key] = (frame.iloc[kept].reset_index(drop=True), x, y)
        return cache[key]

    mdir = Path(manifest_dir)
    results = []
    for system in systems:
        cond = system.condition
        train = pd.read_csv(mdir / system.train_manifest)
        _, x_train, y_train = featurise(train, (system.train_manifest, cond), train_roots[cond])
        scaler = lc.Standardiser().fit(x_train)
        w, b = lc.fit_logistic(scaler.transform(x_train), y_train)
        for test_set, (scoring_set, tools) in TEST_SETS.items():
            suffix = "" if cond == "clean" else "_channel20"
            det_path = Path(scores_dir) / f"{system.tag}__{scoring_set}_{cond}_scores.csv"
            if not det_path.is_file():
                print(f"  [missing] {det_path}", flush=True)
                continue
            frame = select(pd.read_csv(mdir / f"{scoring_set}{suffix}.csv"), tools)
            rows, x_test, _ = featurise(frame, (test_set, cond), test_roots[cond])
            floor_scores = lc.predict(scaler.transform(x_test), w, b)
            detector = pd.read_csv(det_path)
            detector = detector[detector["filepath"].isin(rows["filepath"])]
            paired = fc.pair_scores(rows, floor_scores, detector)
            labels = fc.binary_labels(paired["label"])
            stats = fc.speaker_bootstrap(
                labels,
                paired["speaker"].to_numpy(),
                paired["detector"].to_numpy(),
                paired["floor"].to_numpy(),
                n_boot=n_boot,
                seed=seed,
            )
            stats.update(
                system=system.name,
                test_set=test_set,
                condition=cond,
                clips=int(len(paired)),
                verdict=fc.verdict(stats["diff_ci"]),
                floor_fit_on=system.train_manifest,
            )
            results.append(stats)
            print(
                f"  {system.name:40s} {test_set:13s} detector {100 * stats['eer_a']:6.2f}  "
                f"floor {100 * stats['eer_b']:6.2f}  diff {100 * stats['diff']:+6.2f} "
                f"[{100 * stats['diff_ci'][0]:+6.2f}, {100 * stats['diff_ci'][1]:+6.2f}]  "
                f"{stats['verdict']}",
                flush=True,
            )
    return results


def markdown(results: list[dict]) -> str:
    lines = [
        "# Detector against shortcut floor on the week-13 sets (v1)",
        "",
        "Same procedure as `shortcut_floor_v2.md` (`src/training/floor_compare.py`): each",
        "system's floor is fitted on its own training manifest and scored on the same clips",
        "as the detector; the difference (detector - floor, EER points) has a 95% interval",
        "from a bootstrap over whole speakers. Built by `src/training/floor_new_sets.py`.",
        "",
        "| System | Test set | Detector EER | Floor EER | Difference [95% CI] | Verdict |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['system']} | {r['test_set']} | {100 * r['eer_a']:.2f} | {100 * r['eer_b']:.2f} | "
            f"{100 * r['diff']:+.2f} [{100 * r['diff_ci'][0]:+.2f}, {100 * r['diff_ci'][1]:+.2f}] | "
            f"{r['verdict']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Detector vs floor on the week-13 sets")
    parser.add_argument("--manifests", default="data/manifests")
    parser.add_argument("--train-clean-root", required=True)
    parser.add_argument("--train-channel-root", required=True)
    parser.add_argument("--test-clean-root", required=True)
    parser.add_argument("--test-channel-root", required=True)
    parser.add_argument("--scores", default=SCORES)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--out", default="experiments/results/floor_new_sets_v1.json")
    parser.add_argument("--doc", default="docs/results/floor_new_sets_v1.md")
    args = parser.parse_args()
    results = run(
        args.manifests,
        {"clean": args.train_clean_root, "channel": args.train_channel_root},
        {"clean": args.test_clean_root, "channel": args.test_channel_root},
        args.scores,
        n_boot=args.n_boot,
    )
    Path(args.out).write_text(
        json.dumps(
            {"n_boot": args.n_boot, "results": results},
            indent=2,
            default=lambda o: float(o) if isinstance(o, np.floating) else str(o),
        )
    )
    Path(args.doc).write_text(markdown(results), encoding="utf-8")
    print(markdown(results))


if __name__ == "__main__":
    main()
