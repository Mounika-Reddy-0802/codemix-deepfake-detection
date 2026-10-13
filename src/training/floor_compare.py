"""Detector against shortcut floor on the same clips, with a paired speaker bootstrap.

The external review (comment 5) asked how the floor should be read, and how far the
recording-domain differences left after normalisation undermine a detector-versus-
floor comparison. Two weaknesses in the original comparison make the question fair.

1. **The two sides saw different information.** The CM04 floor was fitted on half of
   CM04 itself, Tortoise fakes included, and scored on the other half. The detector
   never sees Tortoise and is scored on all of CM04. The methodology says the floor is
   fitted on the adaptation pool. Here every detector gets its own floor: the same
   eight statistics, fitted on exactly the manifest that detector was trained on, and
   scored on exactly the clips the detector was scored on. The two then differ only in
   what they can hear.
2. **No paired test.** Two EERs with overlapping intervals were compared by eye. Here
   the difference (detector minus floor) gets one interval from a bootstrap that
   resamples *speakers* -- all of a speaker's clips in or out together. Clips from one
   speaker are not independent, and the test sets hold only 12-15 speakers, so a
   clip-level interval is too narrow.

Reading the result. A difference whose whole interval lies below zero means the
detector does better than the eight statistics can from the same training data. It
does not prove the detector ignores recording provenance: a large network can exploit
subtler domain cues than these eight. It rules out the cheapest explanation and no
more. An interval that straddles zero is inconclusive, not a failure.

Numpy and pandas only, like the floor and the metrics (P-004): this produces paper
numbers, so it is tested here rather than trusted to a dependency CI never installs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.training.metrics import eer

#: Verdicts for a detector-minus-floor interval.
BEATS_FLOOR = "beats floor"
WORSE_THAN_FLOOR = "worse than floor"
INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class System:
    """One row of the paper's main table: what it was trained on, where its scores are."""

    name: str
    train_manifest: str
    condition: str  # "clean" or "channel"
    scores_prefix: str  # "<prefix>__<test set>_scores.csv"


#: The eight S2/S3 systems of the main results table.
SYSTEMS: tuple[System, ...] = (
    System(
        "S2 LoRA, XTTS, clean", "codemix_adapt_train.csv", "clean", "results/w10/lora_norm_clean"
    ),
    System(
        "S2 LoRA, XTTS, channel",
        "codemix_adapt_train_channel20.csv",
        "channel",
        "results/w10/lora_norm_channel",
    ),
    System(
        "S2 LoRA, XTTS+RVC, clean",
        "codemix_adapt_train_rvc.csv",
        "clean",
        "results/w10/lora_norm_rvc_clean",
    ),
    System(
        "S2 LoRA, XTTS+RVC, channel (deployed)",
        "codemix_adapt_train_rvc_channel20.csv",
        "channel",
        "results/w10/lora_norm_rvc_channel",
    ),
    System(
        "S3 native, XTTS, clean",
        "codemix_adapt_train.csv",
        "clean",
        "docs/results/s3_native/s3_native_clean",
    ),
    System(
        "S3 native, XTTS, channel",
        "codemix_adapt_train_channel20.csv",
        "channel",
        "docs/results/s3_native/s3_native_channel",
    ),
    System(
        "S3 native, XTTS+RVC, clean",
        "codemix_adapt_train_rvc.csv",
        "clean",
        "docs/results/s3_native/s3_native_rvc_clean",
    ),
    System(
        "S3 native, XTTS+RVC, channel",
        "codemix_adapt_train_rvc_channel20.csv",
        "channel",
        "docs/results/s3_native/s3_native_rvc_channel",
    ),
)

#: Test set -> (clean manifest, channel manifest).
TEST_SETS: dict[str, tuple[str, str]] = {
    "eval_pool": ("codemix_eval.csv", "codemix_eval_channel20.csv"),
    "cm04": ("score_cm04_norm.csv", "score_cm04_norm_channel20.csv"),
    "rvc_test": ("rvc_holdout_test.csv", "rvc_holdout_test_channel20.csv"),
}


def binary_labels(labels: pd.Series) -> np.ndarray:
    """``bonafide`` -> 1, anything else -> 0, matching the floor and the metrics."""
    return labels.astype(str).str.lower().eq("bonafide").astype(np.int64).to_numpy()


def speaker_bootstrap(
    labels: np.ndarray,
    speakers: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    ci: float = 0.95,
) -> dict:
    """Paired bootstrap over speakers for EER(a), EER(b) and EER(a) - EER(b).

    Each resample draws speakers with replacement and takes all of a drawn speaker's
    clips, for both systems at once, so the two EERs in a resample are computed on
    identical clips. Resamples that lose a class are skipped and counted.

    The two-sided p-value is twice the smaller tail of the resampled differences
    around zero (a percentile bootstrap test), floored at ``1 / n_used``.
    """
    labels = np.asarray(labels, dtype=np.int64)
    speakers = np.asarray(speakers).astype(str)
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if not (len(labels) == len(speakers) == len(a) == len(b)):
        raise ValueError("labels, speakers and both score arrays must be the same length")

    unique = np.unique(speakers)
    members = [np.flatnonzero(speakers == s) for s in unique]
    rng = np.random.default_rng(seed)
    eer_a, eer_b = [], []
    skipped = 0
    for _ in range(n_boot):
        drawn = rng.integers(0, len(unique), len(unique))
        idx = np.concatenate([members[i] for i in drawn])
        y = labels[idx]
        if y.min() == y.max():
            skipped += 1
            continue
        eer_a.append(eer(a[idx], y))
        eer_b.append(eer(b[idx], y))
    if not eer_a:
        raise ValueError("every bootstrap resample lost a class")

    ea, eb = np.asarray(eer_a), np.asarray(eer_b)
    diff = ea - eb
    tail = (1.0 - ci) / 2.0
    q = [100 * tail, 100 * (1.0 - tail)]
    used = len(diff)
    p = 2.0 * min(np.mean(diff <= 0), np.mean(diff >= 0))
    return {
        "eer_a": eer(a, labels),
        "eer_a_ci": [float(v) for v in np.percentile(ea, q)],
        "eer_b": eer(b, labels),
        "eer_b_ci": [float(v) for v in np.percentile(eb, q)],
        "diff": eer(a, labels) - eer(b, labels),
        "diff_ci": [float(v) for v in np.percentile(diff, q)],
        "p_value": float(min(1.0, max(p, 1.0 / used))),
        "n_speakers": int(len(unique)),
        "n_boot_used": int(used),
        "n_boot_skipped": int(skipped),
    }


def verdict(diff_ci: list[float]) -> str:
    """Read a detector-minus-floor interval: below zero, above zero, or straddling it."""
    lo, hi = diff_ci
    if hi < 0:
        return BEATS_FLOOR
    if lo > 0:
        return WORSE_THAN_FLOOR
    return INCONCLUSIVE


def per_speaker_eer(labels: np.ndarray, speakers: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    """EER for each speaker on its own clips; NaN where a speaker lacks one class."""
    frame = pd.DataFrame(
        {"label": labels, "speaker": np.asarray(speakers).astype(str), "score": scores}
    )
    rows = []
    for speaker, group in frame.groupby("speaker", sort=True):
        y = group["label"].to_numpy()
        n_bona, n_spoof = int(y.sum()), int((y == 0).sum())
        value = eer(group["score"].to_numpy(), y) if n_bona and n_spoof else float("nan")
        rows.append({"speaker": speaker, "n_bonafide": n_bona, "n_spoof": n_spoof, "eer": value})
    return pd.DataFrame(rows)


def pair_scores(
    test_rows: pd.DataFrame, floor_scores: np.ndarray, detector: pd.DataFrame
) -> pd.DataFrame:
    """Join floor and detector scores clip by clip on ``filepath``.

    Raises if the two do not cover the same clips: a comparison on different clips is
    exactly what this module exists to stop.
    """
    floor = test_rows[["filepath", "label", "speaker"]].copy()
    floor["floor"] = floor_scores
    det = detector[["filepath", "score"]].rename(columns={"score": "detector"})
    joined = floor.merge(det, on="filepath", how="inner", validate="one_to_one")
    if len(joined) != len(floor) or len(joined) != len(det):
        raise ValueError(
            f"floor ({len(floor)}) and detector ({len(det)}) cover different clips; "
            f"only {len(joined)} match"
        )
    return joined


def run(
    manifest_dir: str,
    clean_root: str,
    channel_root: str,
    systems: tuple[System, ...] = SYSTEMS,
    n_boot: int = 1000,
    seed: int = 0,
    scores_dir: str | None = None,
) -> tuple[list[dict], pd.DataFrame]:
    """Fit each system's floor on its training manifest, compare on every test set."""
    from src.data import lowlevel_cue as lc

    roots = {"clean": clean_root, "channel": channel_root}
    cache: dict[tuple[str, str], tuple[pd.DataFrame, np.ndarray, np.ndarray]] = {}

    def featurise(manifest: str, root: str):
        key = (manifest, root)
        if key not in cache:
            frame = pd.read_csv(Path(manifest_dir) / manifest)
            print(f"featurising {manifest} ({len(frame)} clips) from {root}", flush=True)
            x, y, kept = lc.extract(frame, root, return_index=True)
            cache[key] = (frame.iloc[kept].reset_index(drop=True), x, y)
        return cache[key]

    results: list[dict] = []
    speaker_rows: list[pd.DataFrame] = []
    for system in systems:
        root = roots[system.condition]
        _, x_train, y_train = featurise(system.train_manifest, root)
        scaler = lc.Standardiser().fit(x_train)
        w, b = lc.fit_logistic(scaler.transform(x_train), y_train)
        for test_set, manifests in TEST_SETS.items():
            manifest = manifests[0] if system.condition == "clean" else manifests[1]
            test_rows, x_test, _ = featurise(manifest, root)
            floor_scores = lc.predict(scaler.transform(x_test), w, b)
            detector = pd.read_csv(f"{system.scores_prefix}__{test_set}_scores.csv")
            paired = pair_scores(test_rows, floor_scores, detector)
            labels = binary_labels(paired["label"])
            stats = speaker_bootstrap(
                labels,
                paired["speaker"].to_numpy(),
                paired["detector"].to_numpy(),
                paired["floor"].to_numpy(),
                n_boot=n_boot,
                seed=seed,
            )
            stats.update(
                {
                    "system": system.name,
                    "test_set": test_set,
                    "condition": system.condition,
                    "clips": int(len(paired)),
                    "verdict": verdict(stats["diff_ci"]),
                    "floor_fit_on": system.train_manifest,
                }
            )
            results.append(stats)
            print(
                f"  {system.name:40s} {test_set:9s} detector {100 * stats['eer_a']:6.2f}  "
                f"floor {100 * stats['eer_b']:6.2f}  diff {100 * stats['diff']:+6.2f} "
                f"[{100 * stats['diff_ci'][0]:+6.2f}, {100 * stats['diff_ci'][1]:+6.2f}]  "
                f"{stats['verdict']}",
                flush=True,
            )

            det_sp = per_speaker_eer(labels, paired["speaker"], paired["detector"].to_numpy())
            flo_sp = per_speaker_eer(labels, paired["speaker"], paired["floor"].to_numpy())
            det_sp = det_sp.rename(columns={"eer": "eer_detector"})
            det_sp["eer_floor"] = flo_sp["eer"].to_numpy()
            det_sp.insert(0, "test_set", test_set)
            det_sp.insert(0, "system", system.name)
            speaker_rows.append(det_sp)
            if scores_dir:
                out = Path(scores_dir)
                out.mkdir(parents=True, exist_ok=True)
                stem = Path(system.scores_prefix).name
                paired.to_csv(out / f"{stem}__{test_set}_paired.csv", index=False)
    return results, pd.concat(speaker_rows, ignore_index=True)


def main() -> None:
    """CLI: ``python -m src.training.floor_compare --clean-root ... --channel-root ...``."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Detector vs shortcut floor, paired by speaker")
    parser.add_argument("--manifests", default="data/manifests")
    parser.add_argument("--clean-root", required=True, help="normalised clean bundle")
    parser.add_argument("--channel-root", required=True, help="normalised G.711 channel bundle")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="experiments/results/floor_compare_v2.json")
    parser.add_argument("--scores-dir", default="experiments/results/floor_v2")
    args = parser.parse_args()

    results, speakers = run(
        args.manifests,
        args.clean_root,
        args.channel_root,
        n_boot=args.n_boot,
        seed=args.seed,
        scores_dir=args.scores_dir,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"n_boot": args.n_boot, "seed": args.seed, "results": results}, indent=2),
        encoding="utf-8",
    )
    speakers.to_csv(Path(args.scores_dir) / "per_speaker_eer.csv", index=False)
    print(f"\nwrote {args.out} and {Path(args.scores_dir) / 'per_speaker_eer.csv'}")


if __name__ == "__main__":
    main()
