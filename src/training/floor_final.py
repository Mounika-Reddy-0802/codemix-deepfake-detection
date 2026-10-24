"""Shortcut floors for the two systems trained on the final-recipe data.

The final recipe (248 genuine speakers, every clip in both simulated G.711 and real AMR-NB)
trains two detectors in week 13: the wav2vec2-base LoRA adapter (run 4) and SSL-AASIST, the
larger encoder (runs 7 and 8). Both see exactly ``codemix_adapt_train_final.csv``, so one
floor serves both: the same eight low-level statistics as ``src.training.floor_compare``,
fitted on that manifest, scored on exactly the clips each detector was scored on, and the
difference (detector minus floor) given an interval from a bootstrap over whole speakers.

What is new here is only where the clips live. The Kaggle runs linked every dataset into
one folder; on the laptop the clips sit in several bundle folders, so each manifest path
(``${DATA_ROOT}/clips/<name>`` or ``${DATA_ROOT}/clips/amr/<name>``) is found by name in an
ordered list of folders, first match kept, the same rule the Kaggle runs used. The
``filepath`` column itself is left untouched, which is what the detector scores join on.

Every seed of every detector gets its own comparison; the summary counts the seeds whose
whole interval lies below zero. Nothing here is chosen on a test set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.training import floor_compare as fc

TRAIN_MANIFEST = "codemix_adapt_train_final.csv"

#: Test set -> (scoring manifest, spoof tools kept or None for all, detector score key).
TEST_SETS: dict[str, tuple[str, tuple[str, ...] | None, str]] = {
    "eval_pool": ("codemix_eval_channel20.csv", None, "eval_pool"),
    "cm04": ("score_cm04_norm_channel20.csv", None, "cm04"),
    "rvc_test": ("rvc_holdout_test_channel20.csv", None, "rvc_test"),
    "new_xtts": ("score_ext_channel20.csv", ("xtts_v2",), "new_speakers"),
    "new_tortoise": ("score_ext_channel20.csv", ("tortoise",), "new_speakers"),
    "cm07": ("score_cm07_channel20.csv", None, "cm07"),
    "new_xtts_amr": ("score_ext_amr.csv", ("xtts_v2",), "new_speakers_amr"),
    "new_tortoise_amr": ("score_ext_amr.csv", ("tortoise",), "new_speakers_amr"),
    "cm04_amr": ("score_cm04_norm_amr.csv", None, "cm04_amr"),
    "cm07_amr": ("score_cm07_amr.csv", None, "cm07_amr"),
    "cm09": ("score_cm09_channel20.csv", None, "cm09"),
}

TOKEN = "${DATA_ROOT}/clips/"


def clip_index(phone_dirs: list[str], amr_dirs: list[str]) -> dict[str, str]:
    """``clips/<name>`` and ``clips/amr/<name>`` -> absolute path, first folder wins."""
    index: dict[str, str] = {}
    for prefix, dirs in (("", phone_dirs), ("amr/", amr_dirs)):
        for d in dirs:
            folder = Path(d)
            if not folder.is_dir():
                continue
            for f in folder.iterdir():
                index.setdefault(prefix + f.name, str(f))
    return index


def locate(frame: pd.DataFrame, index: dict[str, str]) -> pd.DataFrame:
    """A copy whose ``filepath`` is the local file; raises if any clip is not found."""
    keys = frame["filepath"].astype(str).str.replace(TOKEN, "", regex=False)
    missing = sorted(set(keys) - set(index))
    if missing:
        raise FileNotFoundError(f"{len(missing)} clips not found, e.g. {missing[:3]}")
    out = frame.copy()
    out["filepath"] = keys.map(index)
    return out


def select(frame: pd.DataFrame, tools: tuple[str, ...] | None) -> pd.DataFrame:
    """Genuine rows plus the spoof rows of ``tools`` (all spoof rows when None)."""
    if tools is None:
        return frame.reset_index(drop=True)
    genuine = frame["label"].str.lower() == "bonafide"
    return frame[genuine | frame["tool"].isin(tools)].reset_index(drop=True)


#: Optional folder of per-clip feature caches, so a run stopped part-way resumes.
CACHE: Path | None = None


def featurise(frame: pd.DataFrame, index: dict[str, str]):
    """Rows kept (original ``filepath``), features and labels.

    With ``CACHE`` set, each clip's eight statistics are appended to a cache file as they
    are computed and read back on the next run; the numbers are the same either way.
    """
    from src.data import lowlevel_cue as lc

    if CACHE is None:
        x, y, kept = lc.extract(locate(frame, index), None, return_index=True)
        return frame.iloc[kept].reset_index(drop=True), x, y

    from src.utils.audio_utils import load_wav

    CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE / "features.csv"
    done: dict[str, np.ndarray] = {}
    if cache_file.is_file():
        cached = pd.read_csv(cache_file)
        done = {
            k: row for k, row in zip(cached["filepath"], cached.iloc[:, 1:].to_numpy(), strict=True)
        }
    located = locate(frame, index)
    pending = []
    for i, (key, path) in enumerate(zip(frame["filepath"], located["filepath"], strict=True)):
        if key in done:
            continue
        try:
            audio, sr = load_wav(path, target_sr=16_000)
        except Exception as exc:  # noqa: BLE001 - one unreadable clip must not stop the run
            print(f"  [skip] {key}: {type(exc).__name__}: {exc}", flush=True)
            continue
        done[key] = lc.features(audio, sr)
        pending.append((key, done[key]))
        if len(pending) >= 250 or i == len(frame) - 1:
            _append(cache_file, pending)
            pending = []
            print(f"  featurised {i + 1}/{len(frame)}", flush=True)
    _append(cache_file, pending)
    kept = [i for i, key in enumerate(frame["filepath"]) if key in done]
    rows = frame.iloc[kept].reset_index(drop=True)
    x = np.stack([done[k] for k in rows["filepath"]])
    y = rows["label"].astype(str).str.lower().eq("bonafide").astype(np.int64).to_numpy()
    return rows, x, y


def _append(cache_file: Path, items: list) -> None:
    if not items:
        return
    out = pd.DataFrame([v for _, v in items], columns=[f"f{j}" for j in range(len(items[0][1]))])
    out.insert(0, "filepath", [k for k, _ in items])
    out.to_csv(cache_file, mode="a", header=not cache_file.is_file(), index=False)


def compare_one(rows, floor_scores, detector_csv: Path, n_boot: int, seed: int) -> dict:
    detector = pd.read_csv(detector_csv)
    detector = detector[detector["filepath"].isin(rows["filepath"])]
    paired = fc.pair_scores(rows, floor_scores, detector)
    stats = fc.speaker_bootstrap(
        fc.binary_labels(paired["label"]),
        paired["speaker"].to_numpy(),
        paired["detector"].to_numpy(),
        paired["floor"].to_numpy(),
        n_boot=n_boot,
        seed=seed,
    )
    stats.update(clips=int(len(paired)), verdict=fc.verdict(stats["diff_ci"]))
    return stats


def run(
    manifest_dir: str,
    index: dict[str, str],
    detectors: dict[str, list[Path]],
    n_boot: int = 1000,
    seed: int = 0,
) -> list[dict]:
    """``detectors``: name -> score folders; files are ``<run>__<key>_scores.csv``."""
    from src.data import lowlevel_cue as lc

    mdir = Path(manifest_dir)
    train = pd.read_csv(mdir / TRAIN_MANIFEST)
    print(f"featurising {TRAIN_MANIFEST} ({len(train)} clips)", flush=True)
    _, x_train, y_train = featurise(train, index)
    scaler = lc.Standardiser().fit(x_train)
    w, b = lc.fit_logistic(scaler.transform(x_train), y_train)
    results = []
    for test_set, (manifest, tools, key) in TEST_SETS.items():
        frame = select(pd.read_csv(mdir / manifest), tools)
        print(f"featurising {test_set} ({len(frame)} clips)", flush=True)
        rows, x_test, _ = featurise(frame, index)
        floor_scores = lc.predict(scaler.transform(x_test), w, b)
        for name, folders in detectors.items():
            for folder in folders:
                for csv in sorted(Path(folder).glob(f"final_s*__{key}_scores.csv")):
                    stats = compare_one(rows, floor_scores, csv, n_boot, seed)
                    stats.update(detector=name, run=csv.name.split("__")[0], test_set=test_set)
                    results.append(stats)
                    print(
                        f"  {name:10s} {stats['run']:14s} {test_set:17s} "
                        f"det {100 * stats['eer_a']:6.2f} floor {100 * stats['eer_b']:6.2f} "
                        f"diff {100 * stats['diff']:+6.2f} [{100 * stats['diff_ci'][0]:+6.2f}, "
                        f"{100 * stats['diff_ci'][1]:+6.2f}] {stats['verdict']}",
                        flush=True,
                    )
    return results


def summarise(results: list[dict]) -> list[dict]:
    """Per detector and test set: mean EERs over seeds and how many seeds beat the floor."""
    frame = pd.DataFrame(results)
    out = []
    for (det, test_set), g in frame.groupby(["detector", "test_set"], sort=False):
        out.append(
            {
                "detector": det,
                "test_set": test_set,
                "seeds": int(len(g)),
                "detector_eer": float(g["eer_a"].mean()),
                "floor_eer": float(g["eer_b"].mean()),
                "diff": float(g["diff"].mean()),
                "beats_floor": int((g["verdict"] == fc.BEATS_FLOOR).sum()),
                "worse_than_floor": int((g["verdict"] == fc.WORSE_THAN_FLOOR).sum()),
            }
        )
    return out


def markdown(summary: list[dict]) -> str:
    lines = [
        "# Shortcut floors for the final-recipe systems (v1)",
        "",
        "One floor (eight low-level statistics, `src.training.floor_compare`) fitted on",
        f"`{TRAIN_MANIFEST}`, the training data of both systems, and scored on exactly the",
        "clips each detector was scored on. Per seed, the detector-minus-floor difference has a",
        "95% interval from a bootstrap over whole speakers; 'beats floor' counts the seeds whose",
        "whole interval is below zero. EER (%) are means over seeds. Built by",
        "`src/training/floor_final.py`.",
        "",
        "| Detector | Test set | Seeds | Detector EER | Floor EER | Difference | Beats floor | Worse |",
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
            "## Reading",
            "",
            f"The floor is above chance on {', '.join(above)}: fitted on the training",
            "recordings, the eight statistics rank genuine and fake the wrong way round there,",
            "because those recordings differ from the training ones. Beating such a floor is a",
            "weak bar, so these rows show that the detector does not lean on these cues, not",
            "that it beats an informative baseline.",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Shortcut floors for the final-recipe systems")
    parser.add_argument("--manifests", default="data/manifests")
    parser.add_argument(
        "--phone-dirs", nargs="+", required=True, help="G.711 clip folders, in order"
    )
    parser.add_argument("--amr-dirs", nargs="+", required=True, help="AMR-NB clip folders")
    parser.add_argument("--adapter", nargs="+", required=True, help="adapter score folders")
    parser.add_argument("--ssl", nargs="+", required=True, help="SSL-AASIST score folders")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--cache", help="folder for per-clip feature caches (resumable)")
    parser.add_argument("--out", default="experiments/results/floor_final_v1.json")
    parser.add_argument("--doc", default="docs/results/floor_final_v1.md")
    args = parser.parse_args()

    global CACHE
    CACHE = Path(args.cache) if args.cache else None
    index = clip_index(args.phone_dirs, args.amr_dirs)
    print(f"{len(index)} clips indexed", flush=True)
    detectors = {"adapter": args.adapter, "SSL-AASIST": args.ssl}
    results = run(args.manifests, index, detectors, n_boot=args.n_boot)
    summary = summarise(results)
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
