"""Measure the voice check on audio it has not been measured on yet.

The paper measured the detectors on telephone audio (simulated G.711, real AMR-NB, 30
real calls). A voice check that accepts any file also meets wideband audio: a studio clone
downloaded as wav, speech recorded straight into a laptop microphone. This module scores
such audio through exactly the pipeline the demo uses (``src.inference.any_audio.judge``)
and reports how often each kind is judged correctly.

Three steps, each resumable::

    # 1. a manifest of the CLEAN (wideband) copies of the dev and test clips
    python -m src.inference.any_audio_eval build --out experiments/any_audio/clean.csv

    # 2. score it with the front end on and off (rows already scored are skipped)
    python -m src.inference.any_audio_eval score experiments/any_audio/clean.csv \
        --out experiments/any_audio/clean_scored.csv --front-end g711 none

    # 3. the accuracy table; the front end is chosen on the DEV rows only
    python -m src.inference.any_audio_eval report experiments/any_audio/clean_scored.csv \
        --out experiments/results/any_audio_clean.md

Labelled recordings collected with the demo (``data/field_validation/log.csv``) are a
manifest too, so the same ``score`` and ``report`` steps measure them.

**No choice is made on test rows.** The threshold is each checkpoint's frozen operating
point; the only decision this module makes, front end on or off, is taken on the 506
development clips (two speakers, genuine and XTTS-v2) by balanced accuracy, and only then
are the test rows reported.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

#: The calibration manifests (``${DATA_ROOT}/clips/<name>``): 506 dev clips, 300 clips of
#: each test set and 1,000 HiACC clips, as scored in the live-path validation.
MANIFEST_DIR = "D:/kaggle_realcall/ssl_manifests"
SETS = ("dev", "eval_pool", "new_speakers", "cm04", "cm07", "cm09", "rvc_test", "hiacc")
#: Folders holding the clean (pre-channel) copies, searched by file name.
CLEAN_ROOTS = (
    "D:/dfdata/ext_bundle/clean/clips",
    "D:/dfdata/lora_bundle_norm",
    "D:/dfdata/hiacc_norm/clips",
)
KEY = ["filepath", "front_end"]


def _index(roots) -> dict[str, str]:
    found: dict[str, str] = {}
    for root in roots:
        for folder, _, files in os.walk(root):
            for name in files:
                if name.endswith(".wav"):
                    found.setdefault(name, os.path.join(folder, name).replace("\\", "/"))
    return found


def build(manifest_dir: str = MANIFEST_DIR, roots=CLEAN_ROOTS) -> pd.DataFrame:
    """Every calibration clip, pointed at its clean copy; a clip with none is dropped."""
    index = _index(roots)
    frames = []
    for name in SETS:
        df = pd.read_csv(Path(manifest_dir) / f"{name}.csv")
        clean = df["filepath"].str.split("/").str[-1].map(index)
        missing = int(clean.isna().sum())
        if missing:
            print(f"{name}: {missing} clip(s) have no clean copy and are left out")
        frames.append(
            pd.DataFrame(
                {
                    "filepath": clean,
                    "label": df["label"].str.lower(),
                    "tool": df["tool"],
                    "set": name,
                    "split": "dev" if name == "dev" else "test",
                    "condition": "clean",
                }
            ).dropna(subset=["filepath"])
        )
    return pd.concat(frames, ignore_index=True)


def score(manifest: pd.DataFrame, out: Path, front_ends, config_file: str) -> pd.DataFrame:
    """Judge every row with every front end; append to ``out`` as it goes (resumable)."""
    from src.inference import any_audio as aa

    config = aa.DetectorConfig.from_file(config_file)
    threshold = config.threshold()
    done = pd.read_csv(out) if out.is_file() else pd.DataFrame(columns=KEY)
    seen = set(zip(done["filepath"], done["front_end"], strict=True))
    todo = [
        (r, f) for f in front_ends for r in manifest.itertuples() if (r.filepath, f) not in seen
    ]
    print(f"{len(seen)} already scored, {len(todo)} to go", flush=True)
    if not todo:
        return done
    score_fn = aa.load_detector(config)
    out.parent.mkdir(parents=True, exist_ok=True)
    batch = []
    for i, (row, front) in enumerate(todo, 1):
        try:
            verdict = aa.judge(
                aa.load_audio(row.filepath), score_fn, threshold, front, row.filepath
            )
            result = verdict.summary()
            result["window_scores"] = " ".join(f"{p:.4f}" for p in verdict.scored)
        except Exception as exc:  # a broken file is recorded, not fatal
            result = {"source": row.filepath, "front_end": front, "verdict": f"error: {exc}"}
        batch.append(
            {
                **{k: getattr(row, k) for k in manifest.columns},
                "front_end": front,
                "checkpoint": config.checkpoint,
                **{k: v for k, v in result.items() if k not in ("source", "front_end")},
            }
        )
        if len(batch) == 50 or i == len(todo):
            pd.DataFrame(batch).to_csv(out, mode="a", header=not out.is_file(), index=False)
            batch = []
            print(f"  {i}/{len(todo)}", flush=True)
    return pd.read_csv(out)


def _correct(df: pd.DataFrame) -> pd.Series:
    want = np.where(df["label"].str.lower() == "bonafide", "real", "cloned")
    return pd.Series(df["verdict"].to_numpy() == want, index=df.index)


def _eer(df: pd.DataFrame) -> float | None:
    """Threshold-free ranking quality of one set: genuine against every spoof in it."""
    sub = df.dropna(subset=["p_genuine"])
    y = (sub["label"] == "bonafide").to_numpy()
    if y.all() or not y.any():
        return None
    from src.training.metrics import compute_eer

    eer, _ = compute_eer(sub["p_genuine"].to_numpy(float), y.astype(int))
    return 100.0 * float(eer)


def balanced_accuracy(df: pd.DataFrame) -> float:
    ok = _correct(df)
    genuine = df["label"] == "bonafide"
    parts = [ok[genuine].mean(), ok[~genuine].mean()]
    return float(np.nanmean(parts))


def choose_front_end(scored: pd.DataFrame) -> tuple[str, dict]:
    """The front end with the higher balanced accuracy on the DEV rows (ties: g711)."""
    dev = scored[(scored["split"] == "dev") & ~scored["verdict"].str.startswith("error")]
    by = {f: balanced_accuracy(g) for f, g in dev.groupby("front_end")}
    best = max(sorted(by, key=lambda f: f != "g711"), key=lambda f: by[f])
    return best, by


def report(scored: pd.DataFrame) -> str:
    scored = scored[~scored["verdict"].astype(str).str.startswith("error")].copy()
    scored["kind"] = np.where(scored["label"] == "bonafide", "genuine", scored["tool"])
    best, dev = choose_front_end(scored) if (scored["split"] == "dev").any() else (None, {})
    lines = ["# Voice check on audio not measured before", ""]
    if dev:
        lines += [
            "Front end chosen on the development clips only (balanced accuracy): "
            + ", ".join(f"{f} {100 * v:.1f} %" for f, v in sorted(dev.items()))
            + f" -> **{best}**.",
            "",
        ]
    fronts = sorted(scored["front_end"].unique())
    lines += [
        "% judged correctly at the frozen threshold (number of recordings in brackets).",
        "",
        "| Set | Kind | " + " | ".join(fronts) + " |",
        "|---|---|" + "---|" * len(fronts),
    ]
    for (split, name, kind), _ in scored.groupby(["split", "set", "kind"], sort=False):
        cells = []
        for f in fronts:
            g = scored[
                (scored["set"] == name) & (scored["kind"] == kind) & (scored["front_end"] == f)
            ]
            cells.append(f"{100 * _correct(g).mean():.1f} ({len(g)})" if len(g) else "–")
        tag = " (dev)" if split == "dev" else ""
        lines.append(f"| {name}{tag} | {kind} | " + " | ".join(cells) + " |")
    lines += ["", "Equal error rate per set (threshold-free), %:", ""]
    lines += ["| Set | " + " | ".join(fronts) + " |", "|---|" + "---|" * len(fronts)]
    for name in scored["set"].unique():
        cells = []
        for f in fronts:
            e = _eer(scored[(scored["set"] == name) & (scored["front_end"] == f)])
            cells.append("–" if e is None else f"{e:.2f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="manifest of the clean copies of the dev and test clips")
    b.add_argument("--manifest-dir", default=MANIFEST_DIR)
    b.add_argument("--out", required=True)
    s = sub.add_parser("score", help="judge every row of a manifest (resumable)")
    s.add_argument("manifest")
    s.add_argument("--out", required=True)
    s.add_argument("--front-end", nargs="+", default=["g711", "none"])
    s.add_argument("--config", default="configs/voice_check.yaml")
    s.add_argument("--limit", type=int, default=None, help="first N rows only (smoke test)")
    r = sub.add_parser("report", help="accuracy table of a scored CSV")
    r.add_argument("scored")
    r.add_argument("--out")
    args = parser.parse_args(argv)

    if args.command == "build":
        df = build(args.manifest_dir)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(df.groupby(["set", "label", "tool"]).size().to_string())
    elif args.command == "score":
        manifest = pd.read_csv(args.manifest)
        if args.limit:
            manifest = manifest.groupby("set", sort=False).head(args.limit)
        score(manifest, Path(args.out), args.front_end, args.config)
    else:
        text = report(pd.read_csv(args.scored))
        print(text)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
