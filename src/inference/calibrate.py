"""Set the live demo's operating threshold from streaming-path window scores (W8-T5, P-029).

Every reported EER scores one deterministic 4 s crop per clip. The live call scores
every 4 s window, level-normalised and smoothed, and the verdict engine acts on the
smoothed value. P-029 showed a threshold copied from a batch results file calls
every real clip fake. So the threshold is chosen on the thing the demo actually
computes:

1. **Score** each clip of a manifest through ``StreamingScorer``, the exact live
   path, and keep every window: raw score, smoothed score, level.
2. **Choose** the operating point on the *dev* manifest only; the eval sets never
   touch this choice. The window-level equal-error point is reported but **not
   used**: on a minute-long call, an 8% chance per window of a real window scoring
   low makes two low windows in a row likely, and genuine callers get warned. A
   false alarm on a real caller is the expensive error for a phone warning, so the
   threshold and the ladder length are swept over candidates and the chosen point
   is the most sensitive one whose **false-alarm rate on stitched 60 s dev calls of
   real speech is at most** ``MAX_REAL_CALL_ALERT_PCT``.
3. **Validate** on held-out sets by replaying clips through ``VerdictEngine`` with
   that threshold, and report what a receiver would see: how many real calls raise
   any alert (false alarm) and how many fake calls reach ``SUSPICIOUS`` or
   ``LIKELY_FAKE`` (detection).

Eval clips last 7-15 s, about two windows each. The verdict ladder needs two windows
of warm-up and then two fake-leaning windows in a row, so a single clip mostly
measures its own length. Real calls last a minute or more. Validation therefore also
**stitches clips of the same label into 30 s and 60 s calls**: windows are laid end to
end in a fixed shuffled order and the moving average is recomputed across the whole
call from the raw window scores, as the live path would compute it.

Scores are resumable per clip, since a CPU pass over a thousand clips takes an hour.

Run::

    python -m src.inference.calibrate score --checkpoint best.pt \\
        --manifest data/manifests/codemix_adapt_dev_channel20.csv --data-root ROOT \\
        --out experiments/calibration/dev_windows.csv
    python -m src.inference.calibrate choose --dev experiments/calibration/dev_windows.csv \\
        --validate experiments/calibration/eval_pool_windows.csv ... \\
        --out configs/threshold.yaml --report experiments/results/threshold_calibration.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from live_call.verdict_engine import EngineConfig, State, VerdictEngine
from src.inference.streaming import SAMPLE_RATE, StreamingScorer

#: Highest tolerated share of genuine 60 s dev calls that raise any alert.
MAX_REAL_CALL_ALERT_PCT = 5.0
#: Candidate thresholds: these quantiles of real dev windows' smoothed scores.
REAL_WINDOW_QUANTILES = (0.002, 0.005, 0.01, 0.02, 0.05)
LADDER_LENGTHS = ((2, 4), (3, 5))  # (suspicious_after, fake_after)

WINDOW_COLUMNS = [
    "filepath",
    "label",
    "tool",
    "window",
    "end_seconds",
    "level_dbfs",
    "score",
    "smoothed",
]


def stratified_subset(frame: pd.DataFrame, n: int | None, seed: int = 0) -> pd.DataFrame:
    """A deterministic label-stratified subset of about ``n`` clips (all when None)."""
    if n is None or n >= len(frame):
        return frame.reset_index(drop=True)
    parts = [
        group.sample(n=max(1, round(n * len(group) / len(frame))), random_state=seed)
        for _, group in frame.groupby(frame["label"].str.lower())
    ]
    return pd.concat(parts).sort_index().reset_index(drop=True)


def window_rows(filepath: str, label: str, tool: str, audio: np.ndarray, score_fn) -> list[dict]:
    """Every window of one clip through the live path. Short clips are padded to one window."""
    scorer = StreamingScorer(score_fn)
    if 0 < audio.size < scorer.windower.window:
        audio = np.pad(audio, (0, scorer.windower.window - audio.size))
    rows = []
    for i, r in enumerate(scorer.push(audio.astype(np.float32, copy=False))):
        rows.append(
            {
                "filepath": filepath,
                "label": label,
                "tool": tool,
                "window": i,
                "end_seconds": r.end_seconds,
                "level_dbfs": round(r.level_dbfs, 2),
                "score": r.score,
                "smoothed": r.smoothed if not r.skipped else None,
            }
        )
    return rows


def score_manifest(
    manifest: str, data_root: str, out: str, score_fn, limit: int | None = None
) -> pd.DataFrame:
    """Score a manifest window by window, appending to ``out`` so a restart resumes."""
    from src.utils.audio_utils import load_wav
    from src.utils.paths import resolve

    frame = stratified_subset(pd.read_csv(manifest), limit)
    out_path = Path(out)
    done: set[str] = set()
    if out_path.is_file():
        done = set(pd.read_csv(out_path)["filepath"].astype(str))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    todo = frame[~frame["filepath"].astype(str).isin(done)]
    print(
        f"{manifest}: {len(frame)} clips, {len(done)} already scored, {len(todo)} to go", flush=True
    )
    for n, row in enumerate(todo.itertuples(index=False), 1):
        audio, _ = load_wav(resolve(row.filepath, data_root), target_sr=SAMPLE_RATE)
        rows = window_rows(row.filepath, row.label, row.tool, audio, score_fn)
        pd.DataFrame(rows, columns=WINDOW_COLUMNS).to_csv(
            out_path, mode="a", header=not out_path.is_file(), index=False
        )
        if n % 25 == 0:
            print(f"  {n}/{len(todo)} clips", flush=True)
    return pd.read_csv(out_path)


def eer_threshold(real: np.ndarray, fake: np.ndarray) -> tuple[float, float]:
    """(eer, threshold) with bonafide accepted when score >= threshold."""
    from src.training.metrics import compute_eer

    scores = np.concatenate([real, fake])
    labels = np.concatenate([np.ones(real.size), np.zeros(fake.size)])
    return compute_eer(scores, labels)


def choose_threshold(dev: pd.DataFrame) -> dict:
    """EER point of smoothed window scores on the dev windows."""
    scored = dev.dropna(subset=["smoothed"])
    real = scored.loc[scored["label"] == "bonafide", "smoothed"].to_numpy(float)
    fake = scored.loc[scored["label"] == "spoof", "smoothed"].to_numpy(float)
    if real.size == 0 or fake.size == 0:
        raise ValueError("dev windows need both real and fake clips")
    eer, threshold = eer_threshold(real, fake)
    return {
        "threshold": float(threshold),
        "dev_window_eer": round(100 * float(eer), 2),
        "dev_real_windows": int(real.size),
        "dev_fake_windows": int(fake.size),
        "dev_clips": int(scored["filepath"].nunique()),
    }


def replay_calls(windows: pd.DataFrame, config: EngineConfig) -> pd.DataFrame:
    """Run every clip's windows through the verdict engine; one row per clip."""
    out = []
    for filepath, group in windows.sort_values(["filepath", "window"]).groupby(
        "filepath", sort=False
    ):
        engine = VerdictEngine(config)
        kinds = []
        for row in group.itertuples(index=False):
            smoothed = None if pd.isna(row.smoothed) else float(row.smoothed)
            kinds += [e.kind.value for e in engine.update(float(row.end_seconds), smoothed)]
        engine.end_call()
        out.append(
            {
                "filepath": filepath,
                "label": group["label"].iloc[0],
                "tool": group["tool"].iloc[0],
                "windows": len(group),
                "peak_state": engine.summary.peak_state.value,
                "alerted": engine.summary.peak_state in (State.SUSPICIOUS, State.LIKELY_FAKE),
                "likely_fake": engine.summary.peak_state == State.LIKELY_FAKE,
            }
        )
    return pd.DataFrame(out)


def stitch_calls(
    windows: pd.DataFrame, seconds: float, hop: float = 2.0, alpha: float = 0.5, seed: int = 0
) -> pd.DataFrame:
    """Join clips of one label into calls of about ``seconds``, re-smoothing across each call."""
    per_call = max(1, int(round(seconds / hop)) - 1)  # windows in a call of that length
    rows = []
    for label, group in windows.groupby("label"):
        clips = list(group.groupby("filepath", sort=True))
        order = np.random.default_rng(seed).permutation(len(clips))
        stream = pd.concat([clips[i][1].sort_values("window") for i in order], ignore_index=True)
        for c, start in enumerate(range(0, len(stream) - per_call + 1, per_call)):
            chunk = stream.iloc[start : start + per_call]
            ema = None
            for w, row in enumerate(chunk.itertuples(index=False)):
                if not pd.isna(row.score):
                    ema = (
                        float(row.score)
                        if ema is None
                        else alpha * float(row.score) + (1 - alpha) * ema
                    )
                rows.append(
                    {
                        "filepath": f"{label}_call_{c:04d}",
                        "label": label,
                        "tool": row.tool,
                        "window": w,
                        "end_seconds": 4.0 + hop * w,
                        "smoothed": None if pd.isna(row.score) else ema,
                    }
                )
    return pd.DataFrame(rows)


def call_metrics(calls: pd.DataFrame) -> dict:
    """What a receiver sees: false alarms on real calls, detections on fake ones."""
    real, fake = calls[calls["label"] == "bonafide"], calls[calls["label"] == "spoof"]

    def pct(x) -> float | None:
        return None if len(x) == 0 else round(100 * float(x.mean()), 2)

    return {
        "real_calls": int(len(real)),
        "fake_calls": int(len(fake)),
        "real_calls_alerted_pct": pct(real["alerted"]),
        "real_calls_likely_fake_pct": pct(real["likely_fake"]),
        "fake_calls_alerted_pct": pct(fake["alerted"]),
        "fake_calls_likely_fake_pct": pct(fake["likely_fake"]),
        "mean_windows_per_call": round(float(calls["windows"].mean()), 2) if len(calls) else None,
    }


def sweep(dev: pd.DataFrame) -> tuple[dict, list[dict]]:
    """Every candidate operating point on dev, and the one chosen by the false-alarm rule."""
    scored = dev.dropna(subset=["smoothed"])
    real = scored.loc[scored["label"] == "bonafide", "smoothed"].to_numpy(float)
    eer_point = choose_threshold(dev)["threshold"]
    candidates = sorted({float(np.quantile(real, q)) for q in REAL_WINDOW_QUANTILES} | {eer_point})
    stitched = stitch_calls(dev, 60)
    rows = []
    for threshold in candidates:
        for suspicious_after, fake_after in LADDER_LENGTHS:
            config = EngineConfig(
                threshold=threshold, suspicious_after=suspicious_after, fake_after=fake_after
            )
            calls60 = call_metrics(replay_calls(stitched, config))
            clips = call_metrics(replay_calls(dev, config))
            rows.append(
                {
                    "threshold": threshold,
                    "suspicious_after": suspicious_after,
                    "fake_after": fake_after,
                    "real_windows_below_pct": round(100 * float((real < threshold).mean()), 2),
                    "dev_calls_60s": calls60,
                    "dev_per_clip": clips,
                }
            )
    allowed = [
        r for r in rows if r["dev_calls_60s"]["real_calls_alerted_pct"] <= MAX_REAL_CALL_ALERT_PCT
    ]
    pool = allowed or rows
    best = max(
        pool,
        key=lambda r: (
            r["dev_calls_60s"]["fake_calls_alerted_pct"],
            r["dev_per_clip"]["fake_calls_alerted_pct"],
            -r["dev_calls_60s"]["real_calls_alerted_pct"],
        ),
    )
    return {**best, "rule_satisfied": bool(allowed), "eer_threshold": eer_point}, rows


def main() -> None:
    """CLI: ``score`` a manifest, or ``choose`` and validate the threshold."""
    import argparse

    import yaml

    parser = argparse.ArgumentParser(description="Demo threshold from streaming-path scores")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("score")
    sc.add_argument("--checkpoint", required=True)
    sc.add_argument("--manifest", required=True)
    sc.add_argument("--data-root", required=True)
    sc.add_argument("--out", required=True)
    sc.add_argument("--limit", type=int, default=None, help="label-stratified clip subset")
    sc.add_argument("--device", default="cpu")
    ch = sub.add_parser("choose")
    ch.add_argument("--dev", required=True)
    ch.add_argument("--validate", nargs="*", default=[])
    ch.add_argument("--checkpoint-name", required=True)
    ch.add_argument("--out", default="configs/threshold.yaml")
    ch.add_argument("--report", default="experiments/results/threshold_calibration.json")
    args = parser.parse_args()

    if args.cmd == "score":
        from src.inference.predict import load_detector, make_score_fn

        model, device = load_detector(args.checkpoint, device=args.device)
        score_manifest(
            args.manifest, args.data_root, args.out, make_score_fn(model, device), args.limit
        )
        return

    dev = pd.read_csv(args.dev)
    window_eer = choose_threshold(dev)
    chosen, candidates = sweep(dev)
    config = EngineConfig(
        threshold=chosen["threshold"],
        suspicious_after=chosen["suspicious_after"],
        fake_after=chosen["fake_after"],
    )
    report = {
        "checkpoint": args.checkpoint_name,
        "chosen_on": args.dev,
        "rule": f"most sensitive candidate with <= {MAX_REAL_CALL_ALERT_PCT}% of genuine 60 s dev calls alerted",
        "rule_satisfied": chosen["rule_satisfied"],
        "threshold": chosen["threshold"],
        "dev_window_eer_point": window_eer,
        "engine": {k: getattr(config, k) for k in EngineConfig.__dataclass_fields__},
        "dev_at_chosen": {"calls_60s": chosen["dev_calls_60s"], "per_clip": chosen["dev_per_clip"]},
        "candidates": candidates,
        "validation": {},
    }
    for path in args.validate:
        windows = pd.read_csv(path)
        report["validation"][Path(path).stem] = {
            "per_clip": call_metrics(replay_calls(windows, config)),
            **{
                f"calls_{int(sec)}s": call_metrics(replay_calls(stitch_calls(windows, sec), config))
                for sec in (30, 60)
            },
        }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    Path(args.out).write_text(
        "# Live-demo operating point (W8-T5, P-029). Chosen by src.inference.calibrate on\n"
        "# streaming-path smoothed window scores of the dev manifest only, for a low false-alarm\n"
        f"# rate on genuine 60 s calls; validated on held-out sets in {args.report}.\n"
        + yaml.safe_dump(
            {
                "threshold": config.threshold,
                "checkpoint": args.checkpoint_name,
                "margin": config.margin,
                "min_windows": config.min_windows,
                "suspicious_after": config.suspicious_after,
                "fake_after": config.fake_after,
                "recover_after": config.recover_after,
            },
            sort_keys=False,
        )
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
