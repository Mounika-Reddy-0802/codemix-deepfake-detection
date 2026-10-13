"""False alarms on genuine speech from new speakers and a new speaking style.

External review comment 3 asks for evaluation on additional unseen speakers and
speaking styles. Every code-mixed test set so far is MUCS lecture audio: read,
prepared speech. HiACC's adult partition is different on both counts -- 24 speakers
the detector never met, talking spontaneously -- but it holds genuine speech only,
so it cannot give an EER. What it can give is the number a deployment cares about
first on new callers: how often the detector calls a genuine speaker fake.

This module scores a genuine-only manifest and reports that false-alarm rate at the
deployed operating point, with a bootstrap over speakers (a speaker's clips are not
independent, so resampling clips would understate the uncertainty), the rate per
speaker, and the difference from a reference set of genuine clips -- MUCS read
speech, scored by the same model -- so the effect of the speaking style is measured
against something rather than in isolation.

Scores are P(genuine): a clip is called fake when its score is below the threshold.
HiACC is evaluation-only (licence and dataset rules); its child partition is never
read by anything in this project.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: The deployed operating point: P(genuine) below this is reported as a clone.
DEPLOYED_THRESHOLD = 0.5


def _genuine(scores: pd.DataFrame) -> pd.DataFrame:
    """Only the genuine rows, with the columns this module needs."""
    missing = {"score", "speaker", "label"} - set(scores.columns)
    if missing:
        raise ValueError(f"scores are missing columns: {sorted(missing)}")
    genuine = scores[scores["label"].astype(str).str.lower() == "bonafide"]
    if genuine.empty:
        raise ValueError("no genuine clips to measure false alarms on")
    return genuine


def false_alarm_rate(scores: np.ndarray, threshold: float = DEPLOYED_THRESHOLD) -> float:
    """Share of genuine clips whose P(genuine) falls below ``threshold``."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        raise ValueError("no scores")
    return float(np.mean(scores < threshold))


def _speaker_resamples(
    genuine: pd.DataFrame, threshold: float, n_boot: int, seed: int
) -> np.ndarray:
    """False-alarm rate for each bootstrap resample of whole speakers."""
    flags = genuine["score"].to_numpy(dtype=np.float64) < threshold
    speakers = genuine["speaker"].astype(str).to_numpy()
    unique = np.unique(speakers)
    members = [np.flatnonzero(speakers == s) for s in unique]
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        drawn = rng.integers(0, len(unique), len(unique))
        idx = np.concatenate([members[i] for i in drawn])
        out[b] = flags[idx].mean()
    return out


def summarise(
    scores: pd.DataFrame,
    threshold: float = DEPLOYED_THRESHOLD,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """False-alarm rate with a 95% speaker-bootstrap interval, and the per-speaker spread."""
    genuine = _genuine(scores)
    boot = _speaker_resamples(genuine, threshold, n_boot, seed)
    per = per_speaker(genuine, threshold)
    return {
        "threshold": threshold,
        "clips": int(len(genuine)),
        "speakers": int(genuine["speaker"].nunique()),
        "false_alarm_rate": false_alarm_rate(genuine["score"], threshold),
        "false_alarm_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "median_score": float(genuine["score"].median()),
        "per_speaker_min": float(per["false_alarm_rate"].min()),
        "per_speaker_median": float(per["false_alarm_rate"].median()),
        "per_speaker_max": float(per["false_alarm_rate"].max()),
        "speakers_above_10pct": int((per["false_alarm_rate"] > 0.10).sum()),
    }


def per_speaker(scores: pd.DataFrame, threshold: float = DEPLOYED_THRESHOLD) -> pd.DataFrame:
    """Clips and false-alarm rate for each speaker, highest rate first."""
    genuine = _genuine(scores)
    flagged = genuine.assign(flag=genuine["score"] < threshold)
    table = flagged.groupby(flagged["speaker"].astype(str)).agg(
        clips=("flag", "size"), false_alarms=("flag", "sum")
    )
    table["false_alarm_rate"] = table["false_alarms"] / table["clips"]
    return table.sort_values("false_alarm_rate", ascending=False).reset_index()


def compare(
    target: pd.DataFrame,
    reference: pd.DataFrame,
    threshold: float = DEPLOYED_THRESHOLD,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Target minus reference false-alarm rate, with a 95% interval.

    The two sets hold different speakers, so they are resampled independently (each
    by speaker) rather than paired.
    """
    a = _speaker_resamples(_genuine(target), threshold, n_boot, seed)
    b = _speaker_resamples(_genuine(reference), threshold, n_boot, seed + 1)
    diff = a - b
    point = false_alarm_rate(_genuine(target)["score"], threshold) - false_alarm_rate(
        _genuine(reference)["score"], threshold
    )
    return {
        "difference": point,
        "difference_ci": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
    }


def main() -> None:
    """CLI: score a genuine-only manifest and report its false alarms against a reference."""
    import argparse
    import json
    from pathlib import Path

    from src.training.evaluate import score_manifest

    parser = argparse.ArgumentParser(description="False alarms on genuine speech")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True, help="genuine-only manifest to score")
    parser.add_argument("--data-root", default=None)
    parser.add_argument(
        "--reference-scores",
        default=None,
        help="per-clip scores CSV of a reference set (its genuine rows are used)",
    )
    parser.add_argument("--encoder", default="wav2vec2-base")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=4.0)
    parser.add_argument("--threshold", type=float, default=DEPLOYED_THRESHOLD)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--scores-out", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    scored = score_manifest(
        args.checkpoint,
        args.manifest,
        encoder=args.encoder,
        batch_size=args.batch_size,
        max_seconds=args.max_seconds,
        device=args.device,
        data_root=args.data_root,
        num_workers=args.num_workers,
        partial_path=str(Path(args.scores_out).with_suffix(".partial.csv")),
    )
    Path(args.scores_out).parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.scores_out, index=False)

    result = {
        "checkpoint": args.checkpoint,
        "manifest": args.manifest,
        "summary": summarise(scored, args.threshold, args.n_boot),
    }
    if args.reference_scores:
        reference = pd.read_csv(args.reference_scores)
        result["reference"] = {
            "scores": args.reference_scores,
            "summary": summarise(reference, args.threshold, args.n_boot),
        }
        result["difference"] = compare(scored, reference, args.threshold, args.n_boot)
    result["per_speaker"] = per_speaker(scored, args.threshold).to_dict(orient="records")
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")

    s = result["summary"]
    print(
        f"\n{s['clips']} genuine clips, {s['speakers']} speakers: false alarms "
        f"{100 * s['false_alarm_rate']:.2f}% [{100 * s['false_alarm_ci'][0]:.2f}, "
        f"{100 * s['false_alarm_ci'][1]:.2f}] at P(genuine) < {args.threshold}"
    )
    if "reference" in result:
        r, d = result["reference"]["summary"], result["difference"]
        print(
            f"reference: {100 * r['false_alarm_rate']:.2f}% on {r['clips']} clips; difference "
            f"{100 * d['difference']:+.2f} points [{100 * d['difference_ci'][0]:+.2f}, "
            f"{100 * d['difference_ci'][1]:+.2f}]"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
