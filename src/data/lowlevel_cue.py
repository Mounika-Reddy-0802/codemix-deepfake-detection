"""W5-T4 — the low-level-cue shortcut check (AffectDF Appendix G replica).

The question this answers. Every EER this project reports is measured on spoofs it
generated itself, against bonafide drawn from a different corpus. That is exactly
the setup in which a detector can score beautifully without hearing anything: if the
XTTS clips are quieter, or less clipped, or sit at a different DC offset than the
MUCS recordings, then *loudness* separates the classes and the model only has to
learn loudness. The result would look like deepfake detection and be a recording
artifact.

AffectDF checks this with a logistic regression over eight cheap signal statistics
(their Appendix G). On a sound corpus it lands near chance -- theirs reaches 53.16%
EER. The plan (W5-T4) adopts the check as a **gate**: "must be near chance, else fix
the pipeline before trusting anything."

Read the number the right way round. A near-chance EER here does not prove the
detector is listening to anything meaningful; it only removes one specific, cheap
explanation for its success. A *low* EER here is the informative outcome, and it is
bad news -- it means these eight numbers alone separate the classes, and every model
result on this data is suspect until the imbalance is fixed.

Hand-rolled in numpy rather than sklearn, for the reason the metrics and the channel
simulation are (P-004/P-005/P-006): CI installs only ruff/pytest/numpy/pandas/pyyaml,
and a number that goes in the paper is unit-tested here rather than trusted to a
dependency that CI never exercises.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: The eight statistics, in the order the coefficient table reports them.
FEATURE_NAMES: tuple[str, ...] = (
    "rms_mean",
    "rms_std",
    "peak_amplitude",
    "clipping_ratio",
    "dc_offset",
    "zero_crossing_rate",
    "spectral_rolloff_hz",
    "hf_energy_ratio",
)

#: |x| at or above this counts as a clipped sample.
CLIP_THRESHOLD = 0.99
#: Rolloff point: the frequency below which this share of energy sits.
ROLLOFF_FRACTION = 0.85
#: Narrowband ceiling, the same 4 kHz the channel simulation band-limits to.
HF_CUTOFF_HZ = 4_000.0


def features(signal: np.ndarray, sr: int, frame: int = 1_024) -> np.ndarray:
    """The eight low-level statistics for one clip, in :data:`FEATURE_NAMES` order."""
    x = np.asarray(signal, dtype=np.float64).ravel()
    if x.size == 0:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float64)

    # Framed RMS: the mean says how loud, the std how much the level moves. A
    # constant-level generator and a live lecture differ on the second even when
    # normalisation has equalised the first.
    n_frames = max(1, x.size // frame)
    trimmed = x[: n_frames * frame].reshape(n_frames, frame)
    frame_rms = np.sqrt(np.mean(trimmed**2, axis=1))

    spectrum = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(x.size, d=1.0 / sr)
    total = float(spectrum.sum())
    if total > 0.0:
        cumulative = np.cumsum(spectrum) / total
        rolloff = float(freqs[int(np.searchsorted(cumulative, ROLLOFF_FRACTION))])
        hf_ratio = float(spectrum[freqs >= HF_CUTOFF_HZ].sum() / total)
    else:
        rolloff = 0.0
        hf_ratio = 0.0

    return np.array(
        [
            float(np.mean(frame_rms)),
            float(np.std(frame_rms)),
            float(np.max(np.abs(x))),
            float(np.mean(np.abs(x) >= CLIP_THRESHOLD)),
            float(np.mean(x)),
            float(np.mean(np.abs(np.diff(np.signbit(x))))),
            rolloff,
            hf_ratio,
        ],
        dtype=np.float64,
    )


def extract(manifest: pd.DataFrame, data_root: str | None = None, target_sr: int = 16_000):
    """Feature matrix ``[N, 8]`` and label vector ``[N]`` (1 = bonafide) for a manifest."""
    from src.utils.audio_utils import load_wav
    from src.utils.paths import resolve as resolve_path

    rows: list[np.ndarray] = []
    labels: list[int] = []
    for i, row in enumerate(manifest.to_dict(orient="records")):
        try:
            audio, sr = load_wav(str(resolve_path(row["filepath"], data_root)), target_sr=target_sr)
        except Exception as exc:  # noqa: BLE001 - one unreadable clip must not stop the check
            print(f"  [skip] {row.get('filepath')}: {type(exc).__name__}: {exc}")
            continue
        rows.append(features(audio, sr))
        labels.append(1 if str(row["label"]).lower() == "bonafide" else 0)
        if (i + 1) % 500 == 0:
            print(f"  featurised {i + 1}/{len(manifest)}", flush=True)
    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.int64)


class Standardiser:
    """Zero-mean unit-variance scaling, fitted on train and reused on test.

    Fitting on the test set would leak its distribution into the check and is
    exactly the sort of shortcut this module exists to detect.
    """

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> Standardiser:
        self.mean = x.mean(axis=0)
        spread = x.std(axis=0)
        # A constant column has no information; dividing by its zero spread would
        # produce NaNs that silently poison every downstream coefficient.
        self.scale = np.where(spread < 1e-12, 1.0, spread)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise RuntimeError("Standardiser.transform called before fit")
        return (x - self.mean) / self.scale


def fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    epochs: int = 2_000,
    l2: float = 1e-4,
    class_weighted: bool = True,
) -> tuple[np.ndarray, float]:
    """Full-batch gradient descent on the binary cross-entropy. Returns ``(w, b)``.

    ``class_weighted`` matters for the same reason it does in training: our manifests
    run about 60% spoof, and an unweighted fit can reach a respectable accuracy by
    leaning on the prior instead of the features -- which would understate how
    separable the classes actually are.
    """
    n, d = x.shape
    w = np.zeros(d, dtype=np.float64)
    b = 0.0
    if class_weighted:
        n_pos = float(max(int(y.sum()), 1))
        n_neg = float(max(n - int(y.sum()), 1))
        sample_w = np.where(y == 1, n / (2.0 * n_pos), n / (2.0 * n_neg))
    else:
        sample_w = np.ones(n, dtype=np.float64)
    total_w = float(sample_w.sum())

    for _ in range(epochs):
        p = _sigmoid(x @ w + b)
        residual = (p - y) * sample_w
        w -= lr * ((x.T @ residual) / total_w + l2 * w)
        b -= lr * float(residual.sum() / total_w)
    return w, b


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Overflow-safe logistic function."""
    out = np.empty_like(z, dtype=np.float64)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def predict(x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """P(bonafide) per row, matching the sign convention of the model's own scores."""
    return _sigmoid(x @ w + b)


def run_check(
    train_manifest: pd.DataFrame,
    test_manifest: pd.DataFrame,
    train_root: str | None = None,
    test_root: str | None = None,
    seed: int = 1234,
) -> dict:
    """Fit on ``train_manifest``, score ``test_manifest``, return the verdict dict.

    Fitting and scoring on speaker-disjoint manifests is deliberate: it asks whether
    low-level cues generalise the way the plan's protocol claims the real result does.
    """
    from src.training.metrics import evaluate as eval_metrics

    print(f"featurising train ({len(train_manifest)} clips) ...", flush=True)
    x_train, y_train = extract(train_manifest, train_root)
    print(f"featurising test ({len(test_manifest)} clips) ...", flush=True)
    x_test, y_test = extract(test_manifest, test_root)

    scaler = Standardiser().fit(x_train)
    w, b = fit_logistic(scaler.transform(x_train), y_train)
    scores = predict(scaler.transform(x_test), w, b)

    metrics = eval_metrics(scores, y_test, n_boot=200)
    metrics["score_std"] = float(scores.std())
    return {
        "eer": metrics["eer"],
        "auc": metrics["auc"],
        "metrics": metrics,
        "train_clips": int(len(y_train)),
        "test_clips": int(len(y_test)),
        "test_bonafide": int(y_test.sum()),
        "test_spoof": int((y_test == 0).sum()),
        "coefficients": {
            name: round(float(weight), 4) for name, weight in zip(FEATURE_NAMES, w, strict=True)
        },
        "intercept": round(float(b), 4),
        "seed": seed,
    }


def verdict(eer: float, near_chance_band: float = 0.10) -> str:
    """Human-readable pass/fail against the plan's near-chance requirement.

    The band is two-sided on purpose. An EER *below* 40% means the eight statistics
    separate the classes; an EER far *above* 60% is equally damning, because a
    reliably inverted classifier is just as separable -- flip its sign.
    """
    distance = abs(eer - 0.5)
    if distance <= near_chance_band:
        return f"PASS: {eer:.2%} EER is within {near_chance_band:.0%} of chance"
    return (
        f"FAIL: {eer:.2%} EER is {distance:.2%} from chance -- eight low-level "
        "statistics separate the classes, so model results on this data are not "
        "safe to trust until the imbalance is explained"
    )


def main() -> None:
    """CLI: ``python -m src.data.lowlevel_cue --train ... --test ... --out ...``."""
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="AffectDF Appendix-G low-level-cue check")
    parser.add_argument("--train", required=True, help="manifest to fit the regression on")
    parser.add_argument("--test", required=True, help="manifest to score")
    parser.add_argument("--train-root", default=None)
    parser.add_argument("--test-root", default=None)
    parser.add_argument("--limit", type=int, default=None, help="cap clips per manifest")
    parser.add_argument("--out", default="experiments/results/lowlevel_cue_check.json")
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    if args.limit:
        train = train.head(args.limit)
        test = test.head(args.limit)

    result = run_check(train, test, args.train_root, args.test_root)
    result["train_manifest"] = args.train
    result["test_manifest"] = args.test
    result["verdict"] = verdict(result["eer"])

    print("\n=== low-level-cue check ===")
    print(f"  train {result['train_clips']} clips -> test {result['test_clips']} clips")
    print(f"  EER {result['eer']:.4f}   AUC {result['auc']:.4f}")
    print("\n  standardised coefficients (magnitude = how much the class hinges on it):")
    for name, weight in sorted(
        result["coefficients"].items(), key=lambda kv: abs(kv[1]), reverse=True
    ):
        print(f"    {name:22s} {weight:+.4f}")
    print(f"\n  {result['verdict']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
