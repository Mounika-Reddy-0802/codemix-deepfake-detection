"""Audio file -> P(bonafide) -> verdict; shared by the live call and the Gradio demo.

One checkpoint, one threshold, one answer per file. A file longer than one window is
scored the way a call is: 4 s windows every 2 s through ``StreamingScorer``, silence
skipped, every window normalised to -23 dBFS as in training. The file score is the
**minimum** window score, not the mean. A caller who splices ten seconds of cloned
voice into a minute of real speech should not be averaged away.

**Where the threshold comes from, in order:**

1. ``--threshold`` given explicitly;
2. ``configs/threshold.yaml``, frozen from the DET curve at the results freeze (W8-T5);
3. ``--threshold-from <results.json>``: the equal-error-rate threshold of a scored
   run. Marked **provisional** in every output, because an EER point is a
   measurement, not a chosen operating point.

With none of these the CLI stops. It never falls back to 0.5: at 0.5 the clean
adapter passes 47.2% of Tortoise fakes (P-025).

Torch is imported only inside :func:`load_detector` and :func:`make_score_fn`, so
the threshold and verdict logic run in CI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.inference.streaming import SAMPLE_RATE, StreamingScorer

DEFAULT_THRESHOLD_FILE = "configs/threshold.yaml"


@dataclass(frozen=True)
class Threshold:
    """An operating point on P(bonafide): scores below it are called spoof."""

    value: float
    source: str
    provisional: bool

    def __post_init__(self) -> None:
        if not 0.0 < self.value < 1.0:
            raise ValueError(f"threshold must be strictly between 0 and 1, got {self.value}")


@dataclass(frozen=True)
class Prediction:
    """The verdict for one file."""

    path: str
    score: float | None  # min window P(bonafide); None when the file held no speech
    verdict: str  # "bonafide" | "spoof" | "no speech"
    windows: int
    skipped: int
    seconds: float
    threshold: Threshold

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "score": self.score,
            "verdict": self.verdict,
            "windows": self.windows,
            "skipped_silent_windows": self.skipped,
            "seconds": round(self.seconds, 2),
            "threshold": self.threshold.value,
            "threshold_source": self.threshold.source,
            "provisional": self.threshold.provisional,
        }


def resolve_threshold(
    explicit: float | None = None,
    threshold_file: str = DEFAULT_THRESHOLD_FILE,
    results_json: str | None = None,
) -> Threshold:
    """Pick the operating point by the documented precedence, or refuse."""
    if explicit is not None:
        return Threshold(float(explicit), "command line", provisional=True)
    file = Path(threshold_file)
    if file.is_file():
        import yaml

        raw = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
        return Threshold(float(raw["threshold"]), str(file), provisional=False)
    if results_json:
        pooled = json.loads(Path(results_json).read_text(encoding="utf-8"))
        pooled = pooled.get("pooled", pooled)
        return Threshold(float(pooled["threshold"]), f"EER point of {results_json}", True)
    raise FileNotFoundError(
        f"no operating threshold: {threshold_file} is set at the results freeze (W8-T5). "
        "Pass --threshold, or --threshold-from a scored results JSON for a provisional run."
    )


def verdict_for(score: float | None, threshold: Threshold) -> str:
    """``bonafide`` at or above the threshold (the metrics module's convention)."""
    if score is None:
        return "no speech"
    return "bonafide" if score >= threshold.value else "spoof"


def predict_audio(audio: np.ndarray, score_fn, threshold: Threshold, path: str = "") -> Prediction:
    """Score 16 kHz float32 audio already in memory."""
    scorer = StreamingScorer(score_fn)
    samples = audio.astype(np.float32, copy=False)
    window = scorer.windower.window
    if 0 < samples.size < window:  # short clip: pad so it yields one window
        samples = np.pad(samples, (0, window - samples.size))
    results = list(scorer.push(samples))
    scored = [r.score for r in results if not r.skipped]
    score = min(scored) if scored else None
    return Prediction(
        path=path,
        score=score,
        verdict=verdict_for(score, threshold),
        windows=len(results),
        skipped=len(results) - len(scored),
        seconds=audio.size / SAMPLE_RATE,
        threshold=threshold,
    )


def load_detector(checkpoint: str, encoder: str = "wav2vec2-base", device: str | None = None):
    """Load any of the project's checkpoints (S1, S2 LoRA, S3) for inference."""
    import torch

    from src.models.detector import Detector
    from src.training.evaluate import _encoder_id, _restore_lora
    from src.utils.device import resolve_device

    resolved = resolve_device(device)
    model = Detector(_encoder_id(encoder))
    state = torch.load(checkpoint, map_location=resolved, weights_only=False)
    _restore_lora(model, state)
    model.load_state_dict(state["model"] if "model" in state else state)
    return model.to(resolved).eval(), resolved


def make_score_fn(model, device: str):
    """Wrap a loaded detector as a ``window -> P(bonafide)`` callable."""
    import torch

    def score(window: np.ndarray) -> float:
        with torch.no_grad():
            wav = torch.from_numpy(window).unsqueeze(0).to(device)
            lengths = torch.tensor([window.size], device=device)
            return float(torch.softmax(model(wav, lengths=lengths), dim=1)[0, 1])

    return score


def predict_file(path: str, score_fn, threshold: Threshold) -> Prediction:
    """Load a file (any rate, resampled to 16 kHz) and predict."""
    from src.utils.audio_utils import load_wav

    audio, _ = load_wav(path, target_sr=SAMPLE_RATE)
    return predict_audio(audio, score_fn, threshold, path=path)


def main() -> None:
    """CLI: ``python -m src.inference.predict --checkpoint best.pt clip.wav [...]``."""
    import argparse

    parser = argparse.ArgumentParser(description="Real or cloned? One verdict per file")
    parser.add_argument("files", nargs="+")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--threshold-file", default=DEFAULT_THRESHOLD_FILE)
    parser.add_argument("--threshold-from", default=None, help="results JSON with an EER threshold")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    threshold = resolve_threshold(args.threshold, args.threshold_file, args.threshold_from)
    if threshold.provisional:
        print(f"PROVISIONAL threshold {threshold.value:.4f} from {threshold.source}")
    model, device = load_detector(args.checkpoint, device=args.device)
    score_fn = make_score_fn(model, device)
    for path in args.files:
        print(json.dumps(predict_file(path, score_fn, threshold).as_dict()))


if __name__ == "__main__":
    main()
