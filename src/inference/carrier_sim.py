"""What a carrier does to the detector's scores, measured rather than assumed.

The live demo scores a genuine caller lower than offline scoring of the same clip
does. The obvious explanation is the codec: the browser carries the call with Opus
while the detector is trained on G.711 telephone audio, and the cue it uses is
phase-level. This module exists to test that explanation instead of asserting it.

Each carrier is applied to a clip offline and the clip is rescored through the same
streaming path the live system uses, so the only difference between the two numbers
is the carrier transform.

    python -m src.inference.carrier_sim --clips /path/to/demo_clips

The answer, for the record, is that the codec does **not** account for the live
drop: see ``docs/results/carrier_effect_v1.md``.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
#: WebRTC negotiates Opus at 48 kHz; the server resamples both ways around it.
WIRE_RATE = 48_000
OUT_JSON = "experiments/results/carrier_effect.json"


def opus_roundtrip(
    audio: np.ndarray, sr: int = SAMPLE_RATE, wire_sr: int = WIRE_RATE
) -> np.ndarray:
    """The browser path: resample up, Opus encode and decode, resample back.

    The resampling is part of the transform, not incidental: a browser sends 48 kHz
    Opus and the server resamples to the 16 kHz the model expects, so simulating the
    codec alone would leave out half of what the carrier does.
    """
    import soundfile as sf

    from src.utils.audio_utils import resample

    source = np.asarray(audio, dtype=np.float32)
    wide = resample(source, sr, wire_sr) if sr != wire_sr else source
    buffer = io.BytesIO()
    sf.write(buffer, wide, wire_sr, format="OGG", subtype="OPUS")
    buffer.seek(0)
    decoded, decoded_sr = sf.read(buffer, dtype="float32", always_2d=False)
    if decoded_sr != sr:
        decoded = resample(decoded, decoded_sr, sr)
    return np.asarray(decoded, dtype=np.float32)


def g711_roundtrip(audio: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
    """The telephone path the model was trained on, for comparison."""
    from src.data.channel_sim import ChannelConfig, simulate_channel

    return simulate_channel(
        np.asarray(audio, dtype=np.float32),
        ChannelConfig(codec="g711", snr_db=snr_db, in_sr=SAMPLE_RATE, seed=0),
    )


#: name -> transform. ``none`` is the control: the clip as the evaluation scores it.
CARRIERS = {
    "none": lambda audio: np.asarray(audio, dtype=np.float32),
    "opus_48k": opus_roundtrip,
    "g711_20db": g711_roundtrip,
}


@dataclass(frozen=True)
class CarrierResult:
    """How one carrier changed one clip."""

    clip: str
    carrier: str
    windows: int
    low_windows: int  # below the threshold
    min_score: float
    mean_score: float

    def as_dict(self) -> dict:
        return {
            "clip": self.clip,
            "carrier": self.carrier,
            "windows": self.windows,
            "low_windows": self.low_windows,
            "min_score": round(self.min_score, 4),
            "mean_score": round(self.mean_score, 4),
        }


def window_scores(audio: np.ndarray, score_fn) -> list[float]:
    """Every scored window, through the same path the live system uses."""
    from src.inference.streaming import StreamingScorer

    return [r.score for r in StreamingScorer(score_fn).push(audio) if r.score is not None]


def measure(
    audio: np.ndarray, clip: str, carrier: str, score_fn, threshold: float = 0.5
) -> CarrierResult:
    """Apply one carrier to one clip and score the result."""
    from src.utils.audio_utils import rms_normalize

    transform = CARRIERS[carrier]
    carried = rms_normalize(transform(audio), -23.0)
    scores = window_scores(carried, score_fn)
    if not scores:
        return CarrierResult(clip, carrier, 0, 0, float("nan"), float("nan"))
    return CarrierResult(
        clip=clip,
        carrier=carrier,
        windows=len(scores),
        low_windows=sum(1 for s in scores if s < threshold),
        min_score=float(min(scores)),
        mean_score=float(np.mean(scores)),
    )


def main() -> int:
    import argparse
    import os

    from src.inference.predict import load_detector, make_score_fn
    from src.utils.audio_utils import load_wav, resample, rms_normalize

    parser = argparse.ArgumentParser(description="Measure what each carrier does to the scores")
    parser.add_argument("--clips", default=os.environ.get("DEMO_CLIPS_DIR", ""))
    parser.add_argument("--checkpoint", default=os.environ.get("DFD_CHECKPOINT", ""))
    parser.add_argument("--out", default=OUT_JSON)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    folder = Path(args.clips)
    files = sorted(folder.glob("*.wav")) if folder.is_dir() else []
    if not files:
        print(f"no clips in {folder}; pass --clips or set DEMO_CLIPS_DIR")
        return 1
    if not Path(args.checkpoint).is_file():
        print(f"checkpoint not found: {args.checkpoint}")
        return 1

    model, device = load_detector(args.checkpoint)
    score_fn = make_score_fn(model, device)
    print(f"model on {device}, {len(files)} clips\n")

    results = []
    header = f"{'clip':<18}{'carrier':<12}{'windows':>8}{'low':>6}{'min':>9}{'mean':>9}"
    print(header)
    print("-" * len(header))
    for path in files:
        audio, sr = load_wav(str(path))
        if sr != SAMPLE_RATE:
            audio = resample(audio, sr, SAMPLE_RATE)
        audio = rms_normalize(audio, -23.0)
        for carrier in CARRIERS:
            result = measure(audio, path.stem, carrier, score_fn, args.threshold)
            results.append(result.as_dict())
            print(
                f"{result.clip:<18}{result.carrier:<12}{result.windows:>8}"
                f"{result.low_windows:>6}{result.min_score:>9.3f}{result.mean_score:>9.3f}"
            )
        print()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "what": "each demo clip scored offline under each carrier transform",
                "why": (
                    "to test, rather than assume, that the codec explains why a genuine "
                    "caller scores lower on a live call than offline"
                ),
                "threshold": args.threshold,
                "checkpoint": Path(args.checkpoint).name,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
