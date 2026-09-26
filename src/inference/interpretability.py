"""What the detector responds to: perturbation, layer and language probes.

The detector reads the raw waveform and has no hand-written features, so the only
way to say what it uses is to change one property of the audio at a time and watch
the score. Three probes, one per question:

1. **Perturbations.** Each one destroys a single candidate cue and keeps the rest.
   Shuffling 100 ms chunks removes words, grammar, code-mixing and prosody but keeps
   the local waveform. Phase scrambling keeps the magnitude spectrogram exactly and
   randomises only the phase. Time-stretch and pitch-shift change rhythm and pitch.
   Level, codec and noise changes check robustness.
2. **Layer separation.** How well real and cloned windows separate at the
   convolutional output and after each transformer layer, scored on held-out
   windows along the class-mean direction fitted on the rest. Early separation means
   the decision rests on acoustic detail, not on linguistic content.
3. **Language.** Real and fake English (ASVspoof 2019 LA) through a model adapted on
   Hindi-English speech. If the verdict tracked language, real English would be
   rejected.

Scoring mirrors the live path: 4 s windows every 2 s, windows below the silence
gate skipped, each window normalised to -23 dBFS (:mod:`src.inference.streaming`).

Run: ``python -m src.inference.interpretability --config configs/interpretability.yaml``
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from src.data.channel_sim import apply_g711_ulaw
from src.inference.streaming import MIN_RMS_DBFS, normalise, rms_dbfs
from src.training.metrics import roc_auc

SAMPLE_RATE = 16_000
WINDOW_SECONDS = 4.0
HOP_SECONDS = 2.0

#: ``(audio, rng) -> perturbed audio``; the rng is only used by random perturbations.
Perturbation = Callable[[np.ndarray, np.random.Generator], np.ndarray]


# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #
def speech_windows(
    audio: np.ndarray,
    max_windows: int,
    window_seconds: float = WINDOW_SECONDS,
    hop_seconds: float = HOP_SECONDS,
    sr: int = SAMPLE_RATE,
    min_rms_dbfs: float = MIN_RMS_DBFS,
) -> list[np.ndarray]:
    """Up to ``max_windows`` normalised full-length windows that pass the silence gate."""
    window = int(round(window_seconds * sr))
    hop = int(round(hop_seconds * sr))
    out: list[np.ndarray] = []
    for start in range(0, max(0, audio.size - window) + 1, hop):
        chunk = audio[start : start + window]
        if chunk.size < window:
            break
        if rms_dbfs(chunk) > min_rms_dbfs:
            out.append(normalise(chunk))
        if len(out) >= max_windows:
            break
    return out


# --------------------------------------------------------------------------- #
# Perturbations (each changes one property of the audio)
# --------------------------------------------------------------------------- #
def shuffle_chunks(
    audio: np.ndarray, chunk_ms: float, rng: np.random.Generator, sr: int = SAMPLE_RATE
) -> np.ndarray:
    """Cut into ``chunk_ms`` pieces and put them back in random order.

    The trailing partial chunk is dropped so every piece has the same length.
    """
    n = max(1, int(round(sr * chunk_ms / 1000.0)))
    k = audio.size // n
    if k == 0:
        return audio.astype(np.float32)
    pieces = audio[: k * n].reshape(k, n).copy()
    rng.shuffle(pieces)
    return pieces.reshape(-1).astype(np.float32)


def stft_roundtrip(audio: np.ndarray, n_fft: int = 512) -> np.ndarray:
    """Analyse and resynthesise with no change: a control for the phase probes."""
    import librosa

    hop = n_fft // 4
    spec = librosa.stft(audio, n_fft=n_fft, hop_length=hop)
    return librosa.istft(spec, hop_length=hop, length=audio.size).astype(np.float32)


def phase_scramble(audio: np.ndarray, rng: np.random.Generator, n_fft: int = 512) -> np.ndarray:
    """Keep the magnitude spectrogram, replace every phase with a random one."""
    import librosa

    hop = n_fft // 4
    spec = librosa.stft(audio, n_fft=n_fft, hop_length=hop)
    phase = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, spec.shape))
    return librosa.istft(np.abs(spec) * phase, hop_length=hop, length=audio.size).astype(np.float32)


def white_noise(audio: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add white noise at ``snr_db`` relative to the clip's own power."""
    noise = rng.standard_normal(audio.size).astype(np.float32)
    power = float(np.mean(audio.astype(np.float64) ** 2))
    noise_power = float(np.mean(noise.astype(np.float64) ** 2))
    if power <= 0.0 or noise_power <= 0.0:
        return audio.astype(np.float32)
    noise *= np.sqrt(power / noise_power) / (10.0 ** (snr_db / 20.0))
    return (audio + noise).astype(np.float32)


def band_filter(
    audio: np.ndarray, low_hz: float, high_hz: float, keep: bool, sr: int = SAMPLE_RATE
) -> np.ndarray:
    """Keep only, or remove only, the band ``[low_hz, high_hz]`` (6th-order Butterworth)."""
    from scipy.signal import butter, sosfiltfilt

    nyquist = sr / 2.0
    low = max(low_hz, 20.0)
    high = min(high_hz, nyquist - 20.0)
    if keep and low_hz <= 0.0:
        sos = butter(6, high, btype="lowpass", fs=sr, output="sos")
    else:
        sos = butter(6, [low, high], btype="bandpass" if keep else "bandstop", fs=sr, output="sos")
    return sosfiltfilt(sos, audio).astype(np.float32)


def time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    """Phase-vocoder time-stretch; ``rate`` > 1 is faster. Pitch is unchanged."""
    import librosa

    return librosa.effects.time_stretch(audio, rate=rate).astype(np.float32)


def pitch_shift(audio: np.ndarray, semitones: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Shift pitch by ``semitones`` at constant duration."""
    import librosa

    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones).astype(np.float32)


def registry(sr: int = SAMPLE_RATE) -> dict[str, Perturbation]:
    """Every named perturbation. Configs select from these names."""

    def fixed(fn: Callable[[np.ndarray], np.ndarray]) -> Perturbation:
        return lambda audio, _rng: fn(audio)

    return {
        "original": fixed(lambda a: a.astype(np.float32)),
        "volume_x0.1": fixed(lambda a: (a * 0.1).astype(np.float32)),
        "mu_law_pass": fixed(apply_g711_ulaw),
        "noise_20db": lambda a, r: white_noise(a, 20.0, r),
        "noise_10db": lambda a, r: white_noise(a, 10.0, r),
        "stft_roundtrip": fixed(stft_roundtrip),
        "phase_vocoder_rate_1.0": fixed(lambda a: time_stretch(a, 1.0)),
        "phase_scramble": phase_scramble,
        "time_reverse": fixed(lambda a: a[::-1].astype(np.float32)),
        "shuffle_500ms": lambda a, r: shuffle_chunks(a, 500, r, sr),
        "shuffle_100ms": lambda a, r: shuffle_chunks(a, 100, r, sr),
        "shuffle_25ms": lambda a, r: shuffle_chunks(a, 25, r, sr),
        "shuffle_5ms": lambda a, r: shuffle_chunks(a, 5, r, sr),
        "keep_0_1khz": fixed(lambda a: band_filter(a, 0, 1000, True, sr)),
        "keep_1_2khz": fixed(lambda a: band_filter(a, 1000, 2000, True, sr)),
        "keep_2_4khz": fixed(lambda a: band_filter(a, 2000, 4000, True, sr)),
        "remove_0.3_1khz": fixed(lambda a: band_filter(a, 300, 1000, False, sr)),
        "remove_2_4khz": fixed(lambda a: band_filter(a, 2000, 3990, False, sr)),
        "slower_x1.4": fixed(lambda a: time_stretch(a, 0.7)),
        "faster_x0.7": fixed(lambda a: time_stretch(a, 1.4)),
        "pitch_up_3": fixed(lambda a: pitch_shift(a, 3.0, sr)),
        "pitch_down_3": fixed(lambda a: pitch_shift(a, -3.0, sr)),
    }


# --------------------------------------------------------------------------- #
# Probes
# --------------------------------------------------------------------------- #
def mean_window_score(
    audio: np.ndarray, score_fn: Callable[[np.ndarray], float], max_windows: int
) -> float | None:
    """Mean P(genuine) over the clip's speech windows; ``None`` when there are none."""
    windows = speech_windows(audio, max_windows)
    if not windows:
        return None
    return float(np.mean([score_fn(w) for w in windows]))


def perturbation_table(
    clips: dict[str, np.ndarray],
    names: list[str],
    score_fn: Callable[[np.ndarray], float],
    max_windows: int,
    seed: int,
) -> dict[str, dict[str, float | None]]:
    """``{perturbation: {clip: mean score}}``, seeded per (perturbation, clip) pair."""
    available = registry()
    unknown = [n for n in names if n not in available]
    if unknown:
        raise ValueError(f"unknown perturbations: {unknown}")
    table: dict[str, dict[str, float | None]] = {}
    for i, name in enumerate(names):
        row: dict[str, float | None] = {}
        for j, (clip, audio) in enumerate(clips.items()):
            rng = np.random.default_rng([seed, i, j])
            row[clip] = mean_window_score(available[name](audio, rng), score_fn, max_windows)
        table[name] = row
    return table


def layer_separation(
    features: np.ndarray, labels: np.ndarray, train_frac: float, seed: int
) -> list[float]:
    """Held-out AUC per layer along the class-mean direction fitted on the train split.

    ``features`` is ``(windows, layers, dims)`` of time-averaged hidden states;
    ``labels`` is 1 for genuine and 0 for cloned. The direction is the difference of
    class means on the train split, so there is nothing to overfit beyond two means.
    """
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if features.ndim != 3 or features.shape[0] != labels.size:
        raise ValueError("features must be (windows, layers, dims) matching labels")
    order = np.random.default_rng(seed).permutation(labels.size)
    cut = int(round(train_frac * labels.size))
    train, test = order[:cut], order[cut:]
    if len(set(labels[train])) < 2 or len(set(labels[test])) < 2:
        raise ValueError("both splits need both classes")
    aucs = []
    for layer in range(features.shape[1]):
        x = features[:, layer, :]
        direction = x[train][labels[train] == 1].mean(0) - x[train][labels[train] == 0].mean(0)
        direction /= np.linalg.norm(direction) + 1e-12
        aucs.append(roc_auc(x[test] @ direction, labels[test]))
    return aucs


def hidden_state_means(model, windows: list[np.ndarray], device: str) -> np.ndarray:
    """``(windows, layers, dims)``: each hidden state averaged over time."""
    import torch

    encoder = model.encoder.model
    rows = []
    with torch.no_grad():
        for window in windows:
            wav = torch.from_numpy(window).unsqueeze(0).to(device)
            states = encoder(wav, output_hidden_states=True).hidden_states
            rows.append(np.stack([s[0].mean(0).float().cpu().numpy() for s in states]))
    return np.stack(rows)


def english_check(
    protocol: str,
    audio_dir: str,
    per_class: int,
    windows_per_clip: int,
    seed: int,
    score_fn: Callable[[np.ndarray], float],
) -> dict[str, dict[str, float]]:
    """Score real and fake ASVspoof clips; clips shorter than one window are skipped."""
    from src.utils.audio_utils import load_wav

    rows = [line.split() for line in Path(protocol).read_text().splitlines() if line.strip()]
    rng = np.random.default_rng(seed)
    out: dict[str, dict[str, float]] = {}
    for label in ("bonafide", "spoof"):
        ids = [r[1] for r in rows if r[-1] == label]
        rng.shuffle(ids)
        scores = []
        for clip_id in ids[:per_class]:
            audio, _ = load_wav(str(Path(audio_dir) / f"{clip_id}.flac"), target_sr=SAMPLE_RATE)
            s = mean_window_score(np.asarray(audio, dtype=np.float32), score_fn, windows_per_clip)
            if s is not None:
                scores.append(s)
        arr = np.array(scores)
        correct = arr >= 0.5 if label == "bonafide" else arr < 0.5
        out[label] = {
            "clips_drawn": float(min(per_class, len(ids))),
            "clips_scored": float(arr.size),
            "mean_score": float(arr.mean()) if arr.size else float("nan"),
            "correct_pct": float(100.0 * correct.mean()) if arr.size else float("nan"),
        }
    return out


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run(config: dict) -> dict:
    """Load the checkpoint once and run the probes the config enables."""
    import soundfile as sf

    from src.inference.predict import load_detector, make_score_fn

    model, device = load_detector(config["checkpoint"], device=config.get("device"))
    score_fn = make_score_fn(model, device)
    seed = int(config.get("seed", 0))
    seconds = float(config.get("clip_seconds", 30.0))

    clip_dir = Path(config["demo_clips_dir"])

    def load(file: str) -> np.ndarray:
        audio, sr = sf.read(str(clip_dir / file), dtype="float32")
        if sr != SAMPLE_RATE:
            raise ValueError(f"{file}: expected {SAMPLE_RATE} Hz, got {sr}")
        return audio[: int(seconds * SAMPLE_RATE)]

    clips = {name: load(spec["file"]) for name, spec in config["clips"].items()}
    result: dict = {"checkpoint": config["checkpoint"], "seed": seed, "clip_seconds": seconds}

    probe = config.get("perturbations")
    if probe:
        result["perturbations"] = perturbation_table(
            clips, list(probe["names"]), score_fn, int(probe["max_windows"]), seed
        )

    layers = config.get("layers")
    if layers:
        windows, labels = [], []
        for name, spec in config["clips"].items():
            if name not in layers["clips"]:
                continue
            ws = speech_windows(clips[name], int(layers["windows_per_clip"]))
            windows += ws
            labels += [1 if spec["label"] == "bonafide" else 0] * len(ws)
        feats = hidden_state_means(model, windows, device)
        aucs = layer_separation(feats, np.array(labels), float(layers["train_frac"]), seed)
        result["layers"] = {
            "genuine_windows": int(sum(labels)),
            "cloned_windows": int(len(labels) - sum(labels)),
            "held_out_auc": {
                ("cnn_output" if i == 0 else f"transformer_{i}"): a for i, a in enumerate(aucs)
            },
        }

    english = config.get("english")
    if english:
        root = Path(config["asvspoof_root"])
        result["english"] = english_check(
            str(root / english["protocol"]),
            str(root / english["audio_dir"]),
            int(english["per_class"]),
            int(english["windows_per_clip"]),
            seed,
            score_fn,
        )
    return result


def main() -> None:
    """CLI: run the probes named in a YAML config and write one JSON."""
    import yaml

    parser = argparse.ArgumentParser(description="What the detector responds to")
    parser.add_argument("--config", default="configs/interpretability.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    result = run(config)
    out = Path(config["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
