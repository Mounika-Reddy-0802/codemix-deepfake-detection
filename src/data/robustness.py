"""Post-processing before the phone line: the robustness and evasion sweep (plan task A4).

The question an examiner asks of any detector: what if the attacker re-encodes the clone,
adds noise, or changes its speed or pitch, and what does a genuine caller's room do to
the false alarms? Each transform below is applied to a **clean** clip, which is then
level-normalised and sent through the project's simulated phone line (G.711 at 20 dB),
the same chain every telephone result uses. ``double_g711`` instead adds a second line
pass after the first (a call forwarded over a second hop).

The list is fixed here, in code, before anything is scored, so no transform can be
added or dropped after seeing a result. Every random choice (noise, babble segments,
room response) is seeded per clip, so a rendering is reproducible bit for bit.

Codecs (MP3, Opus, AAC) are round trips through ffmpeg at 16 kHz; speed is a resample
(tempo and pitch together, as a sped-up recording sounds); pitch is a shift without a
tempo change (librosa); the room is a synthetic exponentially decaying response with a
0.4 s reverberation time; babble is four genuine talkers mixed together, taken from the
clean adaptation pool (training speakers), so no test speaker is reused as noise.

The sweep sample is fixed by ``manifest``: up to 300 clips each of the new speakers'
genuine speech, their XTTS-v2 and Tortoise clones, and IndicF5 (all clean). ``render``
writes every clip under every transform (resumable) plus one scoring manifest per
transform, which the existing scorer reads unchanged. Each source is cut to its first
6 s before the transform: the detectors score the first 4 s, and 6 s leaves room for the
0.95x speed change::

    python -m src.data.robustness manifest --out data/manifests/robustness_sweep.csv
    python -m src.data.robustness render --manifest data/manifests/robustness_sweep.csv \\
        --data-root D:/dfdata/ext_bundle/clean --out D:/dfdata/robustness_v1
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # import only for type checkers, never at runtime
    import pandas as pd

SR = 16_000
TARGET_DBFS = -23.0
LINE_SNR_DB = 20.0
RT60_S = 0.4
BABBLE_TALKERS = 4

#: every transform of the sweep, in report order; "none" is the reference
TRANSFORMS = (
    "none",
    "mp3_32k",
    "mp3_64k",
    "opus_12k",
    "aac_64k",
    "white_20db",
    "white_10db",
    "babble_20db",
    "babble_10db",
    "reverb_0.4s",
    "speed_0.95",
    "speed_1.05",
    "pitch_-1",
    "pitch_+1",
    "double_g711",
)

#: ffmpeg encoder and bit rate of each codec transform
CODECS = {
    "mp3_32k": ("libmp3lame", "32k", ".mp3"),
    "mp3_64k": ("libmp3lame", "64k", ".mp3"),
    "opus_12k": ("libopus", "12k", ".ogg"),
    "aac_64k": ("aac", "64k", ".m4a"),
}


def codec_round_trip(audio: np.ndarray, encoder: str, bitrate: str, suffix: str) -> np.ndarray:
    """Encode and decode through ffmpeg; the output is trimmed or padded to the input length."""
    import soundfile as sf

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is needed for the codec transforms")
    with tempfile.TemporaryDirectory() as tmp:
        src, mid, out = Path(tmp) / "in.wav", Path(tmp) / f"mid{suffix}", Path(tmp) / "out.wav"
        sf.write(src, np.asarray(audio, dtype=np.float32), SR)
        run = ["ffmpeg", "-y", "-loglevel", "error"]
        subprocess.run(
            [*run, "-i", str(src), "-c:a", encoder, "-b:a", bitrate, str(mid)], check=True
        )
        subprocess.run([*run, "-i", str(mid), "-ar", str(SR), "-ac", "1", str(out)], check=True)
        decoded, _ = sf.read(out, dtype="float32")
    return fit_length(decoded, len(audio))


def fit_length(audio: np.ndarray, n: int) -> np.ndarray:
    """Trim or zero-pad to ``n`` samples (codecs add a few frames of delay or padding)."""
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) >= n:
        return audio[:n]
    return np.pad(audio, (0, n - len(audio)))


def add_noise(audio: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix ``noise`` (looped or cut to length) into ``audio`` at ``snr_db``."""
    reps = int(np.ceil(len(audio) / max(len(noise), 1)))
    noise = np.tile(np.asarray(noise, dtype=np.float64), reps)[: len(audio)]
    signal_power = float(np.mean(np.asarray(audio, dtype=np.float64) ** 2))
    noise_power = float(np.mean(noise**2))
    if signal_power == 0.0 or noise_power == 0.0:
        return np.asarray(audio, dtype=np.float32)
    gain = np.sqrt(signal_power / (noise_power * 10.0 ** (snr_db / 10.0)))
    return (audio + gain * noise).astype(np.float32)


def room_response(rng: np.random.Generator, rt60: float = RT60_S) -> np.ndarray:
    """A synthetic room impulse response: a direct path, then decaying noise (-60 dB at rt60)."""
    n = int(rt60 * SR)
    t = np.arange(n) / SR
    tail = rng.standard_normal(n) * np.exp(-6.9078 * t / rt60)  # ln(1000) = 6.9078
    tail[0] = 0.0
    response = np.concatenate([[1.0], 0.3 * tail / np.max(np.abs(tail))])
    return response / np.sqrt(np.sum(response**2))


def babble(pool: list[np.ndarray], n: int, rng: np.random.Generator) -> np.ndarray:
    """Four talkers from ``pool``, each a random stretch looped to ``n`` samples, summed."""
    if len(pool) < BABBLE_TALKERS:
        raise ValueError(f"babble needs at least {BABBLE_TALKERS} pool clips")
    picks = rng.choice(len(pool), size=BABBLE_TALKERS, replace=False)
    mix = np.zeros(n, dtype=np.float64)
    for k in picks:
        clip = np.asarray(pool[k], dtype=np.float64)
        start = int(rng.integers(0, max(len(clip) - n, 0) + 1))
        stretch = clip[start : start + n]
        mix += np.tile(stretch, int(np.ceil(n / max(len(stretch), 1))))[:n] / (
            np.sqrt(np.mean(stretch**2)) + 1e-9
        )
    return mix


def transform(
    name: str,
    audio: np.ndarray,
    rng: np.random.Generator,
    babble_pool: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Apply one named transform to clean 16 kHz audio (``double_g711`` is applied in ``render``)."""
    audio = np.asarray(audio, dtype=np.float32)
    if name in ("none", "double_g711"):
        return audio
    if name in CODECS:
        return codec_round_trip(audio, *CODECS[name])
    if name.startswith("white_"):
        return add_noise(audio, rng.standard_normal(len(audio)), float(name[6:-2]))
    if name.startswith("babble_"):
        return add_noise(audio, babble(babble_pool or [], len(audio), rng), float(name[7:-2]))
    if name.startswith("reverb_"):
        return np.convolve(audio, room_response(rng), mode="full")[: len(audio)].astype(np.float32)
    if name.startswith("speed_"):
        from src.utils.audio_utils import resample

        factor = float(name[6:])
        return np.asarray(resample(audio, int(round(SR * factor)), SR), dtype=np.float32)
    if name.startswith("pitch_"):
        import librosa

        return librosa.effects.pitch_shift(audio, sr=SR, n_steps=float(name[6:])).astype(np.float32)
    raise ValueError(f"unknown transform {name!r}; known: {TRANSFORMS}")


def line(audio: np.ndarray, seed: int) -> np.ndarray:
    """Level-normalise, then the simulated phone line (G.711 at 20 dB), seeded per clip."""
    from src.data.channel_sim import ChannelConfig, simulate_channel
    from src.utils.audio_utils import rms_normalize

    normalised = rms_normalize(np.asarray(audio, dtype=np.float32), TARGET_DBFS)
    return simulate_channel(normalised, ChannelConfig(codec="g711", snr_db=LINE_SNR_DB, seed=seed))


def render(
    name: str, audio: np.ndarray, seed: int, babble_pool: list[np.ndarray] | None = None
) -> np.ndarray:
    """The full chain for one clip: transform, then the line (twice for ``double_g711``)."""
    rng = np.random.default_rng((seed, TRANSFORMS.index(name)))
    out = line(transform(name, audio, rng, babble_pool), seed)
    if name == "double_g711":
        out = line(out, seed + 1)
    return out


def sweep_manifest(
    groups: dict[str, pd.DataFrame], per_group: int, seed: int = 1234
) -> pd.DataFrame:
    """A fixed sample of up to ``per_group`` clips from each group, tagged with its group."""
    import pandas as pd

    parts = []
    for name, frame in groups.items():
        part = frame.sample(min(per_group, len(frame)), random_state=seed).copy()
        part["group"] = name
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


#: the sweep's source sets: group -> (manifest, row filter as (column, value))
SWEEP_SOURCES = {
    "genuine": ("data/manifests/score_ext.csv", ("label", "bonafide")),
    "xtts_v2": ("data/manifests/score_ext.csv", ("tool", "xtts_v2")),
    "tortoise": ("data/manifests/score_ext.csv", ("tool", "tortoise")),
    "indicf5": ("data/manifests/cm07.csv", ("tool", "indicf5")),
}
PER_GROUP = 300
BABBLE_MANIFEST = "data/manifests/adapt_ext_genuine.csv"
BABBLE_CLIPS = 40
#: the detectors score the first 4 s; 6 s leaves room for the 0.95x speed change
SOURCE_SECONDS = 6.0


def build_manifest(per_group: int = PER_GROUP) -> pd.DataFrame:
    """The fixed sweep sample from the clean new-speaker and IndicF5 manifests."""
    import pandas as pd

    groups = {}
    for group, (path, (column, value)) in SWEEP_SOURCES.items():
        frame = pd.read_csv(path)
        groups[group] = frame[frame[column] == value]
    return sweep_manifest(groups, per_group)


def render_sweep(
    manifest: pd.DataFrame, data_root: str, out_root: str, babble_root: str | None = None
) -> dict[str, int]:
    """Render every clip under every transform; write one scoring manifest per transform."""
    import pandas as pd

    from src.data.dataset import resolve_path
    from src.utils.audio_utils import load_wav, save_wav

    pool_frame = pd.read_csv(BABBLE_MANIFEST).sample(BABBLE_CLIPS, random_state=1234)
    pool = [
        load_wav(resolve_path(p, babble_root or data_root), target_sr=SR)[0]
        for p in pool_frame["filepath"]
    ]
    out = Path(out_root)
    written: dict[str, int] = {}
    for name in TRANSFORMS:
        rows = []
        for i, row in enumerate(manifest.to_dict(orient="records")):
            clip = Path(str(row["filepath"])).name
            dst = out / name / clip
            if not dst.is_file():
                audio, _ = load_wav(resolve_path(row["filepath"], data_root), target_sr=SR)
                audio = audio[: int(SOURCE_SECONDS * SR)]
                dst.parent.mkdir(parents=True, exist_ok=True)
                save_wav(str(dst), render(name, audio, seed=1234 + i, babble_pool=pool), SR)
                written[name] = written.get(name, 0) + 1
            rows.append({**row, "filepath": f"${{DATA_ROOT}}/{name}/{clip}", "transform": name})
        pd.DataFrame(rows).to_csv(out / f"sweep_{name}.csv", index=False)
        print(f"{name}: {len(rows)} clips ({written.get(name, 0)} new)", flush=True)
    return written


def main() -> None:
    """CLI: ``manifest`` fixes the sample; ``render`` writes the transformed clips."""
    import argparse

    parser = argparse.ArgumentParser(description="Robustness and evasion sweep")
    sub = parser.add_subparsers(dest="command", required=True)
    man = sub.add_parser("manifest")
    man.add_argument("--out", default="data/manifests/robustness_sweep.csv")
    man.add_argument("--per-group", type=int, default=PER_GROUP)
    ren = sub.add_parser("render")
    ren.add_argument("--manifest", default="data/manifests/robustness_sweep.csv")
    ren.add_argument("--data-root", required=True, help="root of the clean clips")
    ren.add_argument("--babble-root", default=None, help="root of the babble pool (default: same)")
    ren.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.command == "manifest":
        frame = build_manifest(args.per_group)
        frame.to_csv(args.out, index=False)
        print(frame["group"].value_counts().to_string())
    else:
        import pandas as pd

        render_sweep(pd.read_csv(args.manifest), args.data_root, args.out, args.babble_root)


if __name__ == "__main__":
    main()
