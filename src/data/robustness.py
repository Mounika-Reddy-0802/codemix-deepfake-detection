"""The robustness sweep (A4): what an attacker or a phone can do to a call, fixed before scoring.

The examiner's question is "what if the attacker re-encodes the clone or adds noise?". This
module answers it with a list of transforms written down before any clip was transformed or
scored (``docs/results/robustness_v1.md``), each applied at the place in a call where it
happens:

- **before the line**, on the caller's side: a clone re-encoded (MP3, Opus), played in a noisy
  or echoing room (white noise, babble, reverberation), sped up or slowed down, or shifted in
  pitch;
- **after the line**, on the receiver's side: the handset's call recorder (AAC) or a second
  G.711 leg.

Every clip then goes through the same telephone line as the project's channel bundle (G.711
with noise at 20 dB, seeded by the clip's name), so with no transform the output is exactly
that bundle's clip and every difference in a score is the transform's.

Run (on Kaggle, where the clean clips live)::

    python -m src.data.robustness plan --out data/manifests/robustness_plan.csv
    python -m src.data.robustness render --plan data/manifests/robustness_plan.csv \\
        --babble data/manifests/codemix_adapt_train.csv --data-root ROOT --out OUT
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SAMPLE_RATE = 16_000
BASE_SEED = 1234
#: The line every clip goes through, as in the channel bundle (``channel_g711_20db``).
LINE_CODEC = "g711"
LINE_SNR_DB = 20.0
#: Babble is this many genuine voices, each from a different training-pool speaker.
BABBLE_VOICES = 6
REVERB_RT60_S = 0.5
SEMITONE = 2.0 ** (1.0 / 12.0)


def _ffmpeg(src: str, dst: str, *args: str) -> None:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError("ffmpeg is needed for the codec, tempo and pitch transforms")
    subprocess.run(
        [exe, "-hide_banner", "-loglevel", "error", "-y", "-i", src, *args, dst], check=True
    )


def _through_ffmpeg(audio: np.ndarray, encode: list[str], suffix: str) -> np.ndarray:
    """Encode 16 kHz mono audio with ``encode`` and decode it back to 16 kHz PCM."""
    import soundfile as sf

    with tempfile.TemporaryDirectory() as tmp:
        src, mid, dst = (str(Path(tmp) / n) for n in ("in.wav", f"mid{suffix}", "out.wav"))
        sf.write(src, np.asarray(audio, dtype=np.float32), SAMPLE_RATE)
        _ffmpeg(src, mid, "-ac", "1", *encode)
        _ffmpeg(mid, dst, "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le")
        out, _ = sf.read(dst, dtype="float32")
    return np.asarray(out, dtype=np.float32)


def _filter(audio: np.ndarray, chain: str) -> np.ndarray:
    """Run an ffmpeg audio filter chain on 16 kHz audio (tempo and pitch)."""
    return _through_ffmpeg(audio, ["-af", chain, "-ar", str(SAMPLE_RATE)], ".wav")


def mp3(kbps: int) -> Callable:
    def apply(audio: np.ndarray, rng: np.random.Generator, babble: np.ndarray | None) -> np.ndarray:
        return _through_ffmpeg(audio, ["-c:a", "libmp3lame", "-b:a", f"{kbps}k"], ".mp3")

    return apply


def opus(kbps: int) -> Callable:
    def apply(audio: np.ndarray, rng: np.random.Generator, babble: np.ndarray | None) -> np.ndarray:
        return _through_ffmpeg(audio, ["-c:a", "libopus", "-b:a", f"{kbps}k"], ".ogg")

    return apply


def aac_recorder(audio: np.ndarray, rng: np.random.Generator, babble: np.ndarray | None):
    """The receiving phone's call recorder, as on every real call: 8 kHz AAC at 48 kbit/s."""
    return _through_ffmpeg(audio, ["-ar", "8000", "-c:a", "aac", "-b:a", "48k"], ".m4a")


def white(snr_db: float) -> Callable:
    def apply(audio: np.ndarray, rng: np.random.Generator, babble: np.ndarray | None) -> np.ndarray:
        from src.data.channel_sim import add_noise_at_snr

        noise = rng.standard_normal(len(audio)).astype(np.float32)
        return add_noise_at_snr(audio, noise, snr_db)

    return apply


def babble_noise(snr_db: float) -> Callable:
    def apply(audio: np.ndarray, rng: np.random.Generator, babble: np.ndarray | None) -> np.ndarray:
        from src.data.channel_sim import add_noise_at_snr

        if babble is None:
            raise ValueError("babble needs its voices: pass the mixed training-pool clips")
        return add_noise_at_snr(audio, babble, snr_db)

    return apply


def room_impulse(rng: np.random.Generator, rt60_s: float = REVERB_RT60_S) -> np.ndarray:
    """A synthetic room: the direct sound, then exponentially decaying noise (-60 dB at RT60)."""
    n = int(rt60_s * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    tail = rng.standard_normal(n) * np.exp(-6.9078 * t / rt60_s)
    tail[: int(0.005 * SAMPLE_RATE)] = 0.0  # the first reflections arrive after 5 ms
    impulse = 0.3 * tail / np.sqrt(np.sum(tail**2))
    impulse[0] = 1.0
    return impulse.astype(np.float32)


def reverb(audio: np.ndarray, rng: np.random.Generator, babble: np.ndarray | None) -> np.ndarray:
    from scipy.signal import fftconvolve

    from src.utils.audio_utils import rms

    wet = fftconvolve(audio, room_impulse(rng))[: len(audio)].astype(np.float32)
    level = rms(wet)
    return wet * (rms(audio) / level) if level > 0 else wet


def tempo(factor: float) -> Callable:
    def apply(audio: np.ndarray, rng: np.random.Generator, babble: np.ndarray | None) -> np.ndarray:
        return _filter(audio, f"atempo={factor:.4f}")

    return apply


def pitch(semitones: int) -> Callable:
    def apply(audio: np.ndarray, rng: np.random.Generator, babble: np.ndarray | None) -> np.ndarray:
        ratio = SEMITONE**semitones
        rate = int(round(SAMPLE_RATE * ratio))
        return _filter(audio, f"asetrate={rate},aresample={SAMPLE_RATE},atempo={1.0 / ratio:.6f}")

    return apply


def second_g711(audio: np.ndarray, rng: np.random.Generator, babble: np.ndarray | None):
    """A second G.711 leg, as when a call is forwarded or bridged."""
    from src.data.channel_sim import NARROWBAND_SR, apply_g711_ulaw
    from src.utils.audio_utils import resample

    narrow = apply_g711_ulaw(resample(audio, SAMPLE_RATE, NARROWBAND_SR))
    return resample(narrow, NARROWBAND_SR, SAMPLE_RATE)


@dataclass(frozen=True)
class Transform:
    name: str
    #: ``before`` the line (the caller's side) or ``after`` it (the receiver's side)
    where: str
    what: str
    apply: Callable


#: The pre-registered list (docs/results/robustness_v1.md). Do not add to it after scoring.
TRANSFORMS: tuple[Transform, ...] = (
    Transform("mp3_32k", "before", "MP3 at 32 kbit/s", mp3(32)),
    Transform("mp3_64k", "before", "MP3 at 64 kbit/s", mp3(64)),
    Transform("opus_12k", "before", "Opus at 12 kbit/s, an internet-call codec", opus(12)),
    Transform("white_10db", "before", "white noise at 10 dB SNR", white(10.0)),
    Transform("white_20db", "before", "white noise at 20 dB SNR", white(20.0)),
    Transform("babble_10db", "before", "six other voices at 10 dB SNR", babble_noise(10.0)),
    Transform("babble_20db", "before", "six other voices at 20 dB SNR", babble_noise(20.0)),
    Transform("reverb", "before", "a room with RT60 0.5 s", reverb),
    Transform("tempo_up_5", "before", "5% faster, same pitch", tempo(1.05)),
    Transform("tempo_down_5", "before", "5% slower, same pitch", tempo(0.95)),
    Transform("pitch_up_1", "before", "one semitone higher, same tempo", pitch(1)),
    Transform("pitch_down_1", "before", "one semitone lower, same tempo", pitch(-1)),
    Transform("aac_recorder", "after", "the handset's call recorder, AAC 48 kbit/s", aac_recorder),
    Transform("g711_second", "after", "a second G.711 leg", second_g711),
)
BY_NAME = {t.name: t for t in TRANSFORMS}


def clip_rng(name: str, transform: str) -> np.random.Generator:
    """The same clip and transform always get the same noise, room and babble."""
    from src.data.channel_bundle import clip_seed

    return np.random.default_rng(clip_seed(f"{transform}|{name}", BASE_SEED))


def through_line(
    audio: np.ndarray, name: str, transform: Transform | None, babble: np.ndarray | None = None
) -> np.ndarray:
    """One clean 16 kHz clip -> its transform at its place -> the bundle's telephone line."""
    from src.data.channel_bundle import clip_seed
    from src.data.channel_sim import ChannelConfig, simulate_channel

    rng = clip_rng(name, transform.name) if transform else None
    x = np.asarray(audio, dtype=np.float32)
    if transform and transform.where == "before":
        x = transform.apply(x, rng, babble)
    line = ChannelConfig(
        codec=LINE_CODEC, snr_db=LINE_SNR_DB, in_sr=SAMPLE_RATE, seed=clip_seed(name, BASE_SEED)
    )
    x = simulate_channel(x, line)
    if transform and transform.where == "after":
        x = transform.apply(x, rng, babble)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    return (x / peak).astype(np.float32) if peak > 1.0 else x.astype(np.float32)


#: The pre-registered sets: genuine and cloned new speakers, and IndicF5 (unseen tool).
PLAN_SETS = {
    "new_speakers": ("data/manifests/score_ext.csv", 300, 222),
    "indicf5": ("data/manifests/score_cm07.csv", 300, 300),
}


def build_plan(root: str | Path = ".") -> pd.DataFrame:
    """The clips the sweep scores: a label-stratified subset of each set, seed 1234."""
    from src.data.stratified_subset import stratified_subset

    parts = []
    for name, (manifest, n_genuine, n_per_attack) in PLAN_SETS.items():
        rows = stratified_subset(
            pd.read_csv(Path(root) / manifest), n_genuine, n_per_attack, seed=BASE_SEED
        )
        parts.append(rows.assign(set=name))
    return pd.concat(parts, ignore_index=True)


def babble_for(name: str, speaker: str, pool: pd.DataFrame, load: Callable) -> np.ndarray:
    """Six genuine training-pool voices summed, none of them the clip's own speaker."""
    rng = clip_rng(name, "babble")
    others = pool[pool["speaker"].astype(str) != str(speaker)]
    speakers = rng.choice(sorted(others["speaker"].astype(str).unique()), BABBLE_VOICES, False)
    voices = []
    for spk in speakers:
        rows = others[others["speaker"].astype(str) == spk]
        voices.append(load(rows.iloc[int(rng.integers(len(rows)))]["filepath"]))
    length = max(len(v) for v in voices)
    mix = np.zeros(length, dtype=np.float32)
    for v in voices:
        mix[: len(v)] += v
    return mix


def render(
    plan: pd.DataFrame,
    babble_pool: pd.DataFrame,
    out_dir: str | Path,
    data_root: str | None = None,
    transforms: tuple[str, ...] = tuple(BY_NAME),
) -> pd.DataFrame:
    """Every plan clip with no transform and with each transform; resumable like the bundle."""
    from src.utils.audio_utils import load_wav, save_wav
    from src.utils.paths import resolve

    def load(path: str) -> np.ndarray:
        return load_wav(str(resolve(path, data_root)), target_sr=SAMPLE_RATE)[0]

    genuine_pool = babble_pool[babble_pool["label"].astype(str) == "bonafide"]
    rows = []
    for row in plan.to_dict(orient="records"):
        name = Path(str(row["filepath"])).name
        clean = None
        for transform in (None, *(BY_NAME[t] for t in transforms)):
            tag = transform.name if transform else "none"
            target = Path(out_dir) / tag / name
            if not target.is_file():
                clean = load(row["filepath"]) if clean is None else clean
                noise = None
                if transform and transform.name.startswith("babble"):
                    noise = babble_for(name, row["speaker"], genuine_pool, load)
                target.parent.mkdir(parents=True, exist_ok=True)
                save_wav(str(target), through_line(clean, name, transform, noise), SAMPLE_RATE)
            rows.append(
                {
                    **row,
                    "filepath": f"${{DATA_ROOT}}/{tag}/{name}",
                    "transform": tag,
                    "condition": f"robust_{tag}",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """CLI: ``plan`` writes the frozen clip list, ``render`` makes every transformed clip."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("plan")
    pl.add_argument("--out", default="data/manifests/robustness_plan.csv")
    rd = sub.add_parser("render")
    rd.add_argument("--plan", default="data/manifests/robustness_plan.csv")
    rd.add_argument("--babble", default="data/manifests/codemix_adapt_train.csv")
    rd.add_argument("--data-root", required=True)
    rd.add_argument("--out", required=True)
    rd.add_argument("--transforms", default=",".join(BY_NAME))
    args = parser.parse_args()
    if args.cmd == "plan":
        plan = build_plan()
        plan.to_csv(args.out, index=False)
        print(f"{len(plan)} clips -> {args.out}")
        return
    manifest = render(
        pd.read_csv(args.plan),
        pd.read_csv(args.babble),
        args.out,
        args.data_root,
        tuple(t for t in args.transforms.split(",") if t),
    )
    manifest.to_csv(Path(args.out) / "robustness_manifest.csv", index=False)
    print(f"{len(manifest)} rows -> {Path(args.out) / 'robustness_manifest.csv'}")


if __name__ == "__main__":
    main()
