"""Is this voice real or cloned? One answer for any recording or audio file.

The input can be anything a person hands the detector: speech recorded on the spot through
a microphone, a phone-call recording (m4a, amr, 3gp), a voice note (ogg, opus), an mp3 or
wav from anywhere, or a recording of audio played aloud from another device. Every input
goes through the same five steps, which are the steps the reported accuracy was measured
with (``src.inference.any_audio_eval``):

1. **Decode** to 16 kHz mono float (``ffmpeg`` for anything ``soundfile`` cannot read).
2. **Trim** silent ends: frames quieter than -45 dBFS, as the training clips were trimmed.
3. **Telephone front end** (default ``g711``): an 8 kHz G.711 mu-law round trip, the
   same ``telephone_leg`` the 30 real calls were scored through. The final-recipe
   detectors were trained only on telephone-band audio (G.711 and AMR-NB), so this puts a
   wideband microphone recording or a studio file in the domain the model knows. Audio
   that already passed through a phone is barely changed by one more round trip. Whether
   the front end helps on clean files is measured, not assumed: ``any_audio_eval`` scores
   the clean test sets with it and without it.
4. **Windows**: 4 s every 2 s, plus one window ending at the last sample so the tail is
   never dropped. A window quieter than -50 dBFS is skipped. Each window is normalised to
   -23 dBFS (the training level) and scored, giving P(genuine).
5. **Decision**: the recording's score is the **mean** P(genuine) over its speech windows,
   compared with the checkpoint's frozen threshold, chosen on development data only.

Why the mean and not the minimum that ``src.inference.predict`` uses: the question here is
whether a whole recording is a cloned voice, and one odd window in a minute of real speech
should not condemn it. Every window's score is returned as well, so a recording that is real
in one part and cloned in another stays visible.

Torch is imported only when a detector is loaded, so the pipeline is tested in CI with a
stand-in score function.

    python -m src.inference.any_audio clip1.mp3 call.m4a note.ogg
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from src.inference.streaming import normalise, rms_dbfs

SAMPLE_RATE = 16_000
WINDOW_SECONDS = 4.0
HOP_SECONDS = 2.0
TRIM_DBFS = -45.0  # silent-end trim, as the training clips (audio_utils.trim_silence)
SILENCE_DBFS = -50.0  # a window this quiet carries no speech (same as the live path)
MIN_SPEECH_SECONDS = 2.0  # the shortest training clip
FRONT_ENDS = ("g711", "none")
CONFIG_FILE = "configs/voice_check.yaml"

REAL, CLONED, NO_SPEECH = "real", "cloned", "no speech"

ScoreFn = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class WindowScore:
    """One 4 s window: where it sits, how loud it was, and its P(genuine) (None if silent)."""

    start_s: float
    end_s: float
    level_dbfs: float
    p_genuine: float | None


@dataclass(frozen=True)
class VoiceVerdict:
    """The answer for one recording."""

    verdict: str
    p_genuine: float | None
    threshold: float
    seconds: float
    speech_seconds: float
    front_end: str
    windows: tuple[WindowScore, ...] = field(default=())
    source: str = ""

    @property
    def scored(self) -> list[float]:
        return [w.p_genuine for w in self.windows if w.p_genuine is not None]

    @property
    def windows_agreeing(self) -> int:
        """How many scored windows fall on the same side of the threshold as the verdict."""
        if self.verdict == REAL:
            return sum(p >= self.threshold for p in self.scored)
        if self.verdict == CLONED:
            return sum(p < self.threshold for p in self.scored)
        return 0

    def summary(self) -> dict:
        out = asdict(self)
        out.pop("windows")
        out["windows_scored"] = len(self.scored)
        out["windows_agreeing"] = self.windows_agreeing
        return out


def load_audio(path: str | Path) -> np.ndarray:
    """Any audio file as 16 kHz mono float32 (wav/flac directly, everything else via ffmpeg)."""
    from src.utils.audio_utils import load_any

    audio = np.asarray(load_any(str(path), target_sr=SAMPLE_RATE), dtype=np.float32)
    if audio.ndim != 1:
        audio = audio.mean(axis=-1)
    return audio


def trim(audio: np.ndarray) -> np.ndarray:
    from src.utils.audio_utils import trim_silence

    return trim_silence(audio, SAMPLE_RATE, threshold_dbfs=TRIM_DBFS)


def front_end(audio: np.ndarray, kind: str = "g711") -> np.ndarray:
    """Bring any input to the training domain (``g711``) or leave it as it is (``none``)."""
    if kind == "none":
        return np.asarray(audio, dtype=np.float32)
    if kind != "g711":
        raise ValueError(f"unknown front end {kind!r}; expected one of {FRONT_ENDS}")
    from src.inference.real_calls import telephone_leg

    out = telephone_leg(np.asarray(audio, dtype=np.float32), SAMPLE_RATE)
    # resampling can shift the length by a sample or two; keep the timeline exact
    if out.size >= audio.size:
        return out[: audio.size].astype(np.float32)
    return np.pad(out, (0, audio.size - out.size)).astype(np.float32)


def window_starts(n_samples: int) -> list[int]:
    """Start samples of the 4 s windows: every 2 s, plus one ending at the last sample.

    A recording shorter than one window is a single window (the detector brings it to its
    input length the way short training clips were).
    """
    win = int(WINDOW_SECONDS * SAMPLE_RATE)
    hop = int(HOP_SECONDS * SAMPLE_RATE)
    if n_samples <= win:
        return [0]
    starts = list(range(0, n_samples - win + 1, hop))
    if starts[-1] + win < n_samples:
        starts.append(n_samples - win)
    return starts


def judge(
    audio: np.ndarray,
    score_fn: ScoreFn,
    threshold: float,
    front: str = "g711",
    source: str = "",
) -> VoiceVerdict:
    """Score one recording (16 kHz mono float) and decide real or cloned."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    seconds = round(audio.size / SAMPLE_RATE, 2)
    speech = trim(audio)
    common = {
        "threshold": float(threshold),
        "seconds": seconds,
        "front_end": front,
        "source": source,
    }
    if speech.size < MIN_SPEECH_SECONDS * SAMPLE_RATE:
        return VoiceVerdict(
            NO_SPEECH, None, speech_seconds=round(speech.size / SAMPLE_RATE, 2), **common
        )
    shaped = front_end(speech, front)
    win = int(WINDOW_SECONDS * SAMPLE_RATE)
    windows = []
    for start in window_starts(shaped.size):
        segment = shaped[start : start + win]
        level = rms_dbfs(segment)
        p = None if level < SILENCE_DBFS else float(score_fn(normalise(segment)))
        windows.append(
            WindowScore(
                round(start / SAMPLE_RATE, 2),
                round((start + segment.size) / SAMPLE_RATE, 2),
                round(level, 2),
                p,
            )
        )
    scored = [w.p_genuine for w in windows if w.p_genuine is not None]
    speech_seconds = round(speech.size / SAMPLE_RATE, 2)
    if not scored:
        return VoiceVerdict(
            NO_SPEECH, None, speech_seconds=speech_seconds, windows=tuple(windows), **common
        )
    p_mean = float(np.mean(scored))
    verdict = REAL if p_mean >= threshold else CLONED
    return VoiceVerdict(
        verdict, p_mean, speech_seconds=speech_seconds, windows=tuple(windows), **common
    )


# ------------------------------------------------------------------------- detector
@dataclass(frozen=True)
class DetectorConfig:
    """Which checkpoint answers, at which frozen threshold, with which front end."""

    kind: str
    checkpoint: str
    threshold_file: str
    front_end: str = "g711"

    @classmethod
    def from_file(cls, path: str = CONFIG_FILE) -> DetectorConfig:
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(
            kind=str(raw["kind"]),
            checkpoint=str(raw["checkpoint"]),
            threshold_file=str(raw["threshold_file"]),
            front_end=str(raw.get("front_end", "g711")),
        )

    def threshold(self) -> float:
        from src.inference.predict import resolve_threshold

        return resolve_threshold(threshold_file=self.threshold_file).value


def load_detector(config: DetectorConfig, device: str = "cpu") -> ScoreFn:
    """The ``window -> P(genuine)`` function of the configured checkpoint."""
    if not Path(config.checkpoint).is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {config.checkpoint}. Set its path in {CONFIG_FILE}."
        )
    from src.inference.predict import load_score_fn

    return load_score_fn(config.checkpoint, kind=config.kind, device=device)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Real or cloned? Judge audio files.")
    parser.add_argument("files", nargs="+")
    parser.add_argument("--config", default=CONFIG_FILE)
    parser.add_argument("--front-end", choices=FRONT_ENDS, default=None)
    parser.add_argument("--out", help="also write one row per file to this CSV")
    args = parser.parse_args(argv)

    config = DetectorConfig.from_file(args.config)
    front = args.front_end or config.front_end
    threshold = config.threshold()
    score_fn = load_detector(config)
    rows = []
    for path in args.files:
        try:
            result = judge(load_audio(path), score_fn, threshold, front, source=path)
        except Exception as exc:  # one unreadable file must not stop the batch
            print(f"{path}: could not read ({exc})", file=sys.stderr)
            continue
        p = "-" if result.p_genuine is None else f"{result.p_genuine:.3f}"
        print(
            f"{result.verdict.upper():9s} P(genuine) {p} (threshold {threshold:.3f})  "
            f"{result.windows_agreeing}/{len(result.scored)} windows agree  {path}"
        )
        rows.append(result.summary())
    if args.out and rows:
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
