"""Pure logic behind the Streamlit demo: no Streamlit, so it can be tested.

Everything here delegates to the pipeline the results were measured with --
``src.data.preprocess`` for loudness, ``src.data.channel_sim`` for the telephone
line, ``src.inference.streaming`` for the 4 s / 2 s windowing and silence gate, and
``src.inference.predict`` for loading a checkpoint and turning it into a scorer.
Nothing is re-implemented here: a demo that scores audio differently from the
evaluation is a demo that proves nothing.

The live-call system in ``live_call/`` is deliberately untouched by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
#: Anything shorter than one window is padded rather than refused.
WINDOW_SECONDS = 4.0
HOP_SECONDS = 2.0


def repo_root() -> Path:
    """The repository root, found from this file rather than hard-coded."""
    return Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ModelChoice:
    """One selectable detector: where it lives and what it is."""

    key: str
    label: str
    path: Path
    blurb: str

    @property
    def available(self) -> bool:
        return self.path.is_file()


def model_choices(env_checkpoint: str | None = None) -> list[ModelChoice]:
    """The detectors offered in the sidebar, best first.

    ``env_checkpoint`` (``DFD_CHECKPOINT``) is the adapter the live system deploys.
    It is normally outside the repository because it is 365 MB, so it is offered
    first when present and silently skipped when not.
    """
    root = repo_root()
    choices: list[ModelChoice] = []
    if env_checkpoint:
        choices.append(
            ModelChoice(
                "deployed",
                "S2 LoRA, normalised + RVC, channel-matched (deployed)",
                Path(env_checkpoint).expanduser(),
                "The adapter the live demo runs. Trained on level-normalised "
                "code-mixed speech with XTTS and RVC attacks, over a G.711 line.",
            )
        )
    choices += [
        ModelChoice(
            "s2_channel",
            "S2 LoRA, channel-matched",
            root / "checkpoints" / "lora_codemix_channel" / "best.pt",
            "Adapted on code-mixed speech carried over a simulated phone line. "
            "This is the one to use when 'simulate phone line' is on.",
        ),
        ModelChoice(
            "s2_clean",
            "S2 LoRA, clean-trained",
            root / "checkpoints" / "lora_codemix" / "best.pt",
            "Adapted on clean code-mixed speech. Strong on clean audio and it "
            "falls apart over a telephone channel -- switch the line on and see.",
        ),
        ModelChoice(
            "s1",
            "S1 baseline, English-only",
            root / "checkpoints" / "baseline" / "best.pt",
            "Trained on ASVspoof 2019 LA (English) only. Near perfect on English "
            "and worse than chance on Hindi-English over a phone line.",
        ),
    ]
    return choices


def choice_by_key(key: str, env_checkpoint: str | None = None) -> ModelChoice:
    for choice in model_choices(env_checkpoint):
        if choice.key == key:
            return choice
    raise KeyError(f"unknown model {key!r}")


def missing_checkpoint_message(choice: ModelChoice) -> str:
    """What to tell the user when a checkpoint is not on this machine."""
    return (
        f"{choice.label} is not on this machine.\n\n"
        f"Expected at: `{choice.path}`\n\n"
        "Checkpoints are Git LFS pointers in this repository: run `git lfs pull`. "
        "The deployed adapter lives outside the repository -- set `DFD_CHECKPOINT` "
        "in `.env` to its path."
    )


# --------------------------------------------------------------------- audio


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Read any soundfile-readable file as float32 mono at its own rate."""
    from src.utils.audio_utils import load_wav

    return load_wav(str(path))


def prepare(audio: np.ndarray, sr: int, target_dbfs: float = -23.0) -> np.ndarray:
    """Resample to 16 kHz mono and level-normalise, as the corpora were prepared.

    Deliberately *not* the full ``preprocess_signal``: that also runs VAD trimming
    and splits into training segments, which would cut a demo clip into pieces.
    The two steps that matter for scoring -- rate and level -- are the repo's own.
    """
    from src.utils.audio_utils import resample, rms_normalize, to_mono

    x = to_mono(np.asarray(audio, dtype=np.float32))
    if sr != SAMPLE_RATE:
        x = resample(x, sr, SAMPLE_RATE)
    return rms_normalize(x, target_dbfs)


def apply_phone_line(audio: np.ndarray, codec: str = "g711", snr_db: float = 20.0) -> np.ndarray:
    """Push 16 kHz audio through the simulated telephone channel and back."""
    from src.data.channel_sim import ChannelConfig, simulate_channel

    return simulate_channel(
        audio, ChannelConfig(codec=codec, snr_db=snr_db, in_sr=SAMPLE_RATE, seed=0)
    )


# -------------------------------------------------------------------- scoring


@dataclass(frozen=True)
class Window:
    """One scored 4 s window."""

    end_seconds: float
    score: float | None  # None when the window was below the silence gate
    level_dbfs: float

    @property
    def skipped(self) -> bool:
        return self.score is None


@dataclass(frozen=True)
class Result:
    """What the app shows for one clip."""

    score: float | None  # the clip verdict score: the lowest scored window
    verdict: str  # "bonafide" | "spoof" | "no speech"
    windows: list[Window]
    seconds: float
    threshold: float

    @property
    def is_clone(self) -> bool:
        return self.verdict == "spoof"

    @property
    def scored_windows(self) -> list[Window]:
        return [w for w in self.windows if not w.skipped]


def score_audio(audio: np.ndarray, score_fn, threshold: float) -> Result:
    """Window, gate, score and decide -- the same path the evaluation uses.

    The windows are scored exactly once and both the verdict and the timeline are
    derived from them. Calling ``predict_audio`` as well would double the forward
    passes, which on a CPU is the difference between a demo that feels instant and
    one that stalls in front of an examiner.
    """
    from src.inference.predict import Threshold, verdict_for
    from src.inference.streaming import StreamingScorer

    operating = Threshold(float(threshold), "app", provisional=True)
    samples = np.asarray(audio, dtype=np.float32)
    window_samples = int(WINDOW_SECONDS * SAMPLE_RATE)
    if 0 < samples.size < window_samples:  # short clip: pad so it yields one window
        samples = np.pad(samples, (0, window_samples - samples.size))

    windows = [
        Window(r.end_seconds, r.score, r.level_dbfs)
        for r in StreamingScorer(score_fn).push(samples)
    ]
    scored = [w.score for w in windows if w.score is not None]
    # The clip verdict is the worst window, matching src.inference.predict: one
    # confidently synthetic stretch should not be averaged away by a calm one.
    score = min(scored) if scored else None
    return Result(
        score=score,
        verdict=verdict_for(score, operating),
        windows=windows,
        seconds=float(np.asarray(audio).size) / SAMPLE_RATE,
        threshold=float(threshold),
    )


def load_threshold() -> tuple[float, str]:
    """The project's operating point and where it came from."""
    from src.inference.predict import resolve_threshold

    root = repo_root()
    threshold = resolve_threshold(threshold_file=str(root / "configs" / "threshold.yaml"))
    return threshold.value, threshold.source


# ----------------------------------------------------------------- demo clips


@dataclass(frozen=True)
class DemoClip:
    path: Path
    title: str
    truth: str  # "real" | "cloned" | "unknown"

    @property
    def is_clone(self) -> bool:
        return self.truth == "cloned"


#: Filename stem -> how to present it. Anything else found in the folder is listed
#: with an unknown ground truth rather than ignored.
KNOWN_CLIPS: dict[str, tuple[str, str]] = {
    "genuine": ("Genuine caller", "real"),
    "rvc_genuine": ("Genuine caller (RVC source speaker)", "real"),
    "xtts_clone": ("XTTS-v2 cloned voice", "cloned"),
    "rvc_conversion": ("RVC voice conversion", "cloned"),
    "tortoise_clone": ("Tortoise clone (unseen tool)", "cloned"),
}

AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".m4a", ".ogg")


def demo_clips(folder: str | Path | None = None) -> list[DemoClip]:
    """Clips found in ``app/demo_clips``, real ones first so the demo opens green."""
    directory = Path(folder) if folder else repo_root() / "app" / "demo_clips"
    if not directory.is_dir():
        return []
    found = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        title, truth = KNOWN_CLIPS.get(path.stem, (path.stem.replace("_", " "), "unknown"))
        found.append(DemoClip(path, title, truth))
    order = {"real": 0, "cloned": 1, "unknown": 2}
    return sorted(found, key=lambda c: (order[c.truth], c.title))


def verdict_card(result: Result) -> tuple[str, str, str]:
    """``(headline, colour, detail)`` for the verdict box."""
    if result.score is None:
        return ("NO SPEECH", "gray", "No window passed the silence gate.")
    confidence = result.score if not result.is_clone else 1.0 - result.score
    detail = (
        f"P(real) = {result.score:.3f} against a threshold of {result.threshold:.3f} "
        f"· {len(result.scored_windows)} of {len(result.windows)} windows scored "
        f"· {confidence:.1%} confident"
    )
    if result.is_clone:
        return ("CLONED VOICE", "red", detail)
    return ("REAL VOICE", "green", detail)
