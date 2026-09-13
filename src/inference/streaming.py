"""Rolling-window streaming inference over live call audio (W5-T5, owner SK).

A call arrives as a stream of short PCM frames (20 ms from WebRTC, 20 ms of 8 kHz
mu-law from Twilio). The detector was trained on 4-second clips, so the stream is
cut into **4 s windows every 2 s** and each window is scored on its own. Three
things make that usable on a real call rather than on a benchmark:

- **Silence is not scored.** A window whose level is below ``min_rms_dbfs`` carries
  no speech, and a detector asked about silence returns noise. Those windows are
  reported as skipped, never as a verdict.
- **Each window is level-normalised to -23 dBFS** before scoring, exactly as every
  training clip was (``normalise_bundle``, P-026). A quiet phone line would
  otherwise hand the model a level it never saw.
- **Scores are smoothed** with an exponential moving average, so one odd window
  cannot flip the verdict. The verdict itself is decided downstream, in
  ``live_call/verdict_engine.py``.

Everything here is numpy only. The model enters as a plain ``score_fn`` callable,
so the windowing is tested in CI without torch, and the same code serves the
WebRTC harness, the Twilio handler and file replay.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16_000
WINDOW_SECONDS = 4.0
HOP_SECONDS = 2.0
TARGET_DBFS = -23.0
MIN_RMS_DBFS = -50.0
EMA_ALPHA = 0.5

#: Takes one 16 kHz float32 window and returns P(bonafide) in [0, 1].
ScoreFn = Callable[[np.ndarray], float]


def pcm16_to_float(pcm: bytes) -> np.ndarray:
    """Signed 16-bit little-endian PCM bytes to float32 in [-1, 1)."""
    usable = len(pcm) - (len(pcm) % 2)
    return np.frombuffer(pcm[:usable], dtype="<i2").astype(np.float32) / 32768.0


def rms_dbfs(window: np.ndarray) -> float:
    """RMS level in dBFS; -inf for digital silence."""
    if window.size == 0:
        return float("-inf")
    value = float(np.sqrt(np.mean(np.square(window, dtype=np.float64))))
    return 20.0 * np.log10(value) if value > 0 else float("-inf")


def normalise(window: np.ndarray, target_dbfs: float = TARGET_DBFS) -> np.ndarray:
    """Scale to ``target_dbfs`` RMS, never past full scale (same rule as training)."""
    level = rms_dbfs(window)
    if not np.isfinite(level):
        return window.astype(np.float32)
    gain = 10.0 ** ((target_dbfs - level) / 20.0)
    peak = float(np.max(np.abs(window))) * gain
    if peak > 0.999:
        gain *= 0.999 / peak
    return (window * gain).astype(np.float32)


class RollingWindower:
    """Accumulate samples and emit fixed-length overlapping windows."""

    def __init__(
        self,
        window_seconds: float = WINDOW_SECONDS,
        hop_seconds: float = HOP_SECONDS,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        if not 0 < hop_seconds <= window_seconds:
            raise ValueError("hop must be positive and no longer than the window")
        self.window = int(round(window_seconds * sample_rate))
        self.hop = int(round(hop_seconds * sample_rate))
        self.sample_rate = sample_rate
        self._buffer = np.zeros(0, dtype=np.float32)
        self._consumed = 0  # samples dropped off the front so far

    def push(self, samples: np.ndarray) -> Iterator[tuple[float, np.ndarray]]:
        """Add samples; yield ``(end_time_seconds, window)`` for every full window."""
        self._buffer = np.concatenate([self._buffer, samples.astype(np.float32, copy=False)])
        while self._buffer.size >= self.window:
            end = (self._consumed + self.window) / self.sample_rate
            yield end, self._buffer[: self.window].copy()
            self._buffer = self._buffer[self.hop :]
            self._consumed += self.hop


class EmaSmoother:
    """Exponential moving average; the first value passes through unchanged."""

    def __init__(self, alpha: float = EMA_ALPHA) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.value: float | None = None

    def update(self, x: float) -> float:
        self.value = x if self.value is None else self.alpha * x + (1 - self.alpha) * self.value
        return self.value


@dataclass(frozen=True)
class WindowResult:
    """One scored (or skipped) window of the call."""

    end_seconds: float
    level_dbfs: float
    score: float | None  # P(bonafide); None when the window was silent
    smoothed: float | None  # EMA of the scored windows so far

    @property
    def skipped(self) -> bool:
        return self.score is None


class StreamingScorer:
    """Windows, silence gate, normalisation, scoring and smoothing, in one place."""

    def __init__(
        self,
        score_fn: ScoreFn,
        window_seconds: float = WINDOW_SECONDS,
        hop_seconds: float = HOP_SECONDS,
        min_rms_dbfs: float = MIN_RMS_DBFS,
        ema_alpha: float = EMA_ALPHA,
        target_dbfs: float = TARGET_DBFS,
    ) -> None:
        self.score_fn = score_fn
        self.windower = RollingWindower(window_seconds, hop_seconds)
        self.smoother = EmaSmoother(ema_alpha)
        self.min_rms_dbfs = min_rms_dbfs
        self.target_dbfs = target_dbfs

    def push(self, samples: np.ndarray) -> Iterator[WindowResult]:
        """Feed float32 16 kHz samples; yield a result per completed window."""
        for end, window in self.windower.push(samples):
            level = rms_dbfs(window)
            if level < self.min_rms_dbfs:
                yield WindowResult(end, level, None, self.smoother.value)
                continue
            score = float(self.score_fn(normalise(window, self.target_dbfs)))
            yield WindowResult(end, level, score, self.smoother.update(score))

    def push_pcm(self, pcm: bytes) -> Iterator[WindowResult]:
        """Feed 16 kHz mono s16 PCM bytes."""
        yield from self.push(pcm16_to_float(pcm))

    async def run(self, source) -> AsyncIterator[WindowResult]:
        """Consume any ``PcmFrameSource`` (WebRTC harness, Twilio, replay)."""
        async for frame in source.frames():
            if frame.sample_rate != SAMPLE_RATE:
                raise ValueError(f"expected {SAMPLE_RATE} Hz frames, got {frame.sample_rate}")
            for result in self.push_pcm(frame.pcm):
                yield result
