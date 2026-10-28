"""Several detectors listening to one call, side by side (the three-lane demo view).

The live call path serves one model: the server's primary detector from
``DFD_CHECKPOINT``. The demo's centrepiece asks a different question -- what would the
English-only detector, the deployed adapter and the final recipe each have said about
*this* call? -- so this module scores one call through several detectors at once.

Every lane runs the live path unchanged: its own :class:`StreamingScorer` (4 s windows
every 2 s, silence gate, -23 dBFS normalisation, EMA 0.5) and its own
:class:`VerdictEngine` with that model's frozen operating point and ladder. Lanes differ
only in the checkpoint and the threshold file, both named in the environment::

    DFD_LANES=S1 (English only)=checkpoints/baseline/best.pt=configs/threshold.yaml,
              Final recipe=C:/dfdata/checkpoints/final_s1234_best.pt=configs/threshold_final.yaml

Entries are comma separated, each ``name=checkpoint=threshold_file`` (Windows paths
contain no ``=``, so the split is unambiguous). A lane whose checkpoint is the primary
detector's reuses the primary model instead of loading a second copy.

Nothing here imports torch or FastAPI; models arrive through :class:`DetectorService`,
so the whole lane machinery is tested in CI with stand-in score functions.
"""

from __future__ import annotations

import asyncio
import itertools
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from live_call.detector import DetectorService, DetectorSettings
from live_call.replay import FileSource
from live_call.verdict_engine import EngineConfig, State, VerdictEngine
from src.inference.streaming import StreamingScorer, pcm16_to_float

#: Lane colours, the same as the paper figures (S1 grey, deployed orange, final green).
ROLE_COLOURS = {
    "s1": "#8a94a6",
    "deployed": "#eb6834",
    "final": "#1baf7a",
    "ssl": "#8b6cff",
    "other": "#4f8cff",
}
_RANK = {State.LISTENING: 0, State.GENUINE: 1, State.SUSPICIOUS: 2, State.LIKELY_FAKE: 3}


class LaneSpecError(ValueError):
    """``DFD_LANES`` could not be read; the message says which entry and why."""


@dataclass(frozen=True)
class LaneSpec:
    name: str
    checkpoint: str
    threshold_file: str


def parse_lanes(spec: str) -> list[LaneSpec]:
    """Read a ``DFD_LANES`` value. An empty value means no extra lanes."""
    lanes: list[LaneSpec] = []
    for index, entry in enumerate((e.strip() for e in spec.split(",")), start=1):
        if not entry:
            continue
        parts = [p.strip() for p in entry.split("=")]
        if len(parts) != 3 or not all(parts):
            raise LaneSpecError(
                f"DFD_LANES entry {index} ({entry!r}) must be name=checkpoint=threshold_file"
            )
        name, checkpoint, threshold_file = parts
        if any(lane.name == name for lane in lanes):
            raise LaneSpecError(f"DFD_LANES names lane {name!r} twice")
        lanes.append(LaneSpec(name, checkpoint, threshold_file))
    return lanes


def lane_role(name: str, checkpoint: str = "") -> str:
    """Which of the project's models a lane shows, for its colour and the rehearsal file."""
    text = f"{name} {Path(checkpoint).name}".lower()
    if "ssl" in text or "aasist" in text:
        return "ssl"
    if "final" in text:
        return "final"
    if "deployed" in text or "lora_norm_rvc" in text:
        return "deployed"
    if "s1" in text.split() or "english" in text or "baseline" in text:
        return "s1"
    return "other"


def _same_file(a: str, b: str) -> bool:
    if not a or not b:
        return False
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return a == b


class _SharedModel(DetectorService):
    """A lane on the primary's checkpoint: the primary's loaded model, its own threshold."""

    def __init__(self, settings: DetectorSettings, primary: DetectorService) -> None:
        super().__init__(settings)
        self._primary = primary

    @property
    def loaded(self) -> bool:
        return self._primary.loaded

    def score_fn(self):
        return self._primary.score_fn()


@dataclass
class Lane:
    """One detector in the lanes view."""

    name: str
    detector: DetectorService
    role: str
    colour: str

    @property
    def config(self) -> EngineConfig:
        return self.detector.config()

    def describe(self) -> dict:
        status = self.detector.status()
        out = {
            "name": self.name,
            "role": self.role,
            "colour": self.colour,
            "checkpoint": status["checkpoint"],
            "checkpoint_found": status["checkpoint_found"],
            "model_loaded": status["model_loaded"],
            "threshold": status["threshold"],
            "threshold_source": status.get("threshold_source"),
            "error": status["threshold_error"],
        }
        if status["threshold"] is not None:
            cfg = self.config
            out["ladder"] = {"caution_after": cfg.suspicious_after, "likely_after": cfg.fake_after}
        return out


def build_lanes(specs: list[LaneSpec], primary: DetectorService) -> list[Lane]:
    lanes = []
    for spec in specs:
        settings = DetectorSettings(spec.checkpoint, spec.threshold_file, primary.settings.device)
        if _same_file(spec.checkpoint, primary.settings.checkpoint):
            detector: DetectorService = _SharedModel(settings, primary)
        else:
            detector = DetectorService(settings)
        role = lane_role(spec.name, spec.checkpoint)
        lanes.append(Lane(spec.name, detector, role, ROLE_COLOURS[role]))
    return lanes


def primary_lane(primary: DetectorService, name: str = "Live model") -> Lane:
    """The server's own detector as a lane: what the live call path uses."""
    role = lane_role(name, primary.settings.checkpoint)
    return Lane(name, primary, role, ROLE_COLOURS[role])


@dataclass
class LaneSet:
    """The lanes the server was configured with, or why it could not read them."""

    lanes: list[Lane] = field(default_factory=list)
    error: str | None = None

    @classmethod
    def from_env(cls, primary: DetectorService, value: str | None = None) -> LaneSet:
        value = os.environ.get("DFD_LANES", "") if value is None else value
        try:
            specs = parse_lanes(value)
        except LaneSpecError as exc:
            return cls([], str(exc))
        if not specs:  # no lanes configured: the live model alone, so the page still works
            return cls([primary_lane(primary)])
        return cls(build_lanes(specs, primary))

    def problems(self) -> list[str]:
        """Configuration errors that would stop a lanes replay, one line each."""
        if self.error:
            return [self.error]
        out = []
        for lane in self.lanes:
            status = lane.detector.status()
            if not status["checkpoint_found"] and not lane.detector.loaded:
                out.append(
                    f"lane {lane.name!r}: checkpoint not found ({lane.detector.settings.checkpoint})"
                )
            if status["threshold_error"]:
                out.append(f"lane {lane.name!r}: {status['threshold_error']}")
        return out


# --------------------------------------------------------------------------- #
# Scoring one call through every lane
# --------------------------------------------------------------------------- #
class ClockedFileSource(FileSource):
    """A file played at wall-clock speed that never falls behind the clock.

    ``FileSource(realtime=True)`` sleeps 20 ms after every frame, so any time spent
    scoring adds up and the scores drift away from the audio a browser is playing.
    This source schedules frame *n* at ``start + n * 20 ms`` instead: after a slow
    window it catches up, and the window times stay locked to the audio.
    """

    async def frames(self):
        from live_call.audio_source import PcmFrame
        from live_call.replay import FRAME_SECONDS
        from src.inference.streaming import SAMPLE_RATE

        step = int(FRAME_SECONDS * SAMPLE_RATE) * 2
        start = time.monotonic()
        for n, offset in enumerate(range(0, len(self._pcm), step)):
            due = start + n * FRAME_SECONDS
            delay = due - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            yield PcmFrame(pcm=self._pcm[offset : offset + step])


@dataclass
class LaneResult:
    """Everything one lane said about the call."""

    lane: Lane
    engine: VerdictEngine
    windows: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    first_caution_s: float | None = None
    first_likely_s: float | None = None

    def summary(self) -> dict:
        s = self.engine.summary
        return {
            "name": self.lane.name,
            "role": self.lane.role,
            "colour": self.lane.colour,
            "threshold": self.engine.config.threshold,
            "peak_state": s.peak_state.value,
            "final_state": self.engine.state.value,
            "first_caution_s": self.first_caution_s,
            "first_likely_s": self.first_likely_s,
            "scored_windows": s.scored_windows,
            "fake_leaning_windows": s.fake_leaning_windows,
            "worst_score": s.worst_score,
            "duration": s.duration_seconds,
        }


_runs = itertools.count(1)


def new_run_id() -> str:
    return f"lanes-{int(time.time())}-{next(_runs)}"


async def score_lanes(
    audio: np.ndarray,
    lanes: list[Lane],
    executor=None,
    realtime: bool = False,
    emit: Callable[[dict], None] | None = None,
) -> list[LaneResult]:
    """Play ``audio`` once and score it through every lane; returns one result per lane.

    ``emit`` receives a message per scored window and per verdict change, tagged with
    the lane name, as the call goes. Model calls run on ``executor`` (one worker: torch
    already uses every core for a single forward pass).
    """
    loop = asyncio.get_running_loop()
    # loading a checkpoint takes seconds: never on the event loop
    score_fns = await loop.run_in_executor(
        executor, lambda: [lane.detector.score_fn() for lane in lanes]
    )
    scorers = [StreamingScorer(fn) for fn in score_fns]
    results = [LaneResult(lane, VerdictEngine(lane.config)) for lane in lanes]
    source = ClockedFileSource(audio) if realtime else FileSource(audio)
    send = emit or (lambda _m: None)

    def push(samples: np.ndarray) -> list[list]:
        return [list(scorer.push(samples)) for scorer in scorers]

    async for frame in source.frames():
        samples = pcm16_to_float(frame.pcm)
        per_lane = await loop.run_in_executor(executor, push, samples)
        for index, (result, windows) in enumerate(zip(results, per_lane, strict=True)):
            for w in windows:
                before = result.engine.state
                events = result.engine.update(w.end_seconds, None if w.skipped else w.smoothed)
                state = result.engine.state
                if _RANK[state] >= _RANK[State.SUSPICIOUS] and result.first_caution_s is None:
                    result.first_caution_s = w.end_seconds
                if state is State.LIKELY_FAKE and result.first_likely_s is None:
                    result.first_likely_s = w.end_seconds
                message = {
                    "type": "lane_window",
                    "lane": result.lane.name,
                    "lane_index": index,
                    "t": w.end_seconds,
                    "score": w.score,
                    "smoothed": w.smoothed,
                    "level": round(w.level_dbfs, 1),
                    "state": state.value,
                    "changed": state is not before,
                }
                result.windows.append(message)
                send(message)
                for event in events:
                    record = {
                        "type": "lane_event",
                        "lane": result.lane.name,
                        "lane_index": index,
                        "kind": event.kind.value,
                        "at": event.at_seconds,
                        "state": event.state.value,
                        "message": event.message,
                    }
                    result.events.append(record)
                    send(record)
    for result in results:
        result.engine.end_call()
    return results


__all__ = [
    "ClockedFileSource",
    "Lane",
    "LaneResult",
    "LaneSet",
    "LaneSpec",
    "LaneSpecError",
    "ROLE_COLOURS",
    "build_lanes",
    "lane_role",
    "new_run_id",
    "parse_lanes",
    "primary_lane",
    "score_lanes",
]
