"""Replay an audio file as a live call through the exact live-call path (owner SK).

The WebRTC harness needs two browsers and a microphone; Twilio needs an activated
account. Neither is repeatable, and neither can be run the night before a viva to
check that nothing broke. This plays a file as a call instead: the audio is cut into
20 ms :class:`PcmFrame` chunks, handed to ``StreamingScorer.run`` through the same
``PcmFrameSource`` protocol both telephony backends implement, and every window goes
through the same ``VerdictEngine``. What a receiver would see, second by second, is
printed as a timeline, followed by the post-call summary.

It is also how the demo audio arsenal (W8-T5) gets verified: every clip in it should
replay to the verdict it is meant to demonstrate.

Run::

    python -m live_call.replay --checkpoint checkpoints/lora_codemix_channel/best.pt \\
        --threshold 0.97 call.wav

``--realtime`` paces frames at wall-clock speed, to watch it like a call.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass

import numpy as np

from live_call.audio_source import PcmFrame
from live_call.verdict_engine import EngineConfig, Event, VerdictEngine
from src.inference.streaming import SAMPLE_RATE, StreamingScorer, WindowResult

FRAME_SECONDS = 0.02


class FileSource:
    """A ``PcmFrameSource`` over 16 kHz float audio already in memory."""

    def __init__(self, audio: np.ndarray, realtime: bool = False) -> None:
        clipped = np.clip(audio, -1.0, 32767 / 32768)
        self._pcm = (clipped * 32768.0).astype("<i2").tobytes()
        self._realtime = realtime

    async def frames(self) -> AsyncIterator[PcmFrame]:
        step = int(FRAME_SECONDS * SAMPLE_RATE) * 2  # bytes per 20 ms of s16 mono
        for start in range(0, len(self._pcm), step):
            yield PcmFrame(pcm=self._pcm[start : start + step])
            if self._realtime:
                await asyncio.sleep(FRAME_SECONDS)


@dataclass
class ReplayResult:
    windows: list[WindowResult]
    events: list[Event]
    engine: VerdictEngine


async def replay(source, score_fn, config: EngineConfig, on_window=None) -> ReplayResult:
    """Drive a source through scoring and the verdict engine to the end of the call."""
    scorer = StreamingScorer(score_fn)
    engine = VerdictEngine(config)
    windows, events = [], []
    async for result in scorer.run(source):
        new = engine.update(result.end_seconds, result.smoothed if not result.skipped else None)
        windows.append(result)
        events += new
        if on_window:
            on_window(result, engine.state, new)
    events.append(engine.end_call())
    return ReplayResult(windows, events, engine)


def _print_window(result: WindowResult, state, new_events) -> None:
    if result.skipped:
        line = f"{result.end_seconds:6.1f}s  (silence, {result.level_dbfs:6.1f} dBFS)"
    else:
        line = (
            f"{result.end_seconds:6.1f}s  score {result.score:.4f}  smoothed {result.smoothed:.4f}"
        )
    print(f"{line}  -> {state.value}")
    for event in new_events:
        print(f"         ** {event.kind.value.upper()}: {event.message}")


def main() -> None:
    """CLI entry point."""
    import argparse

    from live_call.verdict_engine import load_config
    from src.inference.predict import load_detector, make_score_fn
    from src.utils.audio_utils import load_wav

    parser = argparse.ArgumentParser(description="Replay a file as a live call")
    parser.add_argument("file")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threshold", type=float, default=None, help="provisional operating point")
    parser.add_argument("--threshold-file", default="configs/threshold.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--realtime", action="store_true")
    args = parser.parse_args()

    if args.threshold is not None:
        config = EngineConfig(threshold=args.threshold)
        print(f"PROVISIONAL threshold {args.threshold} (not the frozen operating point)")
    else:
        config = load_config(args.threshold_file)
    model, device = load_detector(args.checkpoint, device=args.device)
    audio, _ = load_wav(args.file, target_sr=SAMPLE_RATE)
    result = asyncio.run(
        replay(
            FileSource(audio, args.realtime), make_score_fn(model, device), config, _print_window
        )
    )
    print(result.events[-1].message)


if __name__ == "__main__":
    main()
