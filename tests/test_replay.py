"""File replay drives the real live-call path (owner SK)."""

from __future__ import annotations

import asyncio

import numpy as np

from live_call.audio_source import PcmFrameSource
from live_call.replay import FileSource, replay
from live_call.verdict_engine import EngineConfig, EventKind, State

SR = 16_000
CFG = EngineConfig(threshold=0.6)


def _speech(seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.1 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)


def test_file_source_satisfies_the_shared_protocol() -> None:
    assert isinstance(FileSource(_speech(1.0)), PcmFrameSource)


def test_file_source_yields_20ms_frames() -> None:
    async def collect():
        return [f async for f in FileSource(_speech(1.0)).frames()]

    frames = asyncio.run(collect())
    assert len(frames) == 50 and all(f.num_samples() == 320 for f in frames)


def test_genuine_call_ends_genuine_with_no_alerts() -> None:
    result = asyncio.run(replay(FileSource(_speech(20.0)), lambda w: 0.95, CFG))
    assert result.engine.state == State.GENUINE
    assert [e.kind for e in result.events] == [EventKind.SUMMARY]


def test_cloned_call_escalates_beep_then_sms() -> None:
    result = asyncio.run(replay(FileSource(_speech(20.0)), lambda w: 0.05, CFG))
    kinds = [e.kind for e in result.events]
    assert kinds == [EventKind.BEEP, EventKind.SMS, EventKind.SUMMARY]
    assert result.engine.summary.peak_state == State.LIKELY_FAKE


def test_callback_sees_every_window() -> None:
    seen = []
    asyncio.run(
        replay(FileSource(_speech(12.0)), lambda w: 0.9, CFG, lambda r, s, e: seen.append(r))
    )
    assert [r.end_seconds for r in seen] == [4.0, 6.0, 8.0, 10.0, 12.0]
