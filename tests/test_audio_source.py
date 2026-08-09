"""Smoke tests for the shared PcmFrameSource contract (Week 1, owner SK).

These are dependency-free: they exercise the format math and the protocol without
aiortc/av/fastapi, so they run in CI. The WebRTC server that implements the
contract (``rtc_server.py``) is validated separately (it needs the live-call deps).
"""

from collections.abc import AsyncIterator

from live_call.audio_source import (
    SAMPLE_WIDTH_BYTES,
    TARGET_CHANNELS,
    TARGET_SAMPLE_RATE,
    PcmFrame,
    PcmFrameSource,
)


def test_target_format_constants() -> None:
    assert TARGET_SAMPLE_RATE == 16_000
    assert TARGET_CHANNELS == 1
    assert SAMPLE_WIDTH_BYTES == 2


def test_frame_sample_and_duration_math() -> None:
    # 16 kHz mono s16 -> 32000 bytes == 1.0 s == 16000 samples.
    frame = PcmFrame(pcm=b"\x00\x00" * 16_000)
    assert frame.num_samples() == 16_000
    assert frame.duration_seconds() == 1.0


def test_empty_frame_is_zero_duration() -> None:
    assert PcmFrame(pcm=b"").duration_seconds() == 0.0


def test_protocol_is_runtime_checkable() -> None:
    class Dummy:
        async def frames(self) -> AsyncIterator[PcmFrame]:
            yield PcmFrame(pcm=b"\x00\x00")

    assert isinstance(Dummy(), PcmFrameSource)
