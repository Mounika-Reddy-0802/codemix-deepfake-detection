"""Shared audio-source contract for the live-call pipeline.

Both telephony backends produce the *same* thing for the detector:

- the free **WebRTC harness** (``webrtc_harness/rtc_server.py``, Weeks 1-5), and
- the **Twilio media handler** (``media_handler.py``, from Week 6),

each expose a stream of **16 kHz mono signed-16-bit PCM** frames via the
``PcmFrameSource`` protocol below. Streaming inference consumes this interface and
never needs to know which backend produced the audio -- so all streaming/alert
code written against the free harness ports to Twilio unchanged.

This module is intentionally dependency-free (stdlib only) so it imports cheaply
in CI and anywhere else.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

#: The one canonical audio format every source emits.
TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2  # signed 16-bit little-endian


@dataclass(frozen=True)
class PcmFrame:
    """A chunk of 16 kHz mono signed-16-bit little-endian PCM audio."""

    pcm: bytes
    sample_rate: int = TARGET_SAMPLE_RATE
    channels: int = TARGET_CHANNELS

    def num_samples(self) -> int:
        """Number of audio samples in this frame."""
        return len(self.pcm) // (SAMPLE_WIDTH_BYTES * self.channels)

    def duration_seconds(self) -> float:
        """Wall-clock duration this frame represents."""
        if self.sample_rate == 0:
            return 0.0
        return self.num_samples() / self.sample_rate


@runtime_checkable
class PcmFrameSource(Protocol):
    """Anything that yields 16 kHz mono PCM frames.

    Implemented by the WebRTC harness now and by the Twilio media handler in
    Week 6. Downstream code depends only on this protocol.
    """

    def frames(self) -> AsyncIterator[PcmFrame]:
        """Yield PCM frames as they arrive, until the source ends."""
        ...
