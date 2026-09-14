"""One monitored call: audio in, verdicts and alerts out (owner SK).

Every audio source the system accepts -- a WebRTC participant, a Twilio Media
Stream, a replayed demo clip -- becomes one :class:`CallSession`. The session runs
the source through ``StreamingScorer`` and ``VerdictEngine`` and publishes each step
to the :class:`Hub`, which every open dashboard listens to. Alert events go to an
``AlertDispatcher`` as well, which knows how to reach the receiver for that kind of
call.

Nothing here imports FastAPI or torch: the detector arrives as a score function and
the hub is plain asyncio, so the whole call lifecycle is tested in CI.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from live_call.verdict_engine import EngineConfig, Event, EventKind, State, VerdictEngine
from src.inference.streaming import StreamingScorer, WindowResult

#: How many past messages a dashboard that connects late is sent per session.
HISTORY = 400


class Hub:
    """Fan-out of JSON-able messages to every subscriber (the open dashboards)."""

    def __init__(self, queue_size: int = 1000) -> None:
        self._subscribers: set[asyncio.Queue] = set()
        self._queue_size = queue_size

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._queue_size)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    def publish(self, message: dict) -> None:
        for queue in list(self._subscribers):
            if queue.full():  # a stalled dashboard drops its oldest message, never blocks calls
                queue.get_nowait()
            queue.put_nowait(message)

    @property
    def subscribers(self) -> int:
        return len(self._subscribers)


@dataclass
class SessionInfo:
    """What the dashboard shows about a call before any audio is scored."""

    session_id: str
    kind: str  # "webrtc" | "twilio" | "replay"
    label: str
    started: float = field(default_factory=time.time)
    meta: dict = field(default_factory=dict)


_ids = itertools.count(1)


def new_session_id(kind: str) -> str:
    return f"{kind}-{int(time.time())}-{next(_ids)}"


class CallSession:
    """Drive one audio source through scoring and the verdict ladder."""

    def __init__(
        self,
        info: SessionInfo,
        score_fn: Callable,
        config: EngineConfig,
        hub: Hub,
        dispatcher=None,
        executor=None,
    ) -> None:
        self.info = info
        self.scorer = StreamingScorer(score_fn)
        self.engine = VerdictEngine(config)
        self.hub = hub
        self.dispatcher = dispatcher
        self.executor = executor
        self.history: list[dict] = []
        self.ended = False

    def _emit(self, message: dict) -> None:
        message = {"session": self.info.session_id, **message}
        self.history.append(message)
        del self.history[:-HISTORY]
        self.hub.publish(message)

    def snapshot(self) -> dict:
        s = self.engine.summary
        return {
            "session": self.info.session_id,
            "kind": self.info.kind,
            "label": self.info.label,
            "started": self.info.started,
            "meta": self.info.meta,
            "state": self.engine.state.value,
            "ended": self.ended,
            "scored_windows": s.scored_windows,
            "fake_leaning_windows": s.fake_leaning_windows,
            "threshold": self.engine.config.threshold,
        }

    async def _alert(self, event: Event) -> None:
        self._emit(
            {
                "type": "event",
                "kind": event.kind.value,
                "at": event.at_seconds,
                "state": event.state.value,
                "score": event.score,
                "message": event.message,
            }
        )
        if self.dispatcher is not None:
            try:
                await self.dispatcher.deliver(self.info, event)
            except Exception as exc:  # an alert channel failing must not end monitoring
                self._emit({"type": "alert_error", "kind": event.kind.value, "error": str(exc)})

    def _window_message(self, result: WindowResult) -> dict:
        return {
            "type": "window",
            "t": result.end_seconds,
            "level": round(result.level_dbfs, 1),
            "score": result.score,
            "smoothed": result.smoothed,
            "state": self.engine.state.value,
        }

    async def run(self, source) -> dict:
        """Monitor until the source ends; returns the final snapshot."""
        self._emit({"type": "session_start", **self.snapshot()})
        try:
            async for result in self.scorer.run(source, self.executor):
                smoothed = None if result.skipped else result.smoothed
                events = self.engine.update(result.end_seconds, smoothed)
                self._emit(self._window_message(result))
                for event in events:
                    await self._alert(event)
        finally:
            self.ended = True
            summary = self.engine.end_call()
            s = self.engine.summary
            self._emit(
                {
                    "type": "session_end",
                    "message": summary.message,
                    "peak_state": s.peak_state.value,
                    "duration": s.duration_seconds,
                    "scored_windows": s.scored_windows,
                    "fake_leaning_windows": s.fake_leaning_windows,
                    "worst_score": s.worst_score,
                }
            )
            if s.peak_state in (State.SUSPICIOUS, State.LIKELY_FAKE):
                await self._alert(summary)
        return self.snapshot()


class SessionRegistry:
    """Live and recently ended sessions, for dashboards that connect mid-call."""

    def __init__(self, keep_ended: int = 20) -> None:
        self._sessions: dict[str, CallSession] = {}
        self._keep_ended = keep_ended

    def add(self, session: CallSession) -> None:
        self._sessions[session.info.session_id] = session
        ended = [s for s in self._sessions.values() if s.ended]
        for stale in ended[: max(0, len(ended) - self._keep_ended)]:
            self._sessions.pop(stale.info.session_id, None)

    def get(self, session_id: str) -> CallSession | None:
        return self._sessions.get(session_id)

    def all(self) -> list[CallSession]:
        return list(self._sessions.values())

    def replay_history(self) -> list[dict]:
        """Every stored message, oldest session first, for a new dashboard."""
        return [m for s in self._sessions.values() for m in s.history]


__all__ = ["CallSession", "EventKind", "Hub", "SessionInfo", "SessionRegistry", "new_session_id"]
