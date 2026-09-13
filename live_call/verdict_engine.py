"""Call verdict state machine and escalation ladder (W8-T4 / W9-T4, owner SK).

``StreamingScorer`` produces a smoothed P(bonafide) every hop. A receiver cannot act
on a number that updates every two seconds, so this turns the stream into a verdict
that is **stable** and **escalates in steps**:

    LISTENING -> GENUINE <-> SUSPICIOUS -> LIKELY_FAKE

- ``LISTENING`` until enough speech has been scored to say anything.
- A window counts as *fake-leaning* when its smoothed score is below the operating
  threshold. ``suspicious_after`` consecutive fake-leaning windows move the call to
  ``SUSPICIOUS``; ``fake_after`` move it to ``LIKELY_FAKE``.
- Recovery needs ``recover_after`` consecutive windows above ``threshold + margin``.
  The margin is hysteresis: a score hovering at the threshold does not flap.
- Silent windows neither advance nor reset the counters. A pause is not evidence.

Each transition upward emits one escalation event, in plan order: a warning **beep**
into the receiver's leg on ``SUSPICIOUS``, a **repeat + SMS** on ``LIKELY_FAKE``, and
a **summary** when the call ends. ``alerts.py`` delivers them; this module only
decides.

**The threshold is not 0.5.** At 0.5 the clean adapter passes 47.2% of Tortoise
fakes (P-025). It comes from ``configs/threshold.yaml``, set from the DET curve at
the results freeze (W8-T5). Until then the engine refuses to start without one.

Stdlib only, so CI tests the whole ladder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class State(str, Enum):
    LISTENING = "listening"
    GENUINE = "genuine"
    SUSPICIOUS = "suspicious"
    LIKELY_FAKE = "likely_fake"


class EventKind(str, Enum):
    BEEP = "beep"  # warning tone into the receiver's leg
    SMS = "sms"  # repeat warning + text message
    SUMMARY = "summary"  # post-call summary


@dataclass(frozen=True)
class EngineConfig:
    """Operating point and ladder timing. Counts are in scored windows (one per hop)."""

    threshold: float
    margin: float = 0.05
    min_windows: int = 2
    suspicious_after: int = 2
    fake_after: int = 4
    recover_after: int = 3

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must be strictly between 0 and 1")
        if not 1 <= self.suspicious_after <= self.fake_after:
            raise ValueError("need 1 <= suspicious_after <= fake_after")


@dataclass(frozen=True)
class Event:
    kind: EventKind
    at_seconds: float
    state: State
    score: float | None
    message: str


@dataclass
class CallSummary:
    """What happened on the call, for the post-call alert and the dashboard."""

    duration_seconds: float = 0.0
    scored_windows: int = 0
    skipped_windows: int = 0
    fake_leaning_windows: int = 0
    worst_score: float | None = None
    peak_state: State = State.LISTENING
    timeline: list[tuple[float, State]] = field(default_factory=list)

    @property
    def fake_fraction(self) -> float:
        return self.fake_leaning_windows / self.scored_windows if self.scored_windows else 0.0


_RANK = {State.LISTENING: 0, State.GENUINE: 1, State.SUSPICIOUS: 2, State.LIKELY_FAKE: 3}


class VerdictEngine:
    """Feed it window results in call order; read ``state`` and collect events."""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.state = State.LISTENING
        self.summary = CallSummary(timeline=[(0.0, State.LISTENING)])
        self._low_run = 0
        self._high_run = 0
        self._ended = False

    def update(self, end_seconds: float, smoothed: float | None) -> list[Event]:
        """One window. ``smoothed`` is None for a silent (skipped) window."""
        if self._ended:
            raise RuntimeError("call already ended")
        s = self.summary
        s.duration_seconds = max(s.duration_seconds, end_seconds)
        if smoothed is None:
            s.skipped_windows += 1
            return []

        cfg = self.config
        s.scored_windows += 1
        s.worst_score = smoothed if s.worst_score is None else min(s.worst_score, smoothed)
        if smoothed < cfg.threshold:
            s.fake_leaning_windows += 1
            self._low_run += 1
            self._high_run = 0
        elif smoothed >= cfg.threshold + cfg.margin:
            self._high_run += 1
            self._low_run = 0
        # between threshold and threshold + margin: neither run advances

        new = self._next_state(s.scored_windows)
        return self._transition(new, end_seconds, smoothed) if new != self.state else []

    def _next_state(self, scored: int) -> State:
        cfg = self.config
        if self._low_run >= cfg.fake_after:
            return State.LIKELY_FAKE
        if self._low_run >= cfg.suspicious_after:
            # never step down from LIKELY_FAKE on a shorter run
            return max(self.state, State.SUSPICIOUS, key=_RANK.get)
        if self.state in (State.SUSPICIOUS, State.LIKELY_FAKE):
            return State.GENUINE if self._high_run >= cfg.recover_after else self.state
        if scored >= cfg.min_windows:
            return State.GENUINE
        return self.state

    def _transition(self, new: State, at: float, score: float) -> list[Event]:
        old, self.state = self.state, new
        self.summary.timeline.append((at, new))
        if _RANK[new] > _RANK[self.summary.peak_state]:
            self.summary.peak_state = new
        if new == State.SUSPICIOUS and _RANK[old] < _RANK[new]:
            return [Event(EventKind.BEEP, at, new, score, "Caution: this voice may be synthetic.")]
        if new == State.LIKELY_FAKE:
            return [
                Event(
                    EventKind.SMS,
                    at,
                    new,
                    score,
                    "Warning: this caller's voice is likely cloned. Do not share codes or money.",
                )
            ]
        return []

    def end_call(self) -> Event:
        """Close the call and return the summary event."""
        self._ended = True
        s = self.summary
        message = (
            f"Call of {s.duration_seconds:.0f} s: peak verdict {s.peak_state.value}, "
            f"{s.fake_leaning_windows}/{s.scored_windows} scored windows fake-leaning."
        )
        return Event(EventKind.SUMMARY, s.duration_seconds, self.state, s.worst_score, message)


def load_config(path: str = "configs/threshold.yaml") -> EngineConfig:
    """Read the frozen operating point. Refuses to invent one."""
    import yaml

    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(
            f"{path} not found: the operating threshold is set from the DET curve at the "
            "results freeze (W8-T5). Pass an explicit --threshold for a provisional run."
        )
    raw = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    known = {k: raw[k] for k in EngineConfig.__dataclass_fields__ if k in raw}
    return EngineConfig(**known)
