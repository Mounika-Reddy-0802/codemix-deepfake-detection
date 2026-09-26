"""The telephone exchange behind the two-handset demo (owner SK).

The Week-10 demo put both parties on one page, so an evaluator could not tell the
caller from the receiver. This module is the missing piece: a small exchange that
two separate browsers connect to, one as the **caller** handset and one as the
**receiver** handset, with the call states a real phone has.

::

    caller                exchange                 receiver
      |  join(caller) ------->|<------- join(receiver)  |
      |  dial --------------->|  incoming ------------->|   (receiver's phone rings)
      |<------- ringing ------|<------- accept ---------|
      |<===== connected =====>|<===== connected =======>|   (audio flows, timer runs)
      |                       |  verdict --------------->|   (receiver only)
      |  hangup ------------->|  ended ----------------->|

States are ``idle -> ringing -> connected -> ended``, plus ``ended`` straight from
``ringing`` when the receiver declines or the caller gives up. Only the receiver is
sent verdicts: a real fraud warning never tells the caller they were detected, and
the demo is more convincing when the caller's screen shows nothing.

No FastAPI and no WebRTC here. A party is a role, a display identity and a ``send``
callable, so the whole thing is exercised in CI; :mod:`live_call.server` wires
``send`` to a WebSocket and the media to the existing ``/offer`` path.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

#: Sends one JSON-able message to a handset. Returns nothing, must not raise.
Send = Callable[[dict], None]


class CallState(str, Enum):
    IDLE = "idle"
    RINGING = "ringing"
    CONNECTED = "connected"
    ENDED = "ended"


CALLER = "caller"
RECEIVER = "receiver"
ROLES = (CALLER, RECEIVER)


@dataclass
class Party:
    """One handset attached to a line."""

    role: str
    name: str
    number: str
    send: Send
    pc_id: str | None = None  # its WebRTC peer connection, for renegotiation

    def identity(self) -> dict:
        return {"role": self.role, "name": self.name, "number": self.number}


@dataclass
class Line:
    """One room: at most one caller handset and one receiver handset, and their call."""

    room: str
    parties: dict[str, Party] = field(default_factory=dict)
    state: CallState = CallState.IDLE
    call_id: str | None = None
    started: float | None = None  # when the caller dialled
    connected: float | None = None  # when the receiver accepted
    ended: float | None = None
    end_reason: str | None = None
    session_id: str | None = None  # detection session for the caller's audio
    verdict: str = "listening"

    def snapshot(self) -> dict:
        return {
            "room": self.room,
            "state": self.state.value,
            "call_id": self.call_id,
            "started": self.started,
            "connected": self.connected,
            "ended": self.ended,
            "end_reason": self.end_reason,
            "session": self.session_id,
            "verdict": self.verdict,
            "parties": {role: p.identity() for role, p in self.parties.items()},
        }


class Exchange:
    """Every line, keyed by room name."""

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._lines: dict[str, Line] = {}
        self._clock = clock
        self._counter = 0

    # ------------------------------------------------------------------ lines
    def line(self, room: str) -> Line:
        line = self._lines.get(room)
        if line is None:
            line = Line(room=room)
            self._lines[room] = line
        return line

    def lines(self) -> list[Line]:
        return list(self._lines.values())

    def _new_call_id(self) -> str:
        self._counter += 1
        return f"call-{int(self._clock())}-{self._counter}"

    # ------------------------------------------------------------- membership
    def join(self, room: str, role: str, name: str, number: str, send: Send) -> Party:
        """Attach a handset. A second handset in the same role replaces the first."""
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {role!r}")
        line = self.line(room)
        existing = line.parties.get(role)
        if existing is not None:
            existing.send({"type": "replaced", "reason": "another handset took this role"})
        party = Party(role=role, name=name, number=number, send=send)
        line.parties[role] = party
        # The joining handset is told what it walked into; the other is told who
        # arrived. One message each, so a client can rely on "joined" coming first.
        party.send({"type": "joined", **line.snapshot(), "you": role})
        self._broadcast(line, {"type": "line"}, exclude=party)
        return party

    def leave(self, room: str, party: Party) -> None:
        """Detach a handset; a call in progress ends because that side is gone."""
        line = self.line(room)
        if line.parties.get(party.role) is not party:
            return  # already replaced by a newer handset
        del line.parties[party.role]
        if line.state in (CallState.RINGING, CallState.CONNECTED):
            self._end(line, f"{party.role} disconnected")
        else:
            self._broadcast(line, {"type": "line"})

    def other(self, line: Line, role: str) -> Party | None:
        return line.parties.get(RECEIVER if role == CALLER else CALLER)

    # ---------------------------------------------------------------- actions
    def dial(self, line: Line) -> None:
        """Caller presses call. The receiver's handset starts ringing."""
        if line.state in (CallState.RINGING, CallState.CONNECTED):
            return
        caller = line.parties.get(CALLER)
        receiver = line.parties.get(RECEIVER)
        if caller is None:
            return
        if receiver is None:
            caller.send({"type": "unavailable", "message": "The receiver handset is not open."})
            return
        line.state = CallState.RINGING
        line.call_id = self._new_call_id()
        line.started = self._clock()
        line.connected = line.ended = line.end_reason = None
        line.verdict = "listening"
        # ``session_id`` is deliberately kept: the caller's handset connects its audio
        # (which is what creates the detection session) just before it dials, so
        # clearing it here would orphan the session and the receiver would see no
        # verdicts at all. A later call attaches its own session over this one.
        self._broadcast(line, {"type": "ringing"})

    def accept(self, line: Line) -> None:
        """Receiver picks up. Both sides go to connected and the timer starts."""
        if line.state is not CallState.RINGING:
            return
        line.state = CallState.CONNECTED
        line.connected = self._clock()
        self._broadcast(line, {"type": "connected"})

    def decline(self, line: Line) -> None:
        if line.state is not CallState.RINGING:
            return
        self._end(line, "declined by receiver")

    def hangup(self, line: Line, by: str) -> None:
        if line.state in (CallState.IDLE, CallState.ENDED):
            return
        self._end(line, f"ended by {by}")

    def _end(self, line: Line, reason: str) -> None:
        line.state = CallState.ENDED
        line.ended = self._clock()
        line.end_reason = reason
        self._broadcast(line, {"type": "ended", "reason": reason, "duration": self.duration(line)})

    def duration(self, line: Line) -> float:
        """Connected seconds, which is what a phone shows: ringing does not count."""
        if line.connected is None:
            return 0.0
        end = line.ended if line.ended is not None else self._clock()
        return max(0.0, end - line.connected)

    # --------------------------------------------------------------- verdicts
    def attach_session(self, line: Line, session_id: str) -> None:
        """Bind the caller's detection session so its verdicts reach the receiver."""
        line.session_id = session_id
        self._broadcast(line, {"type": "line"})

    def verdict(self, line: Line, message: dict) -> None:
        """Forward one detector message to the receiver handset only.

        The detector's own ``type`` (``window``, ``event``, ``session_end``) moves to
        ``event`` so that ``type`` stays the handset-protocol kind, ``verdict``.
        """
        if message.get("state"):
            line.verdict = message["state"]
        receiver = line.parties.get(RECEIVER)
        if receiver is not None:
            receiver.send({**message, "type": "verdict", "event": message.get("type")})

    # ------------------------------------------------------------- messaging
    def relay(self, line: Line, role: str, message: dict) -> None:
        """Pass a signalling message to the other handset (used for renegotiation)."""
        peer = self.other(line, role)
        if peer is not None:
            peer.send(message)

    def _broadcast(self, line: Line, message: dict, exclude: Party | None = None) -> None:
        payload = {**message, **line.snapshot()}
        for party in list(line.parties.values()):
            if party is not exclude:
                party.send({**payload, "you": party.role})


__all__ = ["CALLER", "RECEIVER", "ROLES", "CallState", "Exchange", "Line", "Party"]
