"""The telephone exchange: call states, who is told what, and the verdict routing."""

from __future__ import annotations

import pytest

from live_call.phone import CALLER, RECEIVER, CallState, Exchange


class Handset:
    """A fake handset that records what the exchange sends it."""

    def __init__(self) -> None:
        self.messages: list[dict] = []

    def __call__(self, message: dict) -> None:
        self.messages.append(message)

    def types(self) -> list[str]:
        return [m["type"] for m in self.messages]

    def last(self, type_: str) -> dict | None:
        return next((m for m in reversed(self.messages) if m["type"] == type_), None)


@pytest.fixture()
def wired():
    """An exchange with both handsets on line 'demo', and a clock we control."""
    now = {"t": 1000.0}
    exchange = Exchange(clock=lambda: now["t"])
    caller, receiver = Handset(), Handset()
    exchange.join("demo", CALLER, "Ravi", "+91 1", caller)
    exchange.join("demo", RECEIVER, "Priya", "+91 2", receiver)
    return exchange, exchange.line("demo"), caller, receiver, now


def test_a_full_call_walks_through_every_state(wired):
    exchange, line, caller, receiver, now = wired
    assert line.state is CallState.IDLE

    exchange.dial(line)
    assert line.state is CallState.RINGING
    assert "ringing" in caller.types() and "ringing" in receiver.types()
    assert line.call_id and receiver.last("ringing")["parties"]["caller"]["name"] == "Ravi"

    now["t"] = 1005.0
    exchange.accept(line)
    assert line.state is CallState.CONNECTED
    assert "connected" in caller.types() and "connected" in receiver.types()

    now["t"] = 1035.0
    assert exchange.duration(line) == pytest.approx(30.0)

    exchange.hangup(line, CALLER)
    assert line.state is CallState.ENDED
    assert receiver.last("ended")["duration"] == pytest.approx(30.0)


def test_ringing_time_is_not_counted_as_call_duration(wired):
    exchange, line, _caller, _receiver, now = wired
    exchange.dial(line)
    now["t"] = 1020.0  # rang for 20 s
    exchange.accept(line)
    now["t"] = 1025.0
    assert exchange.duration(line) == pytest.approx(5.0)


def test_declining_ends_the_call_without_connecting(wired):
    exchange, line, caller, _receiver, _now = wired
    exchange.dial(line)
    exchange.decline(line)
    assert line.state is CallState.ENDED
    assert line.connected is None
    assert "declined" in caller.last("ended")["reason"]
    assert exchange.duration(line) == 0.0


def test_dialling_with_no_receiver_tells_the_caller(wired):
    exchange, line, caller, _receiver, _now = wired
    exchange.leave("demo", line.parties[RECEIVER])
    exchange.dial(line)
    assert line.state is CallState.IDLE
    assert caller.last("unavailable") is not None


def test_a_hangup_by_either_side_ends_the_call(wired):
    exchange, line, caller, receiver, _now = wired
    exchange.dial(line)
    exchange.accept(line)
    exchange.hangup(line, RECEIVER)
    assert line.state is CallState.ENDED
    assert "receiver" in line.end_reason
    assert caller.last("ended") is not None and receiver.last("ended") is not None


def test_a_handset_leaving_mid_call_ends_it(wired):
    exchange, line, caller, _receiver, _now = wired
    exchange.dial(line)
    exchange.accept(line)
    exchange.leave("demo", line.parties[RECEIVER])
    assert line.state is CallState.ENDED
    assert "disconnected" in line.end_reason
    assert caller.last("ended") is not None


def test_verdicts_reach_the_receiver_and_never_the_caller(wired):
    exchange, line, caller, receiver, _now = wired
    exchange.dial(line)
    exchange.accept(line)
    exchange.attach_session(line, "webrtc-1")
    exchange.verdict(line, {"type": "window", "score": 0.0002, "state": "likely_fake", "t": 10.0})
    sent = receiver.last("verdict")
    assert sent["score"] == 0.0002 and sent["event"] == "window"
    assert line.verdict == "likely_fake"
    assert caller.last("verdict") is None, "the caller must never learn the verdict"


def test_each_handset_is_told_what_it_joined_and_who_arrived(wired):
    """The joiner gets "joined"; whoever was already there gets "line"."""
    _exchange, line, caller, receiver, _now = wired
    assert caller.last("joined")["you"] == CALLER
    assert receiver.last("joined")["you"] == RECEIVER
    assert caller.last("line")["you"] == CALLER  # told the receiver arrived
    assert receiver.last("line") is None  # it arrived last, nobody after it
    assert set(line.snapshot()["parties"]) == {CALLER, RECEIVER}


def test_every_state_message_reaches_both_handsets(wired):
    exchange, line, caller, receiver, _now = wired
    exchange.dial(line)
    exchange.accept(line)
    for kind in ("ringing", "connected"):
        assert caller.last(kind)["you"] == CALLER
        assert receiver.last(kind)["you"] == RECEIVER


def test_a_second_handset_in_one_role_replaces_the_first(wired):
    exchange, _line, caller, _receiver, _now = wired
    replacement = Handset()
    exchange.join("demo", CALLER, "Ravi on another tab", "+91 1", replacement)
    assert caller.last("replaced") is not None
    assert exchange.line("demo").parties[CALLER].send is replacement


def test_dialling_twice_does_not_restart_a_live_call(wired):
    exchange, line, _caller, receiver, _now = wired
    exchange.dial(line)
    first = line.call_id
    exchange.accept(line)
    exchange.dial(line)
    assert line.call_id == first and line.state is CallState.CONNECTED
    assert receiver.types().count("ringing") == 1


def test_accept_outside_ringing_is_ignored(wired):
    exchange, line, _caller, _receiver, _now = wired
    exchange.accept(line)
    assert line.state is CallState.IDLE


def test_an_unknown_role_is_refused(wired):
    exchange, _line, _caller, _receiver, _now = wired
    with pytest.raises(ValueError, match="role must be"):
        exchange.join("demo", "operator", "X", "+91 3", Handset())


def test_lines_are_independent(wired):
    exchange, line, _caller, receiver, _now = wired
    other = exchange.line("demo2")
    exchange.join("demo2", CALLER, "Other", "+91 9", Handset())
    exchange.join("demo2", RECEIVER, "Other receiver", "+91 8", Handset())
    exchange.dial(other)
    assert line.state is CallState.IDLE
    assert receiver.last("ringing") is None
    assert {line_.room for line_ in exchange.lines()} == {"demo", "demo2"}


def test_dialling_keeps_the_session_attached_by_the_caller_audio(wired):
    """The handset connects its audio, then dials; dialling must not orphan it."""
    exchange, line, _caller, receiver, _now = wired
    exchange.attach_session(line, "webrtc-7")
    exchange.dial(line)
    assert line.session_id == "webrtc-7"
    exchange.accept(line)
    exchange.verdict(line, {"type": "window", "score": 0.9, "state": "genuine", "t": 4.0})
    assert receiver.last("verdict")["score"] == 0.9


def test_a_new_call_replaces_the_previous_session(wired):
    exchange, line, _caller, _receiver, _now = wired
    exchange.attach_session(line, "webrtc-1")
    exchange.dial(line)
    exchange.accept(line)
    exchange.hangup(line, CALLER)
    exchange.attach_session(line, "webrtc-2")
    exchange.dial(line)
    assert line.session_id == "webrtc-2"
    assert line.verdict == "listening"
