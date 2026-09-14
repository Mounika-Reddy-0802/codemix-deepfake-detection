"""A monitored call from first frame to summary, and where its alerts go (owner SK)."""

from __future__ import annotations

import asyncio

import numpy as np

from live_call.alerts import AlertDispatcher, TwilioSettings, announcement_twiml, sms_body
from live_call.replay import FileSource
from live_call.session import CallSession, Hub, SessionInfo, SessionRegistry, new_session_id
from live_call.verdict_engine import EngineConfig, Event, EventKind, State

SR = 16_000
CFG = EngineConfig(threshold=0.5)
NO_TWILIO = TwilioSettings("", "", "", "", "")


def _speech(seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.1 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)


def _run(score: float, kind: str = "replay", meta: dict | None = None, seconds: float = 20.0):
    hub, dispatcher = Hub(), AlertDispatcher(settings=NO_TWILIO)
    queue = hub.subscribe()
    info = SessionInfo(new_session_id(kind), kind, "test call", meta=meta or {})
    session = CallSession(info, lambda w: score, CFG, hub, dispatcher)
    asyncio.run(session.run(FileSource(_speech(seconds))))
    messages = []
    while not queue.empty():
        messages.append(queue.get_nowait())
    return session, dispatcher, messages


def test_cloned_call_publishes_windows_alerts_and_summary():
    session, dispatcher, messages = _run(0.05)
    types = [m["type"] for m in messages]
    assert types[0] == "session_start" and "session_end" in types
    kinds = [m["kind"] for m in messages if m["type"] == "event"]
    assert kinds == ["beep", "sms", "summary"]
    assert session.engine.summary.peak_state == State.LIKELY_FAKE
    assert all(a.channel == "dashboard" for a in dispatcher.sent)  # not a phone call


def test_genuine_call_sends_no_alerts():
    session, dispatcher, messages = _run(0.95)
    assert not [m for m in messages if m["type"] == "event"]
    assert session.snapshot()["state"] == "genuine" and session.ended
    assert dispatcher.sent == []


def test_twilio_call_routes_warning_to_receiver_and_sms_dry_without_credentials():
    meta = {"conference": "call-CA1", "receiver_call_sid": "CA2", "from": "+91999"}
    _, dispatcher, _ = _run(0.05, kind="twilio", meta=meta)
    channels = [(a.channel, a.kind, a.dry_run) for a in dispatcher.sent]
    assert ("announce", "beep", True) in channels
    assert ("announce", "sms", True) in channels
    assert ("sms", "sms", True) in channels and ("sms", "summary", True) in channels


def test_live_dispatcher_calls_twilio_rest():
    calls = []

    class FakeRest:
        async def conference_sid(self, name):
            calls.append(("conference", name))
            return "CF1"

        async def announce_to_participant(self, conference_sid, call_sid, kind):
            calls.append(("announce", conference_sid, call_sid, kind))

        async def send_sms(self, body):
            calls.append(("sms", body))

    dispatcher = AlertDispatcher(settings=NO_TWILIO, rest=FakeRest())
    info = SessionInfo(
        "s1", "twilio", "call", meta={"conference": "call-CA1", "receiver_call_sid": "CA2"}
    )
    asyncio.run(
        dispatcher.deliver(info, Event(EventKind.SMS, 12.0, State.LIKELY_FAKE, 0.1, "warn"))
    )
    assert ("announce", "CF1", "CA2", "sms") in calls
    assert any(c[0] == "sms" for c in calls)


def test_a_failing_alert_channel_does_not_stop_monitoring():
    class Broken:
        async def deliver(self, info, event):
            raise RuntimeError("SMS provider down")

    hub = Hub()
    queue = hub.subscribe()
    session = CallSession(SessionInfo("s", "replay", "x"), lambda w: 0.05, CFG, hub, Broken())
    asyncio.run(session.run(FileSource(_speech(20.0))))
    types = []
    while not queue.empty():
        types.append(queue.get_nowait()["type"])
    assert "alert_error" in types and types[-1] in ("session_end", "event", "alert_error")
    assert session.ended


def test_hub_drops_oldest_for_a_stalled_dashboard():
    hub = Hub(queue_size=3)
    queue = hub.subscribe()
    for i in range(5):
        hub.publish({"i": i})
    assert [queue.get_nowait()["i"] for _ in range(3)] == [2, 3, 4]


def test_registry_keeps_recent_history_for_late_dashboards():
    registry = SessionRegistry(keep_ended=1)
    for _ in range(3):
        session, _, _ = _run(0.95, seconds=6.0)
        registry.add(session)
    assert len([s for s in registry.all() if s.ended]) == 1
    assert registry.replay_history()[0]["type"] == "session_start"


def test_alert_texts():
    info = SessionInfo("s", "twilio", "x", meta={"from": "+91999"})
    body = sms_body(Event(EventKind.SMS, 1.0, State.LIKELY_FAKE, 0.1, "m"), info)
    assert "+91999" in body and "OTP" in body
    assert "<Say" in announcement_twiml("beep", "https://x/beep.wav")
    assert "<Play>https://x/beep.wav</Play>" in announcement_twiml("sms", "https://x/beep.wav")


def test_settings_detect_placeholders():
    placeholder = TwilioSettings(
        "ACxxxxxxxx",
        "your-twilio-auth-token",
        "+1XXXXXXXXXX",
        "+91XXXXXXXXXX",
        "https://your-subdomain",
    )
    assert not placeholder.configured
    assert TwilioSettings(
        "AC123", "tok", "+15550001111", "+919999999999", "https://a.ngrok-free.app"
    ).configured
