"""The WebRTC room: what a joiner is given, and what it keeps after a call ends.

These guard two bugs that only show up in a live call: a room that hands a new
joiner stale tracks (aiortc then fails the whole negotiation), and a Room missing
the queue its audio tap writes into (the tap dies and no window is ever scored).
"""

from __future__ import annotations

import pytest

pytest.importorskip("aiortc")

from live_call.webrtc_harness import rtc_server as rtc  # noqa: E402


class FakeTrack:
    kind = "audio"


def test_a_room_has_the_queue_its_tap_writes_into():
    room = rtc.Room("r")
    assert room.queue is not None and room.queue.maxsize > 0
    assert room.source() is not None


def test_live_tracks_excludes_the_asker_and_anyone_who_left():
    room = rtc.Room("r")
    caller, receiver, gone = FakeTrack(), FakeTrack(), FakeTrack()
    room.pcs = {"pc1": object(), "pc2": object()}
    room.tracks = {"pc1": caller, "pc2": receiver, "pc3": gone}
    assert room.live_tracks(except_pc="pc1") == [receiver]
    assert room.live_tracks(except_pc="pc2") == [caller]
    assert gone not in room.live_tracks(except_pc="pc9"), "a peer that left must not be offered"


def test_forward_gives_each_track_once():
    class Pc:
        def __init__(self):
            self.added = []

        def addTrack(self, track):  # noqa: N802 - aiortc's spelling
            self.added.append(track)

    room, pc, track = rtc.Room("r"), Pc(), FakeTrack()
    assert rtc._forward(room, "pc1", pc, track) is True
    assert rtc._forward(room, "pc1", pc, track) is False, "a repeat must not add a second time"
    assert len(pc.added) == 1
