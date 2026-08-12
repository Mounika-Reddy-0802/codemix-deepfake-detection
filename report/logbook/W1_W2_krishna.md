# Log book — Sai Krishna, Weeks 1–2

## Week 1

**W1-T5 — free WebRTC harness.** Built `live_call/webrtc_harness/rtc_server.py`
using `aiortc`, with a 16 kHz PCM tap off the incoming track. The point of doing
this in Week 1 rather than waiting for Twilio was to get a real audio path we can
develop against for free — the Twilio trial is a 30-day clock and burning it on
development would leave nothing for the demo window.

**W1-T6 — Twilio design note.** Conference + Media Streams architecture, the four
alert layers (beep into the receiver leg, dashboard, SMS, call-end summary), and
the timing argument for activating the trial late.

The decision I would defend: `live_call/audio_source.py` defines a
`PcmFrameSource` protocol — 16 kHz mono signed-16-bit PCM frames — that *both*
backends implement. The WebRTC harness satisfies it now; the Twilio media handler
will satisfy it later. Everything downstream (streaming inference, verdict
smoothing, alerts) is written against that protocol, so none of it needs changing
when the backend swaps. That is the difference between a demo and a system.

## Week 2

**W2-T3 — reference-speaker selection + XTTS-v2 clone driver.** Wrote
`select_reference_speakers` (adult speakers with enough clean audio), the clone
driver with per-file metadata logging (tool, version, speaker, reference,
transcript, settings, pool), and the held-out Tortoise driver in
`src/data/heldout_tts.py` with an assertion that held-out clones are eval-pool
only.

## Reflection

I wrote the generation drivers in Week 2 but did not run them, because the ethics
sign-off has not happened. That felt like being blocked at the time. In Week 3 I
turned the rule into code (`src/data/ethics_gate.py`) so it is not a matter of
anyone's memory — the gate refuses before a model loads, and there is no override
switch. Having the driver ready and the gate closed is the right state to be in;
having the driver ready and having run it would not have been.

**Blocked on:** mentor ethics sign-off (everything generation-related); downloads
and the team listening pass (speaker shortlist → pool freeze).
