# Week 11 — Krishna — the two-handset phone demo and the live console

The week-10 demo worked but did not *look* like a phone call: one browser page held
both sides of the conversation and a settings form, so the thing an evaluator saw
was a web form, not a caller and a victim. This week rebuilt the demonstration
around two real handsets and one monitored line, and then fixed the four defects
that only appear once you actually place calls through it.

## What I did, and why it is built this way

### A call state machine, separate from the detector

`live_call/phone.py` holds an `Exchange`: lines, parties, and the states
`idle → ringing → connected → ended`, plus dial, accept, decline and hangup. It is
pure Python with no sockets in it, so the call logic is testable without a browser
(`tests/test_phone.py`, 16 tests).

The important rule lives here: **verdicts are routed to the receiver only**. A
fraudster whose voice is being scored must not learn that they were caught,
otherwise the system teaches attackers how to evade it. The caller's handset
receives call state and nothing else.

### Two handsets, and a console that shows both

- `/phone/caller` — device A. Picks the voice on the line, dials, and never sees a
  verdict.
- `/phone/receiver` — device B. Sits idle, rings with the caller's name and number,
  and after answering shows the live verdict, the score timeline and the warning.
- `/demo` — one screen holding both handsets, the signal path, the detector's live
  analysis and the measured results. This is the view for an evaluation where
  everyone is looking at one laptop.

The console is not a mock: it opens two signalling sockets and two peer connections,
exactly as two devices would.

### Scoring starts when the call is answered

Monitoring used to begin when the caller's audio track arrived, which is while the
phone is still ringing. The media path is still settling then, and the glitch scored
as low windows. A call is now held in `pending` until the receiver answers.

## Numbers

Every clip replayed through the real browser call path, not offline
(`docs/results/demo_clips_live_check.json`, threshold 0.500, deployed checkpoint
`lora_norm_rvc_channel_best.pt`):

| Clip | Truth | Windows | Low windows | Final verdict | Correct |
|---|---|---:|---:|---|---|
| Genuine caller (RVC source) | real | 10 | 1 | genuine | yes |
| XTTS-v2 clone | cloned | 9 | 9 | likely cloned | yes |
| RVC voice conversion | cloned | 9 | 9 | likely cloned | yes |
| Tortoise clone (unseen tool) | cloned | 9 | 1 | genuine | **no** |
| Genuine caller | real | 9 | 3 | genuine | yes |

Time from answer to the warning on a cloned call: **10 seconds** (2 low windows
raise a caution at 6 s, 4 raise the warning at 10 s).

Tests: 798 pass, ruff clean.

## Four defects this found

1. **The warning never appeared.** An alert carries its severity in `state` and its
   delivery channel in `kind` (`beep`, `sms`). Both the console and the receiver
   handset were reading `kind`, so the most important moment of the demo silently
   did nothing. This is why a UI needs an end-to-end test and not just a unit test.
2. **Rooms accumulated dead audio tracks.** Every past call's track stayed in the
   room, so a new joiner was handed more tracks than its offer had m-lines and
   aiortc failed the whole negotiation. After a few calls the demo simply stopped
   working. Tracks are now keyed by peer and dropped when that peer leaves.
3. **A broken edit left `Room` without the queue its audio tap writes into.** The
   tap died immediately, caught its own exception, logged at info level — which
   uvicorn drops — and every call scored zero windows while looking healthy.
4. **A genuine caller was cautioned.** Offline the clip scores ≥0.993 on every
   window; through the call it dipped to 0.001. The browser encodes with Opus while
   the detector is trained on G.711 telephone audio, and the cue the model uses is
   phase-level, so a thrifty encode of a *real* voice can look synthetic. Fixed by
   forcing constant high-bitrate Opus with DTX off, and by scoring only after the
   call is answered.

## Honest limitations

- **Defect 4 is mitigated, not solved.** One of the two genuine clips still dips to
  3 low windows out of 9 on the live path. It stays under the 4-window warning
  threshold, but it would raise a caution on a longer call. The real fix is training
  or calibrating on Opus-encoded audio, which is not done.
- **The demo clip order is chosen from measurement.** The genuine clip that survives
  the live path cleanly is listed first. That is honest, but it means the default
  demo shows the system at its best.
- **Tortoise is still missed on the live path**, consistent with the 30.99% EER we
  report for the unseen tool in the telephone condition.
- **The Twilio path is implemented but was not exercised this week.** The trial
  credit is exhausted (balance −1.15 USD) and trial accounts cannot use inline TwiML
  or arbitrary SMS bodies, so the PSTN leg is unverified since week 10.
- **The console has no automated UI test.** The defects above were found by driving
  a real browser by hand; that check is not in CI.

## How to run it

```bash
export PYTHONPATH=.
DEMO_CLIPS_DIR=/path/to/demo_clips \
  python -m uvicorn live_call.server:app --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000/demo` for the one-screen console, or
`/phone/caller` and `/phone/receiver` on two devices. The microphone needs
`localhost` or https; sending a prepared clip needs no microphone at all.

```bash
pytest tests/test_phone.py tests/test_rtc_room.py tests/test_live_server.py
```

## What's next

- Put the Twilio leg back under test once the account is funded; the receiver-only
  warning path is written but unproven on a real carrier.
- Calibrate a second operating point for Opus-carried calls, so the WebRTC demo and
  the PSTN deployment do not share a threshold measured on one channel only.
- Record the backup screen capture of the console demo (week-10 risk item 3).
