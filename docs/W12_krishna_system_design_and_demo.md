# Week 12 — Krishna — system design chapter, backup demo, and a Streamlit stand-in

Three pieces this week, all aimed at the same thing: making the live-call system
defensible in front of an examiner rather than merely working on my laptop.

## What I did, and why it is built this way

### The backup demo recording

The risk register has said since week 10 that demo flakiness is a top risk,
mitigated by a recording made in advance. There was no recording. There is one now,
and it is generated rather than filmed: `scripts/record_demo.py` drives the console
through three calls in a real browser and writes a video from the real system.

Generated, not filmed, for one reason — a recording made by hand goes stale the
moment the interface changes and nobody notices until the viva. This one is a
command.

### End-to-end tests of the console

Every serious defect in this demo lived in the wiring between components that each
worked correctly alone, so no unit test could see them. The worst example: an alert
carries its severity in one field and its delivery channel in another, the interface
read the channel, and the warning silently never appeared while every unit test
passed.

`tests/test_console_e2e.py` places real calls against the loaded model and asserts
the four properties that actually matter (below). They skip rather than fail without
Playwright, Chrome or a running server, so CI stays light.

### System design chapter and architecture figure

`paper/sections/system.tex`, with `paper/figures/system_architecture.pdf` generated
by `src/reporting/architecture.py`. The figure is code so it regenerates when the
system changes, and `tests/test_architecture.py` asserts that every module the
diagram names exists — a diagram that claims a component the repository lacks is a
lie in a figure.

The chapter is organised around four requirements, each of which rules out an
otherwise obvious design: the verdict must reach the callee only, it must arrive
during the call, the audio must match the training channel, and one false alarm
costs more than one missed clone.

### A Streamlit stand-in

`app/` is a standalone page for scoring a file, a microphone recording or a prepared
clip while the live-call work continues. It imports nothing from `live_call/`. It
reuses the same preprocessing, channel simulation, windowing and threshold as the
evaluation, because a demo that scores audio differently from the paper proves
nothing.

## Numbers

Backup recording, verified call by call against the server's own record:

| Scene | Clip | Verdict recorded | Expected |
|---|---|---|---|
| 1 | Genuine caller (RVC source) | `genuine` | yes |
| 2 | XTTS-v2 clone | `likely_fake` | yes |
| 3 | Tortoise clone | `genuine` | yes — the documented miss |

1 min 35 s, 10.2 MB, at `C:/dfdata/demo_recordings/live_call_demo_backup.webm`.

| Suite | Result |
|---|---|
| Full test suite | 842 passed, 4 skipped |
| End-to-end console tests | 4 passed against the live server, 4 skipped without it |
| Streamlit helper tests | 16, no Streamlit runtime needed |

## Honest limitations

- **The Twilio leg is still unverified on live numbers.** The trial credit is
  exhausted and trial accounts cannot use inline TwiML or arbitrary SMS bodies. The
  code is written and unit-tested; the telephone network is the protocol this whole
  project was designed around, and it is the one leg not demonstrated.
- **The end-to-end tests are not in CI.** They need a GPU, the checkpoint and a
  browser. They run on demand, which means they only catch a regression when someone
  remembers to run them.
- **One genuine demo clip still dips on the live path** — 3 low windows out of 9,
  under the 4-window warning threshold but not by much. The Opus operating point is
  not calibrated yet.
- **The two-device path has never been exercised over https.** The receiver opens
  its microphone while the phone rings, and that code has only ever run with both
  handsets in one browser.
- **The architecture figure is hand-laid-out in matplotlib.** It regenerates, but
  moving a box means editing coordinates.

## How to run it

```bash
DEMO_CLIPS_DIR=/path/to/clips python -m uvicorn live_call.server:app --port 8000
python scripts/record_demo.py --out /path/for/the/video
pytest tests/test_console_e2e.py            # skips without a server
python -m src.reporting.architecture        # redraws the figure
streamlit run app/streamlit_app.py          # the stand-in demo
```

## What's next

- Calibrate a second operating point for Opus-carried calls, so the browser demo and
  the telephone deployment stop sharing a threshold measured on one channel.
- Verify the Twilio leg once the account is funded.
- Write the remaining chapters I own: introduction, related work, conclusion.
