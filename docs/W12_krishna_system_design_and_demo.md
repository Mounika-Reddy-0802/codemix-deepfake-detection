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

---

# Later in week 12 — paper chapters, the external review, and the Kaggle runs

An external reviewer (IIT Madras) sent six comments. I took the shortcut floor
(comment 5), the bibliography (comment 6) and the unseen-tool work (comment 2), wrote
the introduction, related work and conclusion, and built the Kaggle notebooks that
produced the seed and variant runs.

## What I did, and why it is built this way

### Matched shortcut floors (comment 5)

The Tortoise floor in the paper had been fitted on Tortoise clips the detector never
sees and scored on different clips from the detector. `src/training/floor_compare.py`
now fits each detector's floor on that detector's own training data, scores it on the
detector's exact clips, and compares the two by a paired bootstrap over speakers, so
every claim of "beats the floor" is a paired, speaker-level test. Several cells the
paper had called evidence turned out inconclusive. The old floor is kept, labelled
"oracle".

### The bibliography (comment 6)

`paper/refs.bib` went from 10 entries to 38, every field checked against the
publisher, Crossref or the ISCA Archive (`docs/references_verification.md`); one wrong
author was corrected. Related work was rewritten with a "how this work differs"
subsection.

### Three training changes, chosen by a rule fixed in advance (comment 2)

Opt-in and off by default: a fresh crop per epoch (the old code reused one crop, a
bug), augmentation that randomises the recording cues the floor uses, and a learned
layer mix. The rule, written before any run: lowest dev EER wins, ties within half a
point go to fewer changes, test sets never consulted. `src/reporting/variants.py`
replays it from the run files and runs the exploratory comparison.

### Kaggle notebooks that finish inside one cell

The first two Kaggle runs came back without a summary: the cells after the five-hour
run cell never ran, because Kaggle stops sending later cells when the page loses its
connection. The run, the summary and the download link now share one cell, the summary
is rewritten into the zip after every run, and a crash still ends with both. Before
sending, each notebook ran through a laptop simulation (fake training, four
scenarios), which caught a real bug: the exploratory summary would have counted the
published recipe's seed as the variant's.

## Numbers

| | Result |
|---|---|
| References | 10 -> 38, all verified, all cited |
| Selection (dev EER, seed 1234) | published 5.50 kept; crop 9.10, crop + augment 5.34 (tie), the two layer-mix variants 9.10 |
| Exploratory: crop + augment + layer mix, Tortoise, 5 seeds | **26.63 ± 1.66** against 35.59 ± 3.35 (exact p 0.008; 0.016 on the fresh seeds) |
| Same variant, RVC | 7.72 ± 1.33 against 5.33 ± 1.27 |
| Same variant against its Tortoise floor | inconclusive in all 5 seeds (-9.0 to -13.6 points) |
| Dev EER, variant vs recipe | 7.11 vs 6.96: the dev set cannot see the gain |

## Honest limitations

- **The variant was chosen after seeing a test score.** The fresh seeds remove the
  luck of one run but not the choice; it needs a second unseen tool to confirm it.
- **Both layer-mix runs crashed first** on a strict load of the Stage-1 checkpoint;
  fixed (`load_initial_weights`, tested on the real checkpoint) and re-run. The learned
  layer weights were not saved.
- **My notebooks cost two incomplete runs** before the one-cell fix; nothing measured
  was lost, but two summaries had to be rebuilt from `runs.json`.
- **Score fusion with S3 was not done**: its weights would be set on a dev set that,
  as the variant shows, cannot see the unseen tool.

## How to run it

```bash
python -m src.training.floor_compare --clean-root <clean bundle> --channel-root <channel bundle>   # matched floors, paired tests
python -m src.reporting.variants                # selection replay + exploratory check
pytest tests/test_variants.py tests/test_layer_mix_and_sampler.py tests/test_matched_floor_reporting.py
```

## What's next

- Confirm the augmentation lead on a second unseen tool, with a validation split
  built from a held-out tool rather than the in-domain dev set.
- Pull requests: Mounika's branch first (mine contains her commits), then mine.
