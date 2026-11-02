# Field recordings for the voice check: protocol

The voice check judges any audio a person brings to it: a recording made on the laptop
microphone, a voice note played from a phone, a call heard on speaker. Its accuracy table
(`docs/voice_check.md`) has rows for corpora and for the 30 real calls, but none for audio
recorded the way a user would record it. This protocol fills that row with the three of
us: our own voices into the laptop microphone, our own phone recordings played back into
it, and the clones of our voices made for the real-call test played the same way. It is
small (105 takes, 3 speakers, one laptop) and is reported as a pilot, like the real calls.

The speakers are **P1, P2 and P3**, the same codes as the real-call test, so each person's
clones are the files already made for them. Who is who is written only on the signed
consent forms, never in the repository.

## 0. Before anything is recorded

1. Each of us signs the consent form
   ([`docs/ethics/field_recording_consent_form.md`](../ethics/field_recording_consent_form.md)),
   and the mentor signs its approval box. Scan or photograph the signed pages into
   `docs/ethics/` as `field_recording_consent_<P>.pdf` (or `.jpg`); `.gitignore` keeps every
   PDF and image in that folder out of git.
2. No take of a person is recorded before that person has signed. A person who agreed to
   the recordings but not to replaying their clones skips the clone takes (K and S06-S10).

## What you need

- **The laptop with the voice check page open.** Every take is recorded on that page and
  labelled in its **Help measure accuracy** panel with the take ID from the tables below;
  the panel keeps the file and its label in `data/field_validation/`.
- **Each person's own phone** with its voice-recorder app (batch 1) and a second phone for
  a normal mobile call (batch 2; not WhatsApp or any internet call).
- **The clone files** of the person, from the real-call test:
  `data/real_calls/playback/xtts_v2_<P>_A.wav`, `xtts_v2_<P>_B.wav`,
  `chatterbox_<P>_A.wav` and `chatterbox_<P>_B.wav`, copied to that person's phone.
- **The scripts**: [`docs/real_calls/SCRIPTS.md`](../real_calls/SCRIPTS.md), the same
  sentences as the real calls.
- Batch 1: a quiet room. Batch 2: a different room and a second device that plays music.

## Fixed settings (write them down at the first take and keep them)

- The laptop sits on a table. The speaker sits facing the screen, mouth about 40 cm from
  the laptop.
- A replay is played from the phone's loudspeaker held 15 cm from the laptop, at one media
  volume (for example 70%) that is not changed between replays.
- Each take is about 15 seconds. Start the recording, wait one second, speak or play, and
  stop the recording about 15 seconds later (or when the page stops by itself).
- Nothing is trimmed, denoised or re-recorded to get a better verdict. A take is repeated
  only if it failed for a reason that has nothing to do with the verdict (the wrong file
  played, someone walked in talking), and the log says so.

## Batch 1 (Day 2, quiet room): 25 takes per person

**Phone recordings first.** Before any laptop take, each person records the ten lines of
F01-F10 on their own phone's voice recorder, holding the phone as for a voice note, and
names them `R01` to `R10` on the phone. These are the genuine recordings that R01-R10 play
back.

| Take | Kind | What to do | Label |
|---|---|---|---|
| F01 | live, read | read Script A, lines A01-A03, into the laptop | genuine |
| F02 | live, read | Script A, lines A04-A06 | genuine |
| F03 | live, read | Script A, lines A07-A09 | genuine |
| F04 | live, read | Script A, lines A10-A12 | genuine |
| F05 | live, read | Script B, lines B01-B03 | genuine |
| F06 | live, free talk | what you did this morning | genuine |
| F07 | live, free talk | a film or series you liked, and why | genuine |
| F08 | live, free talk | how to get from your home to college | genuine |
| F09 | live, free talk | your plans for the weekend | genuine |
| F10 | live, free talk | a dish you can cook, step by step | genuine |
| R01-R10 | own phone recording, replayed | play `R01` ... `R10` from the phone into the laptop | genuine |
| K01 | clone, replayed | play `xtts_v2_<P>_A.wav` from the phone | cloned (XTTS-v2) |
| K02 | clone, replayed | play `xtts_v2_<P>_B.wav` | cloned (XTTS-v2) |
| K03 | clone, replayed | play `chatterbox_<P>_A.wav` | cloned (Chatterbox) |
| K04 | clone, replayed | play `chatterbox_<P>_B.wav` | cloned (Chatterbox) |
| K05 | clone, replayed | `xtts_v2_<P>_B.wav` again, phone held in the hand instead of on the table | cloned (XTTS-v2) |

Free talk is about 15 seconds of natural speech in the Hindi-English mix you normally use;
the topic is only a prompt. The genuine replays (R) go through the same phone loudspeaker as
the clones (K), so they are the control for the loudspeaker: a clone caught while its own
speaker's replays pass is caught for being a clone. K05 repeats K02 to show how steady one
verdict is.

## Batch 2 (Day 5, another room, music, a call on speaker): 10 takes per person

The case the product is for: a call on speakerphone, checked by the laptop. A second device
plays instrumental music 1-2 m away at one low level, set once so that speech is clearly
louder, and left there for every take. Two phones are on a normal mobile call: the person
holds the calling phone; the receiving phone lies on the table on speaker, 30 cm from the
laptop, and the laptop records what the receiving phone plays.

| Take | Kind | What to do | Label |
|---|---|---|---|
| S01 | call on speaker, read | Script A, lines A01-A03, spoken into the calling phone | genuine |
| S02 | call on speaker, read | Script A, lines A04-A06 | genuine |
| S03 | call on speaker, read | Script B, lines B04-B06 | genuine |
| S04 | call on speaker, free talk | what you did yesterday | genuine |
| S05 | call on speaker, free talk | a place you would like to visit | genuine |
| S06 | clone into the call | play `xtts_v2_<P>_A.wav` into the calling phone, 10-15 cm from its microphone | cloned (XTTS-v2) |
| S07 | clone into the call | `xtts_v2_<P>_B.wav` | cloned (XTTS-v2) |
| S08 | clone into the call | `chatterbox_<P>_A.wav` | cloned (Chatterbox) |
| S09 | clone into the call | `chatterbox_<P>_B.wav` | cloned (Chatterbox) |
| S10 | clone into the call | `xtts_v2_<P>_B.wav` again | cloned (XTTS-v2) |

The clone files are played from a laptop or a third phone into the calling phone, as in
the real-call replays (C19-C30).

## Names, labels and the plan file

Every take is named `<P>_<take>`, for example `P1_F01` or `P3_S08`, and labelled in the
panel as genuine or cloned with its tool. The same plan, one row per take for all three
people (105 rows), is in [`take_plan.csv`](take_plan.csv); the scoring reads its labels from
there, so a take mislabelled in the panel is caught rather than counted.

| Batch | Per person | All three | Genuine | Cloned |
|---|---|---|---|---|
| 1, quiet room | 25 | 75 | 60 | 15 |
| 2, call on speaker with music | 10 | 30 | 15 | 15 |
| Total | 35 | 105 | 75 | 30 |

## Scoring

The takes are scored by the voice check's own scoring at its frozen operating point; no
threshold, front end or model is chosen on them. The results go into the voice check's
accuracy table as the field rows, split by batch and by kind (live read, free talk, genuine
replay, clone replay, call on speaker), each with its count and a Wilson interval, and every
number is read back from its result file by a test.

## What is kept, and where

| Item | Where | In git? |
|---|---|---|
| This protocol and the take plan | `docs/field_validation/` | yes |
| Per-take scores and verdicts, the accuracy rows | `experiments/results/`, `docs/voice_check.md` | yes (no audio) |
| The takes, the phone recordings, the panel's log | `data/field_validation/` | **never** (`data/` is ignored) |
| Signed consent forms, mentor approval | `docs/ethics/` | **never** (PDFs and images there are ignored) |

No take is used to train or tune any model: `src/training/config_guard.py` rejects any
training row whose source is `field_validation`, as it rejects the real calls.

## What this test cannot show

- Three speakers, our own voices, one laptop and one room per batch: a pilot, not an
  estimate for every user.
- The clones are the two open tools of the real-call test, made from about 20 seconds of
  reading; a commercial service's clones are not in it.
- A phone loudspeaker can move a score by itself. The genuine replays (R01-R10) and the
  genuine call takes (S01-S05) are the controls that separate that from the clone.
