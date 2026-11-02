# Voice check: real or cloned?

A single page that answers one question for any audio: is this voice real or cloned?
There is no phone call. The audio can be:

- speech recorded on the spot through the laptop microphone;
- audio played aloud from another device (a phone, a speaker) into the microphone;
- a phone-call recording, voice note or any audio file (wav, mp3, m4a, aac, amr, ogg, webm).

## Run it

```bash
python -m voice_check.app          # then open http://127.0.0.1:7860
```

Needs about 2.5 GB of free memory: the detector (SSL-AASIST, 632 MB file) is loaded on the
first check. Its path, threshold and front end are in `configs/voice_check.yaml`.

From the command line, for a batch of files:

```bash
python -m src.inference.any_audio clip.mp3 call.m4a note.ogg --out verdicts.csv
```

## How a recording is judged

`src/inference/any_audio.py`, the same steps for every input:

1. decode to 16 kHz mono;
2. cut silent ends (quieter than −45 dBFS);
3. pass through an 8 kHz G.711 telephone round trip, because the detector was trained only
   on telephone-band audio;
4. score 4 s windows every 2 s, each brought to −23 dBFS; silent windows are skipped;
5. average P(genuine) over the windows and compare it with the frozen threshold (0.082,
   chosen on development calls only; since decision D1 part a the development calls include
   20 spontaneous IndicVoices speakers, and the point chosen on read speech alone was 0.580).

A replayed recording of a real person counts as **real**: the question is whether the voice
was cloned, not whether it was played back.

## How accurate it is today

From stored scores of the exact checkpoint the page uses
(`python -m src.reporting.any_audio_accuracy`; full table in
`experiments/results/any_audio_accuracy.md`). Percent judged correctly at the frozen
threshold, and in brackets at the earlier point chosen on read-speech development calls only.

| Audio | Correct at 0.082 (at 0.580) |
|---|---|
| Genuine read speech, telephone line | 99.9 % of 2,400 (99.6 %) |
| Genuine **spontaneous** speech, telephone line | 99.3 % of 1,000 (92.8 %) |
| Genuine live mobile calls | 12 of 12 (12 of 12) |
| Genuine recordings replayed into a phone | 6 of 6 (6 of 6) |
| XTTS-v2 clones (line and AMR-NB codec) | 98.5–99.4 % (99.8–99.9 %) |
| Tortoise clones (never seen in training) | 71.2–81.1 % (91.0–94.6 %) |
| XTTS-v2 clones replayed into a phone | 5 of 6 (6 of 6) |
| RVC voice conversion | 45.4 % (71.3 %) |
| IndicF5 clones (never seen in training) | 12.5 %, 2.3 % through AMR-NB (33.1 %, 11.8 %) |
| Chatterbox clones (never seen in training) | 4.4 %; 0 of 6 replayed into a phone (20.0 %; 0 of 6) |

**Strong:** real voices of every kind, unscripted speech included, and clones from the tool
family it was trained on (XTTS-v2). **Weak:** every other generator. The move from 0.580 to
0.082 removed most false alarms on unscripted speech and let through more clones of every
unseen tool; no single threshold serves both, because the newest generators' clones score
as high as genuine unscripted speech (`docs/results/operating_points_spont_v1.md`).

## The retrained detector, measured here (decision D1 part b)

SSL-AASIST retrained with 150 spontaneous IndicVoices speakers was measured in this voice check
under a rule written before any of its voice-check numbers existed
(`docs/results/spontaneous_training_v1.md`, "Voice check"; commit `98d145c`): both seeds, each
at its own operating point chosen on dev only, had to keep genuine speech (spontaneous at least
97.3%, read at least 99.0%, at least 11 of 12 live calls and 5 of 6 replays), raise the mean over
the five unseen-tool rows by 10 points with none falling more than 10, and keep XTTS-v2 at 98%.
`python -m src.reporting.voice_check_adoption` applies it
(`experiments/results/voice_check_adoption_v1.json`).

| Audio, telephone line | Current: seed 1234 at 0.082 | Retrained seed 1234 at 0.035 | Retrained seed 2025 at 0.515 |
|---|---|---|---|
| Genuine spontaneous (HiACC) | 99.3 % | 99.5 % | 99.1 % |
| Genuine read: new speakers / language control | 99.9 % / 99.9 % | 99.8 % / 99.8 % | **97.6 % / 98.7 %** |
| Genuine live calls / replays | 12 of 12 / 6 of 6 | 12 of 12 / 6 of 6 | 12 of 12 / 6 of 6 |
| XTTS-v2 clones | 99.4 % | 100.0 % | 100.0 % |
| Tortoise (new speakers / CM04) | 81.1 % / 76.9 % | 93.7 % / 84.5 % | 96.4 % / 90.4 % |
| IndicF5 | 12.5 % | 35.6 % | 45.4 % |
| Chatterbox | 4.4 % | 51.6 % | 51.6 % |
| RVC | 45.4 % | 85.8 % | 89.0 % |
| Mean of the five unseen-tool rows | 44.1 % | 70.2 % | 74.6 % |

**Decision: the current detector stays.** Seed 1234 meets every condition; seed 2025 calls
2.4 % of read genuine speech cloned, above the 1 % the rule allows, so the rule is not met and
neither seed is adopted. Both trainings catch far more clones of every unseen tool than the
current detector (IndicF5 over AMR-NB too: 38.6-39.0 % against 2.3 %), at no cost on
spontaneous speech, so this is where the next version of the voice check comes from. Adopting it
needs a recipe whose every training meets the rule, or a rule for the average of the two
checkpoints written before that average is measured; picking the seed that passed would be
choosing on test results.

## Not measured yet, and how to measure it

| Gap | Command |
|---|---|
| Clean wideband files (a studio clone uploaded as wav), with and without the telephone front end | `python -m src.inference.any_audio_eval build --out experiments/any_audio/clean.csv`, then `... score experiments/any_audio/clean.csv --out experiments/any_audio/clean_scored.csv`, then `... report experiments/any_audio/clean_scored.csv` |
| Speech straight into a laptop microphone, and clones played into it | record with the page, open "Help measure accuracy", label each recording; then `... score data/field_validation/log.csv --out experiments/any_audio/field_scored.csv` and `... report` |

The clean manifest covers 3,306 clips (dev and seven test sets). The front end is chosen on
the 506 development clips only, then the test rows are reported. Saved recordings stay in
`data/field_validation/` (ignored by git) and are kept only when the speaker's consent is
ticked.

## Ways to raise accuracy (team decisions)

1. **Average two checkpoints.** The seed-2025 checkpoint catches 49.0 % of Chatterbox clones
   at its own development-chosen point (0.334) where seed 1234 catches 20.0 % at 0.580; the
   average of the two, with its own development-chosen point, is not yet measured here.
2. **Train on more generators.** Adding IndicF5 and Chatterbox clones of some speakers to
   training, and testing on other speakers, is the most direct fix for the weak rows. They
   would then no longer be unseen tools in the paper's sense, so the paper's numbers stay as
   they are and this becomes a separate demo model.
3. **Add spontaneous genuine speech to training** (decision D1 part b, done): retrained with
   150 spontaneous speakers, SSL-AASIST catches 36-52 % of IndicF5 and Chatterbox clones here
   instead of 4-13 %, with spontaneous speech still judged real 99 % of the time; one of its two
   trainings lets read genuine speech fall below the bar set for adoption (section above).
4. **Say "uncertain" between the two development-chosen points** (0.082 and 0.580) instead of
   forcing real or cloned; the files that land there are where unscripted speech and the
   newest generators overlap.
