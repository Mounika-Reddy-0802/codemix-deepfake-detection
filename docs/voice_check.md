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
5. average P(genuine) over the windows and compare it with the frozen threshold (0.073 for
   voice check v2, chosen on development calls only by the unchanged rule; the development
   calls include 20 spontaneous IndicVoices speakers since decision D1 part a).

A replayed recording of a real person counts as **real**: the question is whether the voice
was cloned, not whether it was played back.

## How accurate it is today: voice check v2 (since 8 November)

Since 8 November the page runs **voice check v2** (P-042): SSL-AASIST trained on the
spontaneous recipe plus IndicF5 and Chatterbox clones of 31 speakers, seed 1234 at 0.073. It
replaced the detector below under a rule written before v2 was trained; both seeds met all five
conditions (`docs/results/voice_check_v2.md`). Percent judged correctly, from stored scores of
the exact checkpoint, with the 31 training speakers left out of both columns
(`python -m src.reporting.voice_check_v2_table`; full table in
`experiments/results/voice_check_v2_table.md`).

| Audio | v2 at 0.073 | Before 8 November (0.082) |
|---|---|---|
| Genuine read speech, telephone line | 100.0 % of 1,480 | 99.9 % |
| Genuine **spontaneous** speech, telephone line | 99.8 % of 1,000 | 99.3 % |
| Genuine live mobile calls / replayed into a phone | 12 of 12 / 6 of 6 | 12 of 12 / 6 of 6 |
| XTTS-v2 clones (line and AMR-NB codec) | 100.0 % | 99.0–99.7 % |
| XTTS-v2 clones replayed into a phone | 4 of 6 | 5 of 6 |
| Tortoise clones (in no training set): new speakers / CM04 | 94.1 % / 91.9 % | 80.0 % / 83.7 % |
| IndicF5 clones (held-out speakers): line / AMR-NB | 89.3 % / 68.0 % | 12.3 % / 1.6 % |
| Chatterbox clones (held-out speakers) | 93.8 %; 0 of 6 replayed into a phone | 4.2 %; 0 of 6 |
| RVC voice conversion (held-out speakers) | 73.5 % | 45.4 % |

**Strong:** real voices of every kind, and XTTS-v2, Tortoise (which no training set holds),
IndicF5 and Chatterbox clones over the telephone line. **Weak:** RVC (73.5 %; the spontaneous recipe
without the IndicF5 and Chatterbox rows caught 85.8 %), IndicF5 through AMR-NB (68.0 %), and
clones played aloud into a phone: no model has caught a replayed Chatterbox clone. IndicF5 and
Chatterbox are now tools this model heard (other speakers), so the page must not be described
as catching clones of tools it never heard; only Tortoise shows that.

### The detector before 8 November

SSL-AASIST, final-recipe data, seed 1234, measured with every speaker
(`python -m src.reporting.any_audio_accuracy`; `experiments/results/any_audio_accuracy.md`).
Percent judged correctly at its frozen threshold, and in brackets at the earlier point chosen
on read-speech development calls only.

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

It was strong on real voices of every kind and on XTTS-v2 clones, and weak on every other
generator. The move from 0.580 to
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
choosing on test results. Voice check v2 (P-042) is that recipe plus the generator study's
training half; both of its seeds met their rule and it was adopted on 8 November.

## Clean wideband files (measured 3 November, detector before 8 November)

Voice check v2 on the same recordings, without its 31 training speakers
(`experiments/results/voice_check_v2_clean.md`): IndicF5 clones caught 72.7% and Chatterbox
86.1% where the detector below caught none, RVC 58.4% (20.1%), Tortoise 86.8% (71.1%), XTTS-v2
100%; genuine speech judged as before, with one of 1,455 genuine recordings (HiACC) called
cloned. The rest of this section is the detector before 8 November.

A file that never went through a phone line (a studio clone downloaded as wav, say) is the
case the page meets most often and the paper never measured. The clean copies of the dev and
test clips (3,306 recordings) were judged by the page's own scoring path at the frozen point
(0.082), once with the telephone front end and once without
(`python -m src.inference.any_audio_eval`; per-recording verdicts in
`experiments/any_audio/clean_scored.csv`, table in `experiments/results/any_audio_clean.md`).

**Front end, chosen on the 506 development clips only:** with it 97.4 % balanced accuracy,
without it 93.2 %, so the page keeps converting every input to the telephone domain. The test
rows below are with the front end.

| Clean test recordings | Judged correctly | Recordings |
|---|---|---|
| Genuine speech (read and HiACC spontaneous) | 100.0 % of those with 2 s of speech; 22.1 % too short to judge | 1,695 |
| XTTS-v2 clones | 96.7 % | 305 |
| Tortoise clones | 66.7 % | 81 |
| RVC conversions | 22.6 % | 154 |
| IndicF5 clones | 0 of 105 | 134 |
| Chatterbox clones | 0 of 130 | 134 |

**What it says.** No genuine recording was called cloned, with or without the front end; the
only genuine misses are clips with under two seconds of speech, which the page declines to
judge. Clones of the tool it was trained on are still caught. Every clean clone of the two
newest tools passes as real, worse than over the phone line (12.5 % and 4.4 %): without the
line's distortion, IndicF5 and Chatterbox clones sit with genuine speech (IndicF5 EER 25.8 %
threshold-free). This is the gap the generator study (P-041) is aimed at, and the reason the
page must not be described as catching clones of new tools.

The run took several restarts on the laptop; recordings scored twice got identical scores in
every case, and the table counts each once.

## Not measured yet, and how to measure it

| Gap | Command |
|---|---|
| Speech straight into a laptop microphone, and clones played into it | record with the page, open "Help measure accuracy", label each recording; then `python -m src.inference.any_audio_eval score data/field_validation/log.csv --out experiments/any_audio/field_scored.csv` and `... report` |

Saved recordings stay in `data/field_validation/` (ignored by git) and are kept only when the
speaker's consent is ticked.

## Ways to raise accuracy (team decisions)

1. **Average two checkpoints.** The seed-2025 checkpoint catches 49.0 % of Chatterbox clones
   at its own development-chosen point (0.334) where seed 1234 catches 20.0 % at 0.580; the
   average of the two, with its own development-chosen point, is not yet measured here.
2. **Train on more generators (done: voice check v2, adopted 8 November).** IndicF5 and
   Chatterbox clones of 31 speakers in training, tested on 30 others: 89-94 % of their clones
   caught over the line, at a cost of 11-12 points on RVC. They are no longer unseen tools for
   this model, so the paper's numbers stay as they are.
3. **Add spontaneous genuine speech to training** (decision D1 part b, done): retrained with
   150 spontaneous speakers, SSL-AASIST catches 36-52 % of IndicF5 and Chatterbox clones here
   instead of 4-13 %, with spontaneous speech still judged real 99 % of the time; one of its two
   trainings lets read genuine speech fall below the bar set for adoption (section above).
4. **Say "uncertain" between the two development-chosen points** (0.082 and 0.580) instead of
   forcing real or cloned; the files that land there are where unscripted speech and the
   newest generators overlap.
