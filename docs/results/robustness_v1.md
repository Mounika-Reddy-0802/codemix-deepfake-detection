# Robustness to post-processing (v1): the plan, written before scoring

Written on 1 November 2026, before any clip was transformed or scored. The transforms, the
clips, the models, the measures and the reading rules below are fixed now. Results go under
them once the sweep has run; nothing above the results section changes after that, except
to record a deviation, stated as one.

## The question

"What if the attacker adds noise or re-encodes the clone as MP3?" Today the only answer is
a five-clip perturbation probe (`interpretability_v1.md`). This sweep answers it on whole
sets: which processing helps a clone pass, and which makes a genuine caller look cloned.

## The transforms

`src/data/robustness.py` (`TRANSFORMS`). Each is applied where it happens in a call:
**before** the line, on the caller's side (what an attacker or the caller's room does to the
voice), or **after** it, on the receiver's side (what the receiving phone does).

| Transform | Where | What |
|---|---|---|
| `mp3_32k` | before | MP3 at 32 kbit/s |
| `mp3_64k` | before | MP3 at 64 kbit/s |
| `opus_12k` | before | Opus at 12 kbit/s, an internet-call codec |
| `white_10db` | before | white noise at 10 dB SNR |
| `white_20db` | before | white noise at 20 dB SNR |
| `babble_10db` | before | six other voices at 10 dB SNR |
| `babble_20db` | before | six other voices at 20 dB SNR |
| `reverb` | before | a synthetic room, RT60 0.5 s (direct sound, then decaying noise from 5 ms) |
| `tempo_up_5` | before | 5% faster, same pitch (ffmpeg `atempo`) |
| `tempo_down_5` | before | 5% slower, same pitch |
| `pitch_up_1` | before | one semitone higher, same tempo (ffmpeg `asetrate`, `atempo`) |
| `pitch_down_1` | before | one semitone lower, same tempo |
| `aac_recorder` | after | the handset's call recorder: 8 kHz AAC at 48 kbit/s, as on every real call |
| `g711_second` | after | a second G.711 leg, as when a call is forwarded |

Every clip, with or without a transform, goes through the same telephone line as the
project's channel bundle: G.711 with noise at 20 dB SNR, the noise seeded by the clip's
name. With no transform (`none`) the clip is therefore exactly the bundle's clip, which
`tests/test_robustness.py` checks, so every change in a score is the transform's. Noise,
room and babble are seeded by the clip and the transform, so a re-run is identical.

Babble is six genuine voices, each from a different speaker of the training pool
(`codemix_adapt_train.csv`), never the clip's own speaker.

## The clips

`data/manifests/robustness_plan.csv`, written by `python -m src.data.robustness plan`: a
label-stratified subset of each set, seed 1234 (`stratified_subset`).

| Set | Source | Genuine | Cloned |
|---|---|---:|---:|
| New speakers (never trained on) | `score_ext.csv` | 300 | 222 XTTS-v2 (the seen tool) and 222 Tortoise (unseen) |
| IndicF5 (unseen tool) | `score_cm07.csv` | 300 | 300 |

1,344 clips under 15 conditions (`none` and the 14 transforms): 20,160 clips per model. They
are built from evaluation speakers, so the training guard would reject any of them in a
training manifest.

## The models

The final recipe's adapter, seed 1234 (`final_s1234_best.pt`), and SSL-AASIST, seed 1234
(the retrained checkpoint scored on the real calls). Each clip gets one P(genuine), scored the
way every offline set is.

## The measures

For each model, set and condition:

1. **EER**, with a 95% interval from 1,000 resamples of whole speakers.
2. **The change in EER from `none`** on the same clips, with a paired speaker bootstrap
   (1,000 resamples).
3. **At the operating rule** (a clip is called fake when P(genuine) < 0.5, the rule of the
   language-control and HiACC results; SSL-AASIST's scorer writes log-odds, so for it the same
   rule is a score below 0): the share of genuine clips called fake (false alarms) and the
   share of clones passed as genuine, and their change from `none`, paired the same way.

## The reading rules (fixed now)

- A transform **helps an attacker** on a set if the share of clones passed rises and the
  paired 95% interval of that change lies above zero.
- A transform **hurts a genuine caller** if the false-alarm share rises and its interval lies
  above zero.
- Otherwise it is reported as no measurable effect, with its interval; a wide interval is
  reported as wide, not as safe.
- The EER change is reported for every cell; the two rules above are what the summary says.
- Nothing is chosen on these results: no threshold, model, transform or front end.

These rules are code before any score exists: `src/reporting/robustness.py` (tested in
`tests/test_robustness_report.py` on made-up scores) writes the summary and the table below
from the two scored manifests.

## Where it runs

The clean clips live on Kaggle, so rendering (CPU) and scoring (one T4) run there: about two
GPU-hours for both models. `python -m src.data.robustness render` makes every clip and a
manifest with a `transform` column; it is resumable, like the channel bundle.

## Known before scoring

- One seed per model. The question is the direction and size of each transform's effect;
  seed-to-seed variation is measured elsewhere (`seed_stability_v1.md`).
- Synthetic rooms and noise stand in for real ones; the voice check's field recordings
  (`docs/field_validation/PROTOCOL.md`, a second batch with music in another room) are the
  real-room check.
- Tempo and pitch use ffmpeg's filters, which leave their own processing traces; a detector
  that reacts to them reacts to those traces as much as to the change itself.

## Results

Not scored yet.
