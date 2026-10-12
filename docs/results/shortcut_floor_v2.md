# Shortcut floor v2: matched floors and a paired speaker-level test

Answers external review comment 5: *explain how the shortcut-floor results should be
interpreted, and how residual recording-domain differences affect the reliability of
detector-versus-floor comparisons.*

Code: `src/training/floor_compare.py` (tests: `tests/test_floor_compare.py`).
Results: `experiments/results/floor_compare_v2.json`, per-clip paired scores and
per-speaker EERs in `experiments/results/floor_v2/`.

## 1. What was wrong with the comparison

Checking the comment against the code turned up two problems.

1. **The CM04 floor and the detector saw different information.** The CM04 floor was
   fitted on 1,066 clips of CM04 itself, 237 Tortoise fakes included, and scored on
   the other 958 (`score_cm04_norm_*_fit.csv` / `_score.csv`). The detector never sees
   Tortoise and is scored on all 2,024 clips. `methodology.tex` says the floor is
   fitted "on the adaptation pool", which is true for the eval pool and not for CM04.
   On the floor's own 958 clips the deployed model scores 26.78%, against the floor's
   25.79% [21.92, 29.03]: level, not worse.
2. **No paired test, and intervals over clips.** Detector and floor were compared by
   eye, and every interval resampled clips. The test sets hold 12-15 speakers, and a
   speaker's clips are not independent, so clip-level intervals are too narrow.

## 2. What v2 does

The definition was fixed from the methodology text before any v2 number was seen:

- **Matched floor.** For every detector, the same eight statistics and the same
  logistic regression, fitted on *that detector's own training manifest* and scored
  on *exactly the clips the detector was scored on*. The two sides then differ only
  in what they can hear.
- **Paired speaker bootstrap.** 1,000 resamples of speakers with replacement; each
  resample takes all of a drawn speaker's clips for both the detector and the floor,
  and records EER(detector) − EER(floor). The verdict reads that interval: whole
  interval below zero = *beats floor*; above zero = *worse than floor*; straddles
  zero = *inconclusive*.

Check: for the XTTS-trained systems the matched eval-pool floor is the original
eval-pool floor (same fit set, same clips), and v2 reproduces it exactly: 5.17%
(clean) and 10.01% (channel).

## 3. Results

EER (%). Floor = matched floor. Δ = detector − floor with its 95% speaker-bootstrap
interval. Old floor = the v1 floor in the paper.

| System | Set | Detector | Floor | Δ [95% CI] | Verdict | Old floor |
|---|---|---|---|---|---|---|
| S2 XTTS clean | eval pool | 1.46 | 5.17 | −3.71 [−7.16, −0.24] | beats | 5.17 |
| | CM04 | 5.44 | 29.49 | −24.05 [−32.65, −14.75] | beats | 31.21 |
| | RVC test | 23.91 | 17.93 | +5.98 [−3.38, +19.84] | inconclusive | 28.76 |
| S2 XTTS channel | eval pool | 4.41 | 10.01 | −5.60 [−12.14, +4.05] | inconclusive | 10.01 |
| | CM04 | 28.16 | 33.64 | −5.48 [−19.43, +5.83] | inconclusive | 25.79 |
| | RVC test | 29.56 | 23.91 | +5.65 [−3.45, +19.70] | inconclusive | 22.46 |
| S2 XTTS+RVC clean | eval pool | 1.99 | 5.62 | −3.63 [−7.76, −0.21] | beats | 5.17 |
| | CM04 | 12.01 | 31.43 | −19.42 [−29.51, −8.97] | beats | 31.21 |
| | RVC test | 5.01 | 16.96 | −11.95 [−18.18, −0.87] | beats | 28.76 |
| **S2 XTTS+RVC channel (deployed)** | eval pool | 2.75 | 9.96 | −7.21 [−12.92, −1.10] | **beats** | 10.01 |
| | **CM04** | 30.99 | 37.79 | −6.80 [−22.32, +8.74] | **inconclusive** | 25.79 |
| | RVC test | 5.01 | 20.52 | −15.51 [−23.28, −6.08] | **beats** | 22.46 |
| S3 XTTS clean | eval pool | 1.46 | 5.17 | −3.71 [−6.78, −0.96] | beats | 5.17 |
| | CM04 | 27.72 | 29.49 | −1.77 [−15.31, +11.38] | inconclusive | 31.21 |
| | RVC test | 45.07 | 17.93 | +27.14 [+17.38, +39.68] | **worse** | 28.76 |
| S3 XTTS channel | eval pool | 1.92 | 10.01 | −8.10 [−14.55, −1.29] | beats | 10.01 |
| | **CM04** | 13.71 | 33.64 | **−19.93 [−32.95, −8.91]** | **beats** | 25.79 |
| | RVC test | 32.15 | 23.91 | +8.24 [−0.17, +19.97] | inconclusive | 22.46 |
| S3 XTTS+RVC clean | eval pool | 0.83 | 5.62 | −4.79 [−8.28, −1.61] | beats | 5.17 |
| | CM04 | 21.39 | 31.43 | −10.03 [−24.66, +5.97] | inconclusive | 31.21 |
| | RVC test | 8.24 | 16.96 | −8.72 [−15.77, +2.98] | inconclusive | 28.76 |
| S3 XTTS+RVC channel | eval pool | 1.79 | 9.96 | −8.17 [−14.17, −1.87] | beats | 10.01 |
| | CM04 | 31.67 | 37.79 | −6.12 [−21.32, +9.48] | inconclusive | 25.79 |
| | RVC test | 5.98 | 20.52 | −14.54 [−23.34, −3.81] | beats | 22.46 |

## 4. What it means

1. **The CM04 floor depends on what it was trained on.** Fitted on Tortoise itself
   (v1), the eight statistics reach 25.79% over the phone line. Fitted on the same
   XTTS/RVC data the detector learns from, they reach only 33.64-37.79%: provenance
   cues learned from XTTS do not transfer to Tortoise either. v1 is best read as an
   *oracle* floor (what recording cues achieve with labelled examples of the unseen
   tool); v2 is the fair, like-for-like reference. Report both.
2. **The deployed model does not lose to the shortcut on Tortoise.** It is 6.80 points
   below its matched floor, but the speaker interval [−22.32, +8.74] includes zero:
   inconclusive, not a failure. It clearly beats its floor on the seen tool and on RVC.
3. **Only one phone-line system clearly beats the floor on the unseen tool:** S3
   trained on XTTS only, 13.71% against 33.64%, Δ −19.93 [−32.95, −8.91]. Adding RVC to
   its training removes that advantage (31.67%, inconclusive).
4. **Speaker-level uncertainty is large.** With 15 test speakers, CM04 intervals span
   20-30 points. Several cells the paper marks as evidence (bold) are inconclusive at
   the speaker level, and one is significantly worse than its floor (S3 XTTS clean on
   RVC, +27.14). Claims must be made at the speaker level.
5. **Per speaker, the spread is wide.** The deployed model's CM04 EER ranges from 6.4%
   to 45.5% across the 15 speakers (median 28.1%); 6 of them are worse than their own
   floor. On the seen tool it ranges 0.0-11.0% (median 0.7%). (RVC test: 6 of its 12
   speakers hold both classes, so per-speaker EER exists for 6.)

## 5. How to read the floor (text for the paper)

The floor is the EER that eight cheap recording statistics reach on a test set when
trained on the same data as the detector. It is a reference, not a bound on what is
achievable: a detector *below* its floor uses information beyond those statistics,
which rules out the cheapest explanation for its score but does not prove it ignores
recording provenance, since a large encoder can exploit subtler domain cues (room,
microphone, noise, pauses) that the eight statistics do not measure. A detector *at
or above* its floor is not shown to rely on the shortcut either; its result simply
carries no evidence of learning beyond it. Both judgements are made on the paired,
speaker-level interval of the difference, never by comparing two separately computed
EERs.

## 6. Remaining differences between the two sides

- The floor reads the whole clip (2-10 s); the detector reads one 4 s crop.
- The eight statistics do not measure reverberation, microphone response, noise
  spectrum or pause structure (`lowlevel_cue_check_v1.md`, `bundle_normalisation_v1.md`).
- A bootstrap over 12-15 speakers is itself approximate; more test speakers (comment
  3) would narrow every interval in the table.

## Reproduce

```
python -m src.training.floor_compare --clean-root C:/dfdata/lora_bundle_norm --channel-root C:/dfdata/lora_bundle_norm_ch20 --n-boot 1000
```
About 25 minutes on the laptop CPU (feature extraction over 23,234 clips).
