# The generator study on SSL-AASIST (v1)

**Status: run on 10 November 2026. Answer: carries over** — both folds are confirmed in both
seeds. The rule was written on 8 November 2026 before any of its six trainings, so it could not
be fitted to their results. Team decision P-044. The rule is coded in
`src/reporting/ssl_generator_study.py` and committed with this page; the notebooks run this
commit and stop if they find another.

## Why

The generator study (P-041, `generator_study_v1.md`) asked whether training on one more cloning
tool catches a tool the detector never heard. With the final recipe's adapter the answer was
**partly**: trained also on Chatterbox, its EER on IndicF5 fell from 20.44% to 12.43% in all
five seeds (confirmed); trained also on IndicF5, its EER on Chatterbox fell from 24.32% to
15.75% in all five but genuine clips called fake rose 0.67 points against 0.5 allowed (not
confirmed).

That is one model. Voice check v2 (P-042, P-043) then trained the stronger detector, SSL-AASIST,
on **both** tools, so for it IndicF5 and Chatterbox are seen tools and its numbers on them say
nothing about tools it never heard. This study asks the generator study's question of SSL-AASIST
itself: does a tool it heard help it catch the other, which it did not? If both models answer
alike, the finding is about the data, not one architecture.

## What is trained

SSL-AASIST exactly as voice check v2 (RawBoost, whole network, 3 epochs, learning rate 1e-5,
batch 8, best dev epoch kept, the spontaneous recipe's dev set), seeds 1234 and 2025, on the
spontaneous recipe plus one system's rows of the generator study's training half
(`src/data/ssl_generator_study.py`, `data/manifests/codemix_adapt_train_sslgs_*.csv`):

| System | Added to the spontaneous recipe | Never heard |
|---|---|---|
| control | the 31 `gen_train` speakers' genuine clips (608, both codecs) | IndicF5, Chatterbox |
| fold A | those clips + their IndicF5 clones (282, G.711) | Chatterbox |
| fold B | those clips + their Chatterbox clones (265, G.711) | IndicF5 |
| fold C | both: voice check v2, already trained (reported, not judged) | none of the two |

The control carries the same extra genuine speech, so a fold differs from it only by one tool's
clones. Tortoise is in no training set. The guard checks each manifest through
`configs/train_lora_sslgs_*.yaml`, as `train_lora_vcv2.yaml` does for v2.

## Decision rule (fixed now)

Every number is an EER from per-clip scores on the G.711 line, **on speakers outside
`gen_train`**, for the control and the folds alike (`generator_study.held_out_eer`). Fold A's
measure is Chatterbox (CM09), fold B's is IndicF5 (CM07).

Two seeds cannot carry the exact rank test the adapter study used (the smallest possible p for
two against two is 0.33). The uncertainty from which speakers happen to be in the test half is
taken instead from a speaker-level bootstrap: in each seed, 10,000 resamples of the held-out
speakers with replacement (generator seed 20261108), the EER difference fold minus control
recomputed on the same resampled clips. A fold's hypothesis is **confirmed** only if all four
hold:

1. lower EER on its never-heard tool than the control, in both seeds;
2. in both seeds, the 95% bootstrap interval of that difference lies wholly below zero;
3. new-speaker Tortoise mean EER at most 1.0 point higher than the control's;
4. genuine language-control clips called fake at most 0.5 points more often (mean).

Conditions 3 and 4 are the adapter study's, unchanged. Answer: **carries over** if both folds
are confirmed, **partly** if one is, **does not carry over** if neither, **incomplete** if any of
the six runs is missing. Whether each fold's decision matches the adapter study's is reported.

**What it cannot show.** Two seeds sample training variance only twice; the bootstrap covers
the test speakers, not the training. A confirmed fold here is weaker evidence than the adapter
study's five-seed result, and the page will say so; an unconfirmed fold is reported as measured.

## Run

Six Kaggle notebooks, one training each (`notebooks/kaggle_w15_sslgs_*`), about 6 GPU hours
each, three on each of two team members' Kaggle accounts with the project's datasets shared to
them. Then, from the downloaded run folders (the voice check v2 folders give fold C):

```
python -m src.reporting.ssl_generator_study --results <run folders> \
    experiments/results/voice_check_v2/ssl_s1234 experiments/results/voice_check_v2/ssl_s2025
```

## Results

**Answer: carries over.** `python -m src.reporting.ssl_generator_study` on 10 November
(`experiments/results/ssl_generator_study_v1.json`). All six trainings finished with every step
clean, from the registered commit (`0fbf142`). EER (%) on the G.711 line, on speakers outside
`gen_train`; the bootstrap resamples the 44 held-out speakers of each set (607 Chatterbox and 591
IndicF5 clips, genuine and cloned together) 10,000 times per seed.

| Fold (never heard) | Seed | Control | Fold | Fold − control [95% interval] |
|---|---|---|---|---|
| A: + IndicF5 clones (Chatterbox) | 1234 | 12.35 | 2.31 | −10.04 [−12.97, −7.63] |
| | 2025 | 10.72 | 4.28 | −6.44 [−9.13, −2.51] |
| B: + Chatterbox clones (IndicF5) | 1234 | 24.69 | 7.29 | −17.40 [−21.33, −13.70] |
| | 2025 | 17.25 | 11.15 | −6.10 [−10.67, −1.22] |

| Condition | Fold A | Fold B |
|---|---|---|
| 1. Lower on the never-heard tool, both seeds | yes | yes |
| 2. 95% interval wholly below zero, both seeds | yes | yes |
| 3. Tortoise (new speakers), mean rise ≤ 1.0 point | −0.76 (1.50 to 0.74) | −1.03 (1.50 to 0.47) |
| 4. Language-control genuine called fake, mean rise ≤ 0.5 points | +0.08 (0.12 to 0.19) | −0.12 (0.12 to 0.00) |
| **Decision** | **confirmed** | **confirmed** |
| Adapter study (P-041, five seeds) | not confirmed (false alarms +0.67) | confirmed |

The paper's one-column figure of both studies is drawn from the two result files by
`python -m src.reporting.genstudy_both_figure` (`experiments/figures/generator_study_both_column.{png,pdf}`,
kept out of git with the paper's other figures): each line is one seed, control to fold, on the
tool the fold never heard; left of each panel the adapter study, right this one.

**Every system, mean of two seeds** (EER %, held-out speakers; fold C is voice check v2, which
heard both tools, reported only):

| Set | Control | Fold A (+ IndicF5) | Fold B (+ Chatterbox) | Fold C (both) |
|---|---|---|---|---|
| Chatterbox (CM09) | 11.53 | **3.29** | 3.29 | 1.32 |
| IndicF5 (CM07), G.711 | 20.97 | 2.45 | **9.22** | 2.53 |
| IndicF5 (CM07), AMR-NB | 24.19 | 6.17 | 13.29 | 6.10 |
| Tortoise, new speakers | 1.50 | 0.74 | 0.47 | 0.10 |
| Tortoise, CM04 | 1.57 | 2.00 | 1.36 | 0.50 |
| RVC, held-out half | 6.14 | 5.49 | 5.82 | 4.28 |
| XTTS-v2, new speakers | 0.00 | 0.00 | 0.04 | 0.00 |
| Language-control genuine called fake | 0.12 | 0.19 | 0.00 | 0.04 |

Bold: the tool that fold never heard.

### What it means

- **Both directions carry over on SSL-AASIST.** Training on one of the two newest tools lowers
  the EER on the other, never heard, in both seeds and beyond the spread of the test speakers.
  The never-heard tool's EER falls by about 71% (Chatterbox, 11.53 to 3.29) and 56% (IndicF5,
  20.97 to 9.22). With the adapter the falls were 35% and 39%.
- **The direction the adapter study left unconfirmed is confirmed here.** The adapter's fold A
  lowered Chatterbox in all five seeds but called 0.67 points more genuine clips fake; on
  SSL-AASIST the same fold moves that rate by 0.08 points. So the two models agree that one
  tool's clones help catch the other; they differ on what that costs in false alarms.
- **How close a never-heard tool comes to a heard one.** Chatterbox never heard (fold A, 3.29%)
  is as low as Chatterbox heard (fold B, 3.29%). IndicF5 never heard (fold B, 9.22%) is still
  well above IndicF5 heard (fold A, 2.45%): the extra tool closes about two thirds of that gap,
  not all of it. Hearing both (fold C) is lowest on both.
- **Nothing else got worse beyond the guards.** Tortoise, which no system heard, stays at or
  under 2% in every system. RVC moves within a point.

### What it cannot show

- **Two seeds.** The bootstrap covers which test speakers were drawn, not training variance. The
  size of the gain differs a lot between seeds (fold B: 17.40 points in seed 1234, 6.10 in seed
  2025, whose interval ends at −1.22). The direction is shown in both seeds; the size is not
  pinned down. This is weaker evidence than the adapter study's five seeds.
- **Recipe as well as model.** This study adds the clones to the spontaneous recipe, the adapter
  study to the final recipe, so "the same answer from two detectors" also spans two training
  sets. Both studies share the frozen speaker halves and the measure.
- **Two tools.** IndicF5 and Chatterbox are both recent multilingual TTS systems cloning the
  same 61 speakers from the same references. The study shows that each helps with the other,
  not that any one tool helps with every other.
