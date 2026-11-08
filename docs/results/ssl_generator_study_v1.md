# The generator study on SSL-AASIST (v1)

**Status: registered, not run.** Written on 8 November 2026 before any of its six trainings,
so the rule cannot be fitted to their results. Team decision P-044. The rule is coded in
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

Not run yet.
