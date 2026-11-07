# The generator study: does training on more cloning tools catch a tool never heard? (v1)

**Status: registered, not run.** Written on 3 November 2026 before any training or scoring
of these systems, so the hypotheses and the decision rule cannot be fitted to the results.
Team decision P-041. The rule is coded in `src/reporting/generator_study.py` and committed
with this page.

## Why

Every unseen-tool result in the project comes from detectors trained on XTTS-v2 (and RVC).
The final recipe still misses IndicF5 (CM07, 21.68 ± 3.74% EER over the phone line) and
Chatterbox, and the voice check catches 12.5% and 4.4% of their clones. The question the
field asks of any detector in this position, and the one the external review's comment 2
points at, is whether **more generators in training carry over to a generator the
detector has never heard**. Week 15's variant test showed what does not work (more
augmentation on the same tool fits the training tool more closely); this tests the other
lever, more tools.

A second reason: the development set reads 0.00% EER from the second or third epoch in
every recent run, so it cannot tell a model that generalises from one that memorises
XTTS-v2. Held-out tools are the only way to see that difference, and this study measures it
directly.

## Design

IndicF5 and Chatterbox cloned test speakers only (15 of the `eval` pool and 46 of
`eval_ext`). So, as for RVC (P-025), the speakers are split:

- the 61 cloned speakers are cut into two frozen halves, alternated over sorted ids within
  each pool: `gen_train` (8 eval + 23 eval_ext = 31) and `gen_test` (7 + 23 = 30), in
  `data/manifests/generator_pools.csv` (sha256 `8174f544…`, checked by the tests);
- a study model trains on the `gen_train` half only: its clones and its genuine clips
  (608, both codecs), so those voices are not "always fake" in training;
- **every number of this study is computed on speakers outside `gen_train`**, for every
  system and every set, so no test voice was heard in training.

| System | Training data | Unseen tool |
|---|---|---|
| control | final recipe + the 608 genuine clips | both |
| fold A | control + 282 IndicF5 clones (G.711) | Chatterbox |
| fold B | control + 265 Chatterbox clones (G.711) | IndicF5 |
| fold C | control + both | none (the demo model) |

Clones enter over G.711 only, because Chatterbox exists only there, so folds A and B each
add one tool in the same condition. **Tortoise is in no training set** (the guard refuses
it in every run), so new-speaker Tortoise stays a never-trained check. Everything else is
the final recipe: start, learning rate, epochs, dev set, the five seeds
(1234, 2025, 7, 42, 99). Nothing is tuned.

The control, not the final recipe, is the comparison: it carries the same extra genuine
speech, so a fold differs from it only by one tool's clones.

## Hypotheses (fixed now)

- **H-A.** Fold A (trained on IndicF5) has lower EER on Chatterbox (CM09, G.711) than the
  control.
- **H-B.** Fold B (trained on Chatterbox) has lower EER on IndicF5 (CM07, G.711) than the
  control.

## Decision rule (fixed now)

A fold's hypothesis is **confirmed** only if all four hold against the control's five
paired seeds:

1. lower EER on its unseen tool in at least 4 of 5 seeds;
2. exact two-sided Mann-Whitney p ≤ 0.025 (0.05 split over the two hypotheses; the smallest
   possible p for five against five is 0.008);
3. new-speaker Tortoise (G.711) mean EER at most 1.0 point higher;
4. genuine language-control clips called fake at most 0.5 points more often (mean).

The study's answer is **carries over** if both are confirmed, **partly** if one is,
**does not carry over** if neither, and **incomplete** if any of control, fold A or fold B
has fewer than five finished seeds. Whatever the answer, every number is reported.

**Reported, not judged:** fold C on every set (the candidate demo model); IndicF5 over
AMR-NB; CM04; RVC; XTTS-v2; development EER per epoch. Fold C enters the voice check only
under a separate rule written before it is measured there (as P-040 required).

## Run

Five seeds of each of the four systems (20 trainings), scored on every week-13 set plus
Chatterbox (CM09), all at the registration commit `a66b2a8`; then:

```
python -m src.reporting.generator_study --results <results folder with both parts>
```

**First attempt, AWS (3-4 November).** One g5.2xlarge (A10G) started the 20 trainings on
3 November at 21:54 with `notebooks/aws_w15_generator_study.py` (22 minutes a training). The
account was suspended before the results could be fetched, so none of them are used; the
only number seen was the first run's summary line over all speakers, and nothing in the
design or the rule changed after it.

**Run used, Kaggle (from 7 November).** The same commit in two notebooks of ten trainings
each, started together when the weekly GPU quota refreshed:
`notebooks/kaggle_w15_genstudy_a.ipynb` (control, fold A and fold B for seeds 1234, 2025 and
7, then fold C seed 1234) and `kaggle_w15_genstudy_b.ipynb` (the three judged systems for
seeds 42 and 99, then fold C for the other four seeds). The judged systems run first in both,
so the decision needs only the first nine and six trainings of each part.

**Corrections before any result (7 November, 09:42).** An independent read of the code before
the runs finished found two faults in how results are read, not in the rule: the two Kaggle
parts each write a `runs.json`, so `src/reporting/generator_study.py` now reads one folder per
part and refuses a run found in both; and a run whose only failure was an unjudged set (status
`partial`) is now read, since the rule itself already checks that every quantity it judges is
present. The hypotheses, thresholds and decision are unchanged.

## Results

Not run yet.
