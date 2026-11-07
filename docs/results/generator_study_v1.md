# The generator study: does training on more cloning tools catch a tool never heard? (v1)

**Status: run on 7 November 2026. Answer: partly** (H-B confirmed, H-A not confirmed; see
[Results](#results)). Everything above the Results section was written on 3 November 2026
before any training or scoring of these systems, so the hypotheses and the decision rule
cannot be fitted to the results.
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

All 20 trainings finished (both Kaggle parts: "every step ran without error", every run
status `ok`). Every number below is on speakers outside `gen_train`. The rule's own output is
`experiments/results/generator_study_v1.json`, the per-clip scores are in
`experiments/results/generator_study_v1/` (reproduce with
`python -m src.reporting.generator_study --results experiments/results/generator_study_v1`),
the tables come from `src/reporting/genstudy_tables.py` and the figure is
`experiments/figures/generator_study.png`.

### The answer: partly

| Hypothesis | Unseen tool | Control → fold, mean EER | Lower in | p | Tortoise | Genuine called fake | Decision |
|---|---|---|---|---|---|---|---|
| H-A: fold A (+ IndicF5) | Chatterbox | 24.32 → **15.75** | 5 of 5 | 0.008 | −3.04 points | **+0.67 points** (limit +0.50) | **not confirmed** |
| H-B: fold B (+ Chatterbox) | IndicF5 | 20.44 → **12.43** | 5 of 5 | 0.016 | −2.17 points | +0.48 points | **confirmed** |

- **H-B is confirmed.** A detector that also trained on Chatterbox clones has lower EER on
  IndicF5, a tool it never heard, in all five seeds (mean 20.44% → 12.43%, 39% lower), and
  it is better, not worse, on Tortoise and within the false-alarm limit.
- **H-A is not confirmed.** A detector that also trained on IndicF5 clones has lower EER on
  Chatterbox in all five seeds (24.32% → 15.75%, 35% lower; p = 0.008 is the smallest p five
  seeds against five can give) and is better on Tortoise, but it calls more genuine
  language-control clips fake than the rule allows: 1.01% against 0.34%, about 13 of the 1,290
  clips instead of about 4. The limit was fixed before the run, so the hypothesis is not
  confirmed; the drop on Chatterbox is reported as measured, not as a confirmed carry-over.

Per seed (EER %, unseen tool, held-out speakers):

| Seed | Chatterbox: control | Chatterbox: fold A | IndicF5: control | IndicF5: fold B |
|---|---|---|---|---|
| 1234 | 22.73 | 13.50 | 19.29 | 18.79 |
| 2025 | 24.56 | 16.48 | 20.97 | 8.63 |
| 7 | 23.88 | 17.64 | 22.16 | 12.69 |
| 42 | 24.56 | 15.47 | 22.16 | 11.01 |
| 99 | 25.85 | 15.67 | 17.60 | 11.01 |

### Every measure (reported, held-out speakers)

EER (%), mean ± SD over the five seeds. A tool a system trained on is marked *seen*; the
rest are tools that system never heard.

| Measure | control | fold A (+ IndicF5) | fold B (+ Chatterbox) | fold C (+ both) |
|---|---|---|---|---|
| Chatterbox, G.711 | 24.32 ± 1.14 | 15.75 ± 1.52 | 7.80 ± 4.42 *seen* | 7.04 ± 1.34 *seen* |
| IndicF5, G.711 | 20.44 ± 1.97 | 4.19 ± 1.71 *seen* | 12.43 ± 3.84 | 4.10 ± 0.55 *seen* |
| IndicF5, AMR-NB | 19.40 ± 2.13 | 8.23 ± 3.03 *seen over G.711* | 14.31 ± 4.56 | 7.41 ± 1.47 *seen over G.711* |
| Tortoise, new speakers, G.711 | 7.16 ± 1.98 | 4.13 ± 1.85 | 5.00 ± 2.76 | 4.91 ± 2.31 |
| Tortoise, new speakers, AMR-NB | 4.05 ± 1.24 | 4.89 ± 2.71 | 3.53 ± 0.66 | 4.01 ± 2.79 |
| CM04 Tortoise, G.711 | 13.67 ± 2.13 | 11.84 ± 2.67 | 10.40 ± 3.67 | 12.03 ± 4.50 |
| CM04 Tortoise, AMR-NB | 4.73 ± 1.45 | 4.60 ± 2.43 | 4.16 ± 1.18 | 5.33 ± 4.62 |
| RVC test half | 7.72 ± 0.74 | 6.88 ± 1.01 | 7.75 ± 3.57 | 7.08 ± 1.04 |
| XTTS-v2, new speakers, G.711 | 0.74 ± 0.44 | 0.57 ± 0.30 | 0.75 ± 0.33 | 1.04 ± 0.71 |
| Code-mixed test pool (XTTS-v2) | 0.53 ± 0.21 | 0.81 ± 0.30 | 0.36 ± 0.15 | 0.79 ± 0.49 |
| Language-control genuine called fake | 0.34 ± 0.45 | 1.01 ± 0.61 | 0.82 ± 0.50 | 0.64 ± 0.51 |

- **Tortoise, never in any training set, gets easier for every fold over G.711** (7.16% →
  4.13-5.00%) and on CM04 (13.67% → 10.40-12.03%); over AMR-NB, new-speaker Tortoise stays
  level (4.05% for the control, 3.53-4.89% for the folds).
- **Over AMR-NB the carry-over to IndicF5 is smaller**: fold B lowers it from 19.40% to
  14.31% (G.711: 20.44% → 12.43%). The folds' extra clones are all over G.711 (Chatterbox
  exists only there), so the AMR-NB rows measure a change of codec as well as of tool.
- **Fold C** (both tools; reported, not judged) has the lowest error on both of its seen
  tools on new voices (Chatterbox 7.04%, IndicF5 4.10%), lower than the control on RVC and on
  Tortoise over G.711, within 0.3 points of it on XTTS-v2, and 0.64% genuine called fake
  (+0.30 points). It enters the voice
  check only under a rule written before it is measured there; the voice check's own model
  is trained on the same rows under P-042 (`voice_check_v2.md`).
- **The development set cannot see any of this.** Over the 100 epochs of the 20 runs, dev
  EER is 0.00-3.18% (best epoch 0.00-0.58% in every run), while the unseen-tool EER of the
  same runs is 8.6-25.9%. Held-out tools are what separate the systems, as the design
  expected.

### Shortcut floors

Every system beats its matched floor (eight low-level statistics fitted on that system's own
training manifest) on every study set, in **5 of 5 seeds** each, with the whole 95% interval
below zero (`docs/results/floor_genstudy_v1.md`, `experiments/results/floor_genstudy_v1.json`).
The floors sit at 38-53% EER, so the clones of these tools are not separable by the eight
low-level statistics, and the drops above are not explained by such a cue.

### What it means

- **More tools do carry over, with a cost to watch.** In both directions, adding one tool's
  clones of other speakers lowered the error on a tool the detector never heard by about 8
  points, in every seed, and lowered it on Tortoise too. One direction meets every condition
  fixed in advance; the other meets the error conditions but pays in genuine speech called
  fake. The answer the rule gives, and the one to quote, is **partly**.
- **It answers the review's comment 2** with a controlled leave-one-tool-out experiment: one
  tool changed at a time, against a control with the same extra genuine speech, on speakers
  no system heard, with the decision fixed before the runs.
- **Limits.** Two tools, one codec for the judged numbers, 30 held-out speakers, five seeds.
  The false-alarm cost was measured on one genuine set (the language control); spontaneous
  genuine speech was not part of this study.
