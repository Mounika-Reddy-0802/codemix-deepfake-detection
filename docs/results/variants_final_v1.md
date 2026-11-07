# Confirmatory test of the exploratory variant on the final recipe (v1)

**Status: run, not confirmed.** Written on 31 October 2026, before any training or
scoring of this variant, so the hypothesis and the decision rule cannot be fitted to its
results. Plan task A5; it runs only if GPU hours remain after tasks A1-A4. Committed on
3 November 2026 together with the notebook as launched (Kaggle `w15-variant-final`, started
15:46 that day), while the run was in progress and before any of its results existed. The
launched notebook adds one guard: a seed starts only with 75 minutes of the 6 GPU hours left
that week to spare, so fewer than five seeds can finish; the decision rule then reports the
result as incomplete rather than changing. The rule was coded
(`src/reporting/variant_final.py`) and committed before the results arrived; all five seeds
finished (19:51, about 4 GPU hours).

## Why

In week 12, three training options switched on together (a fresh crop every epoch,
on-the-fly channel augmentation, a learned mix of every encoder layer) lowered CM04
Tortoise from 35.59 ± 3.35% to 26.63 ± 1.66% EER for the deployed recipe over five seeds.
The variant was singled out after its CM04 score had been seen, so the paper reports it
as exploratory and a lead to confirm on another unseen tool
(`docs/external_review_response.md`, comment 2). This is that confirmation, on the final
recipe and on the unseen tools that are still open.

## What is trained

`configs/train_lora_final_variant.yaml`: the final recipe (`train_lora_final.yaml`) with
`crop_per_epoch: true`, `augment: channel_v1` and `layer_weighting: true`, nothing else
changed. Five seeds (7, 42, 99, 1234, 2025), the seeds of the final recipe's run 4, each
scored once on the eleven week-13 sets.

## Hypotheses (fixed now)

- **H1 (primary).** On IndicF5 over G.711 (CM07), the variant's EER is lower than the
  final recipe's (21.68 ± 3.74% over the same five seeds, `w13_runs_v1.md`).
- **H2 (no cost on the solved tool).** On new-speaker Tortoise over G.711, the variant's
  mean EER is no more than 1.0 point above the final recipe's (8.81 ± 1.43%).
- **Secondary, reported but not part of the decision:** IndicF5 over AMR-NB, CM04 over
  both codecs, RVC, false alarms on the language-control set, and Chatterbox. The adapter
  has a Chatterbox score for one seed only (31.98%), so Chatterbox is compared only if
  the final recipe's five checkpoints can be scored on it in the same run.

## Decision rule (fixed now)

The variant is called **confirmed** only if all three hold:

1. lower IndicF5 (G.711) EER than the final recipe in at least 4 of 5 paired seeds;
2. exact Mann-Whitney p ≤ 0.05 on those five-against-five EERs (the smallest possible
   p is 0.008), the test used in `w13_runs_v1.md`;
3. H2 holds.

Otherwise it is reported as **not confirmed**, with every number, and the week-12 lead is
closed. Nothing is tuned: no learning rate, epoch or threshold changes, and the dev set
is the final recipe's (in-domain, so it is not expected to separate the two recipes;
dev EERs are reported, not used to choose).

## Results

All five seeds trained and were scored without error (run health clean, code at main
`e59480c`). Per-set results are in `experiments/results/variant_final_v1/`; the decision,
from the rule as committed, is `experiments/results/variant_final_v1.json`:

```
python -m src.reporting.variant_final --runs experiments/results/variant_final_v1/runs.json
```

**Decision: not confirmed.** The variant is worse than the final recipe on the primary
tool in every seed, and its Tortoise cost is above the 1-point limit.

| Seed | IndicF5, G.711: variant | final recipe | New-speaker Tortoise, G.711: variant | final recipe |
|---|---|---|---|---|
| 1234 | 43.92 | 25.66 | 13.42 | 9.00 |
| 2025 | 38.57 | 18.09 | 12.16 | 8.55 |
| 7 | 28.89 | 23.19 | 9.91 | 6.79 |
| 42 | 29.23 | 24.13 | 5.87 | 8.92 |
| 99 | 30.59 | 17.33 | 11.73 | 10.80 |
| Mean ± SD | 34.24 ± 6.70 | 21.68 ± 3.74 | 10.62 ± 2.94 | 8.81 ± 1.43 |

1. Lower IndicF5 EER in at least 4 of 5 seeds: **no** (0 of 5).
2. Mann-Whitney p ≤ 0.05: p = 0.008, the smallest possible. The test is two-sided, so this
   says the two recipes differ, and here the difference is that the variant is **worse**;
   condition 1 is the one that fixes the direction.
3. Tortoise within 1 point: **no** (+1.81 points).

Secondary sets (EER %, mean ± SD over the same five seeds; reported, not part of the
decision):

| Set | Final recipe | Variant | Variant lower in |
|---|---|---|---|
| Code-mixed test pool, XTTS-v2 (seen tool), G.711 | 1.13 ± 0.18 | 0.87 ± 0.19 | 5 of 5 |
| CM04 Tortoise, G.711 | 17.73 ± 2.27 | 21.37 ± 2.75 | 0 of 5 |
| CM04 Tortoise, AMR-NB | 6.90 ± 2.06 | 11.21 ± 1.66 | 1 of 5 |
| RVC test half | 8.53 ± 1.91 | 15.64 ± 2.29 | 0 of 5 |
| IndicF5, AMR-NB | 21.32 ± 2.27 | 30.23 ± 4.69 | 0 of 5 |
| New-speaker XTTS-v2, G.711 | 0.94 ± 0.34 | 0.70 ± 0.25 | 3 of 5 |
| New-speaker XTTS-v2, AMR-NB | 0.69 ± 0.24 | 0.53 ± 0.31 | 4 of 5 |
| New-speaker Tortoise, AMR-NB | 6.04 ± 1.17 | 7.10 ± 1.55 | 2 of 5 |
| Language-control genuine called fake (%) | 0.51 ± 0.67 | 0.20 ± 0.12 | 3 of 5 |

Chatterbox was not scored in this run (the final recipe's checkpoints were not rescored on
it), so, as registered, it is not compared. Development EER was 0.00% for four seeds and
0.43% for one, so, as expected, the dev set would not have told the two recipes apart.

## What this means

The week-12 lead is closed. On the deployed recipe (21 speakers, one codec) the three
options lowered CM04 Tortoise by 9 points; on the final recipe (248 speakers, both codecs)
they raise the mean EER of every unseen tool (CM04 by 3.6, RVC by 7.1, IndicF5 by 12.6 points) and only
help on XTTS-v2, the tool the model was trained on (test pool and new speakers). One reading, not
tested here, is that the week-12 gain came from extra variety that the final recipe already gets
from its 248 speakers and two codecs, and that with that variety present the options make
the model fit the training tool more closely. The final recipe stays as it is, and the
paper's exploratory week-12 result is reported together with this failed confirmation.
