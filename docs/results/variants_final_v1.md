# Confirmatory test of the exploratory variant on the final recipe (v1)

**Status: registered, running.** Written on 31 October 2026, before any training or
scoring of this variant, so the hypothesis and the decision rule cannot be fitted to its
results. Plan task A5; it runs only if GPU hours remain after tasks A1-A4. Committed on
3 November 2026 together with the notebook as launched (Kaggle `w15-variant-final`, started
15:46 that day), while the run was in progress and before any of its results existed. The
launched notebook adds one guard: a seed starts only with 75 minutes of the 6 GPU hours left
that week to spare, so fewer than five seeds can finish; the decision rule then reports the
result as incomplete rather than changing.

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

Not run yet.
