# Week 12 — Mounika — accuracy, precision, recall and confusion matrices

The paper reports EER and AUC because they do not depend on where the decision line
is drawn. A viva panel asks for accuracy, precision and recall. Both are now
available, from the same per-clip scores, with the caveats that make the second set
readable rather than misleading.

## What I did, and why it is built this way

`src/reporting/classification.py` reads the per-clip score CSVs already in the
repository — nothing is re-scored — and reports every run twice: at the deployed
operating point from `configs/threshold.yaml`, and at that run's own EER threshold.

Three decisions shape the output, and each one exists to stop a number being read
the wrong way:

1. **The positive class is *spoof*.** Recall is the share of cloned clips caught;
   specificity is the share of real speech left alone. Stating this matters because
   the opposite convention turns a system that flags everything into one with
   perfect recall.
2. **The class balance travels with every row.** ASVspoof is 8.7 : 1 spoof-to-real,
   the code-mixed sets are near 1.5 : 1. Accuracy is not comparable across them, and
   for three S3 runs the English accuracy is *exactly* the bonafide share — the
   model calls everything real, and accuracy is quietly reporting the class balance.
3. **An undefined rate is `None`, never 0.0.** Precision with no positive
   predictions is undefined. Writing zero there would be a lie the tests now forbid.

## Numbers

39 model-and-set combinations. The EERs reproduce the published figures exactly
(deployed model: 2.75% XTTS, 5.01% RVC, 30.99% Tortoise, 2.42% English), which is
the check that the arithmetic is right.

Deployed model, S2 LoRA XTTS+RVC channel-matched, at threshold 0.500:

| Test set | Accuracy | Precision | Recall | Specificity | F1 | EER |
|---|---:|---:|---:|---:|---:|---:|
| XTTS (seen) | 96.2% | 99.4% | 94.3% | 99.1% | 96.8% | 2.75% |
| RVC | 90.6% | 98.1% | 83.3% | 98.3% | 90.1% | 5.01% |
| Tortoise (unseen) | 79.2% | 78.5% | **11.1%** | 99.1% | 19.5% | 30.99% |
| English | 79.7% | 100.0% | 77.4% | 100.0% | 87.2% | 2.42% |

Three findings the matrices make concrete:

- **S1 does not merely fail on Hinglish, it rejects real speech.** On XTTS it has
  92.6% recall and **5.2% specificity** — 1,484 of 1,566 genuine clips flagged. High
  recall from calling almost everything fake is worthless.
- **Adding RVC to training is the single biggest win.** Recall on the RVC set goes
  from 17.7% (XTTS-only, channel) to 83.3%.
- **S3 native collapses on English and the matrix shows how.** Three clones caught
  out of 63,882, recall 0.0%. S2 keeps 77.4% on the same set.

## Honest limitations

- **These numbers are a slice, not a summary.** Every accuracy here depends on one
  fixed threshold calibrated for the live code-mixed demo, not tuned per set. Where
  accuracy looks poor beside a good EER, that is a calibration result, not a
  detection failure. The second table at each run's own EER threshold is there so
  the difference is visible rather than argued about.
- **No confidence intervals.** The EER figures elsewhere carry bootstrap CIs; these
  counts do not. On the RVC set, 619 clips, a handful of clips moves recall by more
  than a point.
- **The Tortoise rows are the honest failure.** 51 of 458 caught. It is reported in
  the headline table rather than a footnote.

## How to run it

```bash
python -m src.reporting.classification          # writes the JSON, prints the table
pytest tests/test_classification.py
```

## What's next

- Bootstrap CIs on recall and specificity for the small sets.
- Feed the deployed model's confusion matrix into the results chapter as a figure.

---

# Later in week 12 — seed stability and the phone-line trade-off

The external review (`docs/external_review_response.md`) asked for the phone-line
results across random seeds with intervals (comment 1), and for the trade-off between
Hinglish over the phone line and English (comment 4). I took the statistics for both.

## What I did, and why it is built this way

### Two kinds of uncertainty, reported side by side

Every published number came from one training run, seed 1234, and our intervals
resampled clips: they cover which test clips we happened to have, not training
randomness. `src/reporting/seeds.py` reads the seed repeats and gives, per system and
test set, the mean, standard deviation, range and a t-interval of the mean across
seeds, and for every seed the paired speaker-level test against the matched floor.
The floor depends only on the training data and the test clips, so every seed of a
system is compared with the same floor scores. A claim like "beats its floor" can
now be stated as "in k of 5 seeds".

### The trade-off is read from the result files, not retyped

`src/reporting/tradeoff.py` puts in one table the numbers measured in different
places: each adapter in its own condition, the same adapter in the condition it was
not trained in (scored for this, on the laptop CPU), clean English, and English over
the phone line (Lahari's sample). A cell not yet measured shows as "not measured".

## Numbers

| | Result |
|---|---|
| Deployed, Tortoise, 5 seeds | **35.59 ± 3.35**, beats its floor (37.79) in 0 of 5; published run 30.99 |
| Deployed, RVC, 5 seeds | 5.33 ± 1.27, beats its floor in 5 of 5 |
| Deployed, eval pool, 5 seeds | 3.27 ± 1.17, beats its floor in 3 of 5 |
| S3 XTTS-only, Tortoise, 5 seeds | **19.72 ± 5.36**, every seed below its floor (33.64), 3 significant |
| Deployed adapter, eval pool, clean instead of phone line | 2.75 -> 6.71 |
| Clean-trained adapter, eval pool, phone line instead of clean | 1.99 -> 22.97 |
| Deployed adapter, RVC test, clean instead of phone line | 5.01 -> 13.89 |
| Deployed, English sample, clean -> phone line | 2.30 -> 24.20 (S1: 0.51 -> 6.50) |

## Honest limitations

- **The published Tortoise number was optimistic.** The paper quoted 30.99%; over five
  seeds the deployed model averages 35.59%. The results chapter, abstract and
  conclusion now quote the mean.
- **Five seeds give wide intervals**, and the t-interval assumes the seed EERs are
  roughly normal.
- **One trade-off cell came late**: the deployed adapter on the clean RVC set ran out
  of memory on the laptop twice before it completed (13.89%).
- **The S2 XTTS-only and S3 rows mix sessions**: four new seeds join the published 1234
  run, scored by the same code in an earlier session.

## How to run it

```bash
python -m src.reporting.seeds --results experiments/results/review_runs_v1 --out experiments/results/seeds_v1.json
python -m src.reporting.tradeoff
pytest tests/test_seeds.py tests/test_tradeoff.py
```
Full write-ups: `docs/results/seed_stability_v1.md`, `docs/results/tradeoff_v1.md`.

## What's next

- The same seed analysis for the exploratory variant is in Krishna's
  `docs/results/variants_v1.md`.
