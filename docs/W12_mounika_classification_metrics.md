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
