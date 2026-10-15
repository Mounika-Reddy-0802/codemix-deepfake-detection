# Stability across training seeds v1

Answers external review comment 1: *evaluate the channel-matched LoRA model across
multiple random seeds and report confidence intervals.*

Code: `src/reporting/seeds.py` (tests: `tests/test_seeds.py`).
Runs: `notebooks/kaggle_w12_review_runs.ipynb` on a Kaggle T4, 2026-10-13/14.
Results: `experiments/results/review_runs_v1/` (every run's JSONs, score files and the
training log) and `experiments/results/seeds_v1.json`.

## What was measured

Every published number came from one training run with seed 1234. Three phone-line
systems were retrained with the published configs unchanged except for the seed:

- the deployed S2 (LoRA, XTTS + RVC, channel): seed 1234 re-run, plus 2025, 7, 42, 99;
- S2 XTTS-only, channel: 2025, 7, 42, 99, joined by the published 1234 run;
- S3 native XTTS-only, channel: 2025, 7, 42, 99, joined by the published 1234 run.

Each seed is scored on the three Hinglish test sets over the phone line. Two kinds of
uncertainty are reported, because they answer different questions:

- **across seeds** - mean, standard deviation, range and a 95% t-interval of the mean:
  how much the number moves when the model is trained again;
- **per seed** - the paired, speaker-level comparison with the matched shortcut floor
  (`docs/results/shortcut_floor_v2.md`): does this trained model beat the shortcut on
  these speakers. The floor depends only on the training data and the test clips, so
  every seed of a system is compared with the same floor.

The deployed seed-1234 re-run reproduces the published run closely (CM04 30.14% against
30.99%, eval pool 2.57% against 2.75%, RVC 5.33% against 5.01%), so the pipeline is
the same; what changes between seeds is training randomness.

## Result

EER (%) over five seeds.

| System | Set | Mean ± sd | 95% CI of mean | Range | Matched floor | Seeds beating floor |
|---|---|---|---|---|---:|---:|
| Deployed S2 (XTTS + RVC, channel) | eval pool | 3.27 ± 1.17 | [1.82, 4.73] | 2.42-5.24 | 9.96 | 3 of 5 |
| | **CM04 (Tortoise)** | **35.59 ± 3.35** | [31.43, 39.75] | 30.14-38.43 | 37.79 | **0 of 5** |
| | RVC test | 5.33 ± 1.27 | [3.75, 6.91] | 3.71-7.27 | 20.52 | 5 of 5 |
| S2 XTTS-only, channel | eval pool | 3.45 ± 0.72 | [2.56, 4.34] | 2.75-4.41 | 10.01 | 0 of 5 |
| | CM04 | 25.59 ± 5.27 | [19.05, 32.13] | 17.89-30.61 | 33.64 | 1 of 5 |
| | RVC test | 27.07 ± 2.76 | [23.65, 30.50] | 24.23-30.21 | 23.91 | 0 of 5 |
| S3 native XTTS-only, channel | eval pool | 1.78 ± 0.93 | [0.62, 2.93] | 0.63-3.20 | 10.01 | 4 of 5 |
| | **CM04 (Tortoise)** | **19.72 ± 5.36** | [13.06, 26.38] | 13.71-27.07 | 33.64 | **3 of 5** |
| | RVC test | 36.83 ± 3.39 | [32.62, 41.05] | 32.15-40.39 | 23.91 | 0 of 5 (4 worse) |

## What it means

- **The deployed model's Tortoise number was a good draw.** The published 30.99% is
  near the best of five seeds; the mean is 35.59%, two points under its matched floor
  of 37.79%, and no seed beats the floor at the speaker level (the closest, the 1234
  re-run, is 7.65 points under it, interval [-23.24, +8.83]). The paper's statement that
  the deployed model is "inconclusive against its floor" on Tortoise still holds, but
  its distance below the floor should be quoted from the seed mean, not the single run.
- **The deployed model is stable where it works.** RVC: 5.33 ± 1.27, beating its
  floor in all five seeds. Evaluation pool: 3.27 ± 1.17, beating its floor in three of
  five; the other two are inconclusive, not worse.
- **S3 trained on XTTS alone is the reliable unseen-tool result.** Every one of its five
  seeds is below its Tortoise floor (13.71-27.07% against 33.64%), three of them
  significantly at the speaker level. The paper's Finding 5 rested on its best seed
  (13.71%); the claim survives, with the mean (19.72%) as the number to quote.
- **The price of S3 is RVC.** It is worse than its RVC floor in four of five seeds.
- **Seed variation is large next to the differences the paper discusses.** On CM04 the
  standard deviation across seeds is 3-5 points for every system, so any single-run
  difference smaller than that is not evidence.

## Honest limitations

- Five seeds give a wide interval for the mean; the t-interval assumes the seed EERs
  are roughly normal.
- The S2 XTTS-only and S3 rows join four new seeds to the published 1234 run, scored by
  the same code but in an earlier session; the deployed row uses its own re-run instead.
- Test sets are fixed: seeds measure training randomness, not test-set sampling. That is
  what the per-seed speaker bootstrap is for.

## Reproduce

```
python -m src.reporting.seeds --results experiments/results/review_runs_v1 --out experiments/results/seeds_v1.json
```
The runs themselves: `notebooks/kaggle_w12_review_runs.ipynb`, one Run All on a Kaggle
T4 (about five hours).
