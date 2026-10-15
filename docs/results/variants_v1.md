# Training changes aimed at the unseen tool v1

Answers external review comment 2: *investigate methods to improve generalisation to
unseen voice-cloning tools, particularly Tortoise-TTS.*

Code: `src/reporting/variants.py` (tests: `tests/test_variants.py`); the training
options in `src/data/augment.py`, `src/training/sampling.py` and
`src/models/encoder.py`. Runs: three Kaggle notebooks (`notebooks/kaggle_w12_*.ipynb`),
2026-10-13 to 14. Results: `experiments/results/review_runs_v1/` (variants with crop
and crop + augment), `review_runs_v1_layers/` (the two layer-mix variants),
`review_runs_v1_explore/` (four more seeds of one variant), and
`experiments/results/variants_v1.json`.

## What was tried

Three changes to the deployed recipe (S2 LoRA, XTTS + RVC, phone line), each opt-in and
off by default, so the published configs train exactly as before:

- **crop** - a fresh random 4 s crop of every clip each epoch. The published recipe
  reused one crop for every epoch (a bug found during the review).
- **augment** (`channel_v1`) - on every training clip, real and fake alike: random level
  (±6 dB, always), spectral tilt (±3 dB/octave, half the time), coloured noise
  (10-40 dB SNR, 30%) and a second G.711 pass at 5-30 dB SNR (half the time). These
  randomise the recording cues the shortcut floor found.
- **layer mix** - a learned softmax weighting of all 13 encoder layers instead of the
  last one, since the decision forms by layer 3. It starts at equal weights.

## Selection, by the rule fixed before any run

Lowest dev EER wins; dev EERs within half a point tie; a tie goes to fewer changes.
Test sets are scored but never consulted. Seed 1234, phone line, EER (%):

| Candidate | Dev | Eval pool | CM04 | RVC |
|---|---:|---:|---:|---:|
| **Published recipe (chosen)** | 5.50 | 2.57 | 30.14 | 5.33 |
| + crop | 9.10 | 4.59 | 32.58 | 6.62 |
| + crop + augment | 5.34 | 3.38 | 29.49 | 7.59 |
| + crop + layer mix | 9.10 | 3.38 | 30.78 | 7.92 |
| + crop + augment + layer mix | 9.10 | 3.38 | 26.43 | 7.27 |

The published recipe stays: crop + augment is 0.16 points lower on dev, a tie, and the
tie goes to the recipe. No change improves the model on its own validation data.

## An exploratory check of one variant

Crop + augment + layer mix had the lowest single-run CM04 (26.43%), below every seed
of the published recipe (30.14-38.43). It was singled out **after** seeing that test
score, so this is exploratory, not a selection. It was retrained with seeds 2025, 7,
42 and 99 and compared with the published recipe's five seeds:

| Set | Published recipe | Variant | Difference | Exact Mann-Whitney p (all 5 / 4 new seeds) | Variant seeds beating floor |
|---|---|---|---:|---|---:|
| Eval pool | 3.27 ± 1.17 | 2.65 ± 0.53 | -0.62 | 0.39 / 0.25 | 5 of 5 (floor 9.96) |
| **CM04 (Tortoise)** | 35.59 ± 3.35 | **26.63 ± 1.66** | **-8.96** | **0.008 / 0.016** | 0 of 5 (floor 37.79) |
| RVC test | 5.33 ± 1.27 | 7.72 ± 1.33 | +2.39 | 0.03 / 0.05 | 5 of 5 (floor 20.52) |
| Dev | 6.96 ± 2.70 | 7.11 ± 1.38 | +0.15 | | |

- **The Tortoise gain survives the seeds.** All five variant seeds (24.19-28.81) are
  below all five of the recipe (30.14-38.43). The four seeds trained after the choice,
  which the choice could not have influenced, give the same picture (p = 0.016).
- **It costs RVC.** 7.72% against 5.33%, still beating its floor in every seed.
- **It still does not clear the Tortoise floor.** Per seed, the variant is 9.0 to 13.6
  points under the matched floor, but with 15 test speakers every interval reaches
  zero (e.g. seed 42: -13.60 [-28.96, +1.97]).
- **The dev set cannot see it.** Dev EER is the same for both recipes (7.11 against
  6.96 averaged over seeds). The in-domain dev set (XTTS + RVC, the training tools)
  measures nothing about an unseen tool, which is why a dev-only rule kept the
  published recipe. Selecting for unseen-tool generalisation needs a validation split
  built from a held-out synthesis tool, separate from the test tool.

## Honest limitations

- **Selection bias.** The variant was picked as the best of four on a test score. The
  four fresh seeds remove the luck of the seed-1234 run, but not the fact that the
  choice used the CM04 clips; a second unseen tool is needed to confirm it.
- **Which part helps is not separated.** Augmentation lowered CM04 in both pairs where
  it was added (crop: 32.58 -> 29.49; crop + layer mix: 30.78 -> 26.43), but each of
  those is a single run. Only the full combination has five seeds.
- **The layer-mix variants first failed.** Both crashed at the first step because the
  Stage-1 checkpoint has no layer-mix weights and the load was strict; fixed in
  `src/training/train.py` (`load_initial_weights`) and re-run. The learned layer
  weights were not saved, so we cannot report which layers the mix chose.
- **Three dev EERs are identical (9.10%)** and several eval-pool ones too (3.38%). With
  506 dev clips the EER moves in coarse steps, so ties are expected; the test scores
  differ, so the runs did differ.

## Reproduce

```
python -m src.reporting.variants
```
The runs: `notebooks/kaggle_w12_review_runs.ipynb` (variants block),
`kaggle_w12_layer_variants.ipynb` and `kaggle_w12_explore_seeds.ipynb`, one Run All
each on a Kaggle T4.
