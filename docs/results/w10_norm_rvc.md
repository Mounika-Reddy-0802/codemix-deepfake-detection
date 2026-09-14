# W10: adapters retrained on the normalised bundle, with and without RVC

**W9-T1 ablation + W8-T1 re-evaluation, owner M.** Four LoRA adapters, identical
except for their training data and condition, retrained on the level-normalised
bundle (`bundle_normalisation_v1.md`) and scored in their own condition on three
sets. Full Kaggle run on T4, 13 Sep 2026, branch
`week10-mounika-unseen-attack-scoring`. Per-clip scores and pooled JSONs:
`results/w10/`. Checkpoints: private Kaggle dataset `codemix-w10-results`.

## Every number beside its own floor

A low EER on this corpus is not evidence on its own: eight cheap signal statistics
already separate the classes (`lowlevel_cue_check_v1.md`). So each set gets its own
shortcut floor, measured on the same clips in the same condition with a
speaker-disjoint fit/score split. Before this run only the eval-pool floors existed.
The CM04 and RVC-test floors below are new.

| Set | What it tests | Floor, clean | Floor, channel |
|---|---|---:|---:|
| Eval pool | seen tool (XTTS) | 5.17% | 10.01% |
| CM04 | unseen tool (Tortoise) | 31.21% | 25.79% |
| RVC test | RVC, speakers unseen in training | 28.76% | 22.46% |

## Result

EER with 95% bootstrap CI. **Bold** means the CI's upper bound is below the floor.
~~Struck through~~ means it is not below the floor, so it is not evidence of learning.

| Adapter | Condition | Eval pool | CM04 (unseen tool) | RVC test |
|---|---|---:|---:|---:|
| XTTS only | clean | **1.46%** [1.01, 1.94] | **5.44%** [4.05, 6.57] | **23.91%** [20.33, 27.64] |
| XTTS + RVC | clean | **1.99%** [1.47, 2.45] | **12.01%** [10.42, 13.64] | **5.01%** [3.55, 6.95] |
| XTTS only | channel | **4.41%** [3.78, 5.19] | ~~28.16%~~ [25.44, 30.20] | ~~29.56%~~ [26.49, 32.79] |
| XTTS + RVC | channel | **2.75%** [2.22, 3.37] | ~~30.99%~~ [28.72, 33.84] | **5.01%** [3.39, 6.95] |

Figure: `experiments/figures/ablation_rvc.png`. Machine-readable, with margins:
`experiments/results/ablations.json` (`python -m src.reporting.ablation`).

The two 5.01% cells are not a copied file: their per-clip scores differ (correlation
0.875, no identical values) and happen to reach the same error counts at the
equal-error point.

## What it means

### 1. On the seen tool, every adapter clears its floor

Clean scores 1.46% against a 5.17% floor. Channel scores 4.41% against 10.01%, and
2.75% with RVC in training. These replace the pre-normalisation 1.34% / 3.89%, which
could not be quoted because the floor then was 1.39%.

### 2. The ablation's answer is no: attack diversity costs unseen-tool EER

| Condition | Set | XTTS only | XTTS + RVC | Change |
|---|---|---:|---:|---:|
| clean | CM04 (unseen tool) | 5.44% | 12.01% | **+6.57** |
| channel | CM04 (unseen tool) | 28.16% | 30.99% | **+2.83** |
| clean | RVC test | 23.91% | 5.01% | −18.90 |
| channel | RVC test | 29.56% | 5.01% | −24.55 |

Adding RVC closes the voice-conversion blind spot from P-025, on speakers the adapter
never saw, in both conditions. It also more than doubles the clean unseen-tool EER.
The adapter gets better at the families it is shown and worse at the one it is not,
so training on more attack families is not a free route to generalisation (P-027).

### 3. Channel-matched detection of the unseen tool is not demonstrated

Both channel adapters sit **above** CM04's channel floor (28.16% and 30.99% against
25.79%). Over a phone line, eight signal statistics separate Tortoise from real
speech as well as the model does. The earlier 13.10% from the un-normalised channel
adapter (`unseen_attack_scoring_v1.md`) does not survive normalisation, so part of it
was level. The XTTS-only channel adapter is also above the RVC-test channel floor
(29.56% against 22.46%).

### 4. What this does to the headline

The plan was for the channel-matched condition to carry the headline. On these
numbers it can carry only the **seen-tool** claim. The unseen-tool claim rests on
the **clean** adapter: 5.44%, 25.77 points below its 31.21% floor. The paper states
both, and lists channel-matched unseen-tool detection as a limitation.

## Reproduce

```bash
# floors (CPU, a few minutes each; fit/score split by speaker)
python -m src.data.lowlevel_cue --train data/manifests/score_cm04_norm_channel20_fit.csv \
    --train-root "$CHANNEL_ROOT" --test data/manifests/score_cm04_norm_channel20_score.csv \
    --test-root "$CHANNEL_ROOT" \
    --out experiments/results/lowlevel_cue_check_cm04_normalised_channel20.json
python -m src.data.lowlevel_cue --train data/manifests/rvc_holdout_train_channel20.csv \
    --train-root "$CHANNEL_ROOT" --test data/manifests/rvc_holdout_test_channel20.csv \
    --test-root "$CHANNEL_ROOT" \
    --out experiments/results/lowlevel_cue_check_rvc_holdout_normalised_channel20.json
# clean floors: the same with score_cm04_norm_{fit,score}.csv,
# rvc_holdout_{train,test}.csv and "$CLEAN_ROOT"

# adapters: notebooks/kaggle_w10_retrain_normalised.ipynb with SMOKE = False
python -m src.reporting.ablation   # experiments/results/ablations.json
python -m src.reporting.figures    # experiments/figures/
```
