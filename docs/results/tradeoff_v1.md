# Hinglish over the phone line against English: the trade-off v1

Answers external review comment 4: *quantify the trade-off between Hinglish
phone-line performance and clean-English performance, and say what it means for a
deployment.*

Code: `src/reporting/tradeoff.py` (tests: `tests/test_tradeoff.py`).
Results: `experiments/results/tradeoff_v1.json`, built from
`experiments/results/systems_s1_s2_s3.json` (own condition),
`experiments/results/cross_condition_v1/` (other condition, new here),
`experiments/results/classification_metrics.json` (English at the deployed threshold)
and `experiments/results/english_channel_v1/` (English over the phone line; see
`docs/results/english_channel_v1.md`).

## What was measured

Each S2 adapter is trained in one condition, clean or G.711 phone line. The paper
reported each one only in its own condition, so the cost of the choice was invisible.
New here: all four adapters scored on the three Hinglish test sets in the condition
they were **not** trained in (all 12 cells), on the laptop CPU with the
same evaluation code, the same 4 s window and the same normalised bundles as the
own-condition numbers.

## Result

EER (%). Own-condition numbers are the ones in the paper's main table.

| S2 adapter | Trained on | Hinglish, own condition (eval / CM04 / RVC) | Hinglish, other condition (eval / CM04 / RVC) | English, clean | English fakes caught at 0.5 |
|---|---|---|---|---:|---:|
| XTTS | clean | 1.46 / 5.44 / 23.91 | 27.64 / 18.98 / 39.74 (channel) | 15.83 | 51.8% |
| XTTS | channel | 4.41 / 28.16 / 29.56 | 12.51 / 18.78 / 31.50 (clean) | 2.55 | 67.1% |
| XTTS + RVC | clean | 1.99 / 12.01 / 5.01 | 22.97 / 31.02 / 26.49 (channel) | 3.75 | 79.6% |
| XTTS + RVC | channel (deployed) | 2.75 / 30.99 / 5.01 | 6.71 / 18.33 / 13.89 (clean) | 2.42 | 77.4% |

English on the fixed 6,004-clip sample (Lahari's test, `english_channel_v1.md`):

| Model | English sample, clean | English sample, phone line |
|---|---:|---:|
| S1 (English only) | 0.51 | 6.50 |
| Deployed S2 | 2.30 | 24.20 |

Intervals for every cell are in `tradeoff_v1.json` and in the paper's table.

## What it means

- **Training over the phone line costs little on clean audio; the reverse costs a
  lot.** On the evaluation pool the deployed adapter goes from 2.75% to 6.71% on clean
  audio, while the clean-trained XTTS + RVC adapter goes from 1.99% to 22.97% over the
  phone line. The XTTS-only pair shows the same thing (4.41 -> 12.51 against
  1.46 -> 27.64), and so does RVC: the deployed adapter rises from 5.01% to 13.89% on
  clean audio, its clean-trained counterpart from 5.01% to 26.49% over the phone line.
  The two channel-trained adapters also keep clean English best (2.55% and 2.42%,
  against 15.83% and 3.75%).
- **Tortoise is harder over the phone line whatever the training condition.** All
  four adapters do worse on CM04 over the phone line than on clean CM04 (deployed:
  30.99% against 18.33%; clean-trained XTTS: 18.98% against 5.44%). The channel itself
  hides the unseen tool; choosing the training condition does not fix it.
- **English over the phone line is the real cost.** The deployed model goes from
  2.30% to 24.20% on English once it passes through the same G.711 chain, while S1,
  never trained on phone audio, goes only from 0.51% to 6.50%. The "keeps English at
  2.42%" claim holds only for clean English.

## For a deployment

For a Hindi-English phone service the channel-trained XTTS + RVC adapter remains the
right choice: of the two channel-trained adapters it is better on the evaluation pool
(2.75% against 4.41%) and far better on RVC (5.01% against 29.56%), within three
points on CM04 (30.99% against 28.16%), and it degrades least when the condition
changes. It should not be used to
judge English calls. A service taking both would need a language-identification step
sending English calls to S1, or English phone-line audio in the adaptation data;
neither has been tested.

## Measured late

The deployed adapter on the clean RVC set and S1 on the clean English sample both
stopped twice when the laptop ran out of memory; the third attempts, on 13 October,
completed (13.89% and 0.51%).

## Reproduce

Each cross-condition cell is one evaluate call. For example, the deployed adapter on
the clean evaluation pool:
```
python -m src.training.evaluate --checkpoint <lora_norm_rvc_channel best.pt> --manifest data/manifests/codemix_eval.csv --encoder wav2vec2-base --data-root <clean normalised bundle> --device cpu --batch-size 16 --num-workers 2 --max-seconds 4.0 --scores-out experiments/results/cross_condition_v1/lora_norm_rvc_channel__eval_pool_clean_scores.csv --out experiments/results/cross_condition_v1/lora_norm_rvc_channel__eval_pool_clean.json
python -m src.reporting.tradeoff
```
The manifests are `codemix_eval`, `score_cm04_norm` and `rvc_holdout_test`, with their
`_channel20` versions and the channel bundle for the phone line. About 2-15 minutes
per cell on the laptop CPU.
