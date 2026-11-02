# The second SSL-AASIST checkpoint and the two-checkpoint average (v1)

Plan task A1. Every real-call result for the larger encoder (SSL-AASIST, XLS-R 300M
fine-tuned on the final recipe's data) rested on one checkpoint, seed 1234, and two
trainings of the same recipe differ a lot on the hardest sets (`encoder_spontaneous_v1.md`).
This document scores the second checkpoint, seed 2025, exactly as the first was scored,
and the average of the two.

## How each operating point was chosen (dev only)

All three use the live path (4 s windows every 2 s, smoothed by the same moving average)
and the same rule on the same 506 dev clips (`python -m src.inference.calibrate choose`:
the most sensitive candidate with at most the allowed share of genuine 60 s dev calls
alerted). Nothing was chosen on a call or a test set.

| Model | Threshold | Ladder (caution, likely cloned) | File |
|---|---:|---|---|
| SSL-AASIST seed 1234 | 0.580 | 2, 4 windows | `configs/threshold_ssl_aasist.yaml` |
| SSL-AASIST seed 2025 | 0.334 | 2, 4 | `configs/threshold_ssl_aasist_s2025.yaml` |
| Average of the two | 0.540 | 2, 4 | `configs/threshold_ssl_aasist_avg.yaml` |

The average takes the mean of the two checkpoints' P(genuine) for every window and smooths
the mean exactly as the live path smooths one model (`src/inference/ensemble.py`, tests in
`tests/test_ensemble.py`). The two checkpoints place their scores differently, which is
why each has its own threshold; the rule, not the number, is what is shared.

## The 30 real calls (three speakers, scored offline through the live path)

Calls warned (caution or worse) and taken to "likely cloned", out of the calls of each
kind. **Over G.711**, as the live system's carrier delivers them:

| Model | Genuine live (12): warned / likely | Genuine replays (6): warned / likely | XTTS-v2 clones (6): likely | Chatterbox clones (6): warned / likely |
|---|---|---|---|---|
| Seed 1234 | 1 / 0 | 1 / 1 | 6 | 3 / 1 |
| Seed 2025 | 0 / 0 | 1 / 0 | 6 | 0 / 0 |
| **Average** | **0 / 0** | **0 / 0** | **6** | **3 / 0** |

**As recorded** on the receiving phone:

| Model | Genuine live (12): warned / likely | Genuine replays (6): warned / likely | XTTS-v2 clones (6): likely | Chatterbox clones (6): warned / likely |
|---|---|---|---|---|
| Seed 1234 | 0 / 0 | 1 / 1 | 6 | 3 / 2 |
| Seed 2025 | 0 / 0 | 0 / 0 | 6 | 0 / 0 |
| **Average** | **0 / 0** | **0 / 0** | **6** | **3 / 0** |

Results: `experiments/results/real_calls_v1/` (seed 1234), `.../ssl_s2025/` and
`.../ssl_avg/` (per-window scores, per-call verdicts, `summary.json`, `report.md`).

## Reading

- **The two checkpoints agree where it matters most.** Both take every XTTS-v2 clone call
  to "likely cloned", and neither takes any genuine live call there. The single-checkpoint
  caveat on these rows is removed.
- **They disagree on Chatterbox.** Seed 1234 warns on 3 of 6 Chatterbox calls (1 or 2
  likely cloned); seed 2025 warns on none, although its offline Chatterbox EER is about
  10%. On real calls an unseen generator's result can hinge on the training run.
- **The average warns on no genuine call** (0 of 18 in both conditions) and keeps the
  caution on half the Chatterbox calls, but takes none of them to "likely cloned".
- With 6 calls per clone kind the intervals are wide (3 of 6 is 19-81% at 95%), so the
  Chatterbox differences between the three are not significant.

## Held-out 60 s calls (validation, chooses nothing)

The held-out clip sets, stitched into 60 s calls and replayed through the live path at
each model's dev-chosen operating point, exactly as the first checkpoint was validated
(`python -m src.inference.calibrate choose --validate`; reports
`experiments/results/threshold_calibration_ssl_aasist{,_s2025,_avg}.json`, windows in
`experiments/calibration_ssl{,_s2025,_avg}/`). Calls warned / taken to "likely cloned":

| Held-out calls | Seed 1234 | Seed 2025 | Average |
|---|---|---|---|
| HiACC genuine spontaneous speech (41 calls) | 11 / 1 | **31 / 15** | 23 / 10 |
| Genuine read speech in the CM04, IndicF5 and Chatterbox sets (8-15 calls each) | 0 / 0 | 0 / 0 | 0 / 0 |
| Tortoise, CM04 (6 clone calls) | 6 / 6 | 6 / 6 | 6 / 6 |
| IndicF5 (6) | 6 / 4 | 4 / 2 | 5 / 3 |
| Chatterbox (6) | 5 / 1 | 6 / 6 | 6 / 4 |

**The two checkpoints are opposite on spontaneous speech.** Seed 2025 warns on 31 of the
41 genuine spontaneous calls, seed 1234 on 11. Scored clip by clip (each clip's mean window
P(genuine) below 0.5), seed 1234 calls 6.4% of the 1,000 HiACC clips fake, seed 2025 30.9%
and the average 12.9%. Two earlier trainings of the same encoder and recipe (run 9) split
the same way, 5.8% and 46.7% (`encoder_spontaneous_v1.md`), and in both pairs each training
reached 0% dev EER, so nothing in model selection separates them. On Chatterbox the order
reverses (seed 2025 takes all six held-out calls to "likely cloned", seed 1234 one), and on
the real calls neither pattern holds (seed 2025 caught none of six Chatterbox calls). The
average sits between the two on every row.

**What this means for deployment.** There is no single best version. Seed 1234 is the
safest for spontaneous callers (11 of 41 warned), though over G.711 it warns one genuine
live call and one genuine replay of the 18 real genuine calls; the average warns none of
the 18 and catches more Chatterbox, at the cost of warning 23 of 41 spontaneous calls. The underlying problem
is not the choice of checkpoint: without spontaneous speech in training or in the dev set,
which training generalises to it is luck. That is what plan task A2 addresses.

With six clone calls per held-out set the intervals are wide; differences of one or two
calls between versions are not evidence.

**A second spontaneous corpus removes the advantage.** On 94 genuine 60 s calls of the
IndicVoices test speakers (extempore Hindi, `indicvoices_data_v1.md`), seed 1234 warns 79,
seed 2025 90 and the average 87 (the final adapter 93). Seed 1234's low false-alarm rate on
HiACC does not carry over, so it is not evidence that the larger encoder handles spontaneous
speech. At clip level (scored on Kaggle, one 4 s crop per clip) the order is the HiACC order:
seed 1234 calls 14.6% of the 1,024 clips fake, the average 17.4%, seed 2025 39.5%, against
0.3-0.6% of read speech for each.
