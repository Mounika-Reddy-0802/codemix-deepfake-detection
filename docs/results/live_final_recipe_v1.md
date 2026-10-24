# The live system with the final recipe (v1)

What a caller would meet. Every number here comes from the live scoring path
(`src.inference.calibrate`: 4 s windows every 2 s, level-normalised, smoothed, replayed
through the verdict engine), not from one crop per clip. Clips of the same label are
stitched into 60 s calls, as in the week-8 calibration (P-029).

**Operating point.** Chosen on the dev manifest only, by the rule fixed in week 8: the most
sensitive threshold and warning ladder whose genuine 60 s dev calls raise no more than 5%
alerts. For the final recipe (seed 1234) that is a threshold of 0.687 with a warning after
3 fake-leaning windows and a "likely fake" after 5 (`configs/threshold_final.yaml`); the dev
calls raise 0% alerts and every fake dev call is caught. Re-running the same rule on the
deployed model's dev windows reproduces its published threshold exactly (0.49997), so the
two are compared like for like (`experiments/results/threshold_calibration_final.json`,
`threshold_calibration_deployed_w14.json`).

**Test calls** (300-clip label-stratified subsets per set, as in week 8; HiACC 1,000 clips;
the window scores are in `experiments/calibration/` and `experiments/calibration_final/`):

| 60 s calls | Calls (genuine / fake) | Deployed: genuine warned | Final: genuine warned | Deployed: fakes caught | Final: fakes caught |
|---|---|---:|---:|---:|---:|
| Evaluation pool (XTTS) | 7 / 13 | 0% | 0% | 100% | 100% |
| CM04 (Tortoise, unseen) | 15 / 6 | 0% | 0% | 33% | **100%** |
| RVC test half | 9 / 8 | 0% | 0% | 100% | 100% |
| 60 new speakers (XTTS + Tortoise) | 9 / 8 | **44%** | **0%** | 100% | 100% |
| IndicF5 (unseen) | 8 / 6 | 0% | 0% | 67% | 50% |
| Chatterbox (unseen) | 8 / 6 | 0% | 0% | 33% | 17% |
| HiACC, spontaneous (genuine only) | 41 / 0 | **100%** | **100%** | - | - |

"Fakes caught" counts calls that reach at least a warning; "genuine warned" counts genuine
calls that raise any warning.

## Reading

- **The final recipe is the better live model on read speech.** It stops warning genuine
  new speakers (44% of their calls to 0%) and catches every Tortoise call (33% to 100%),
  with no warning on any genuine read-speech call in any set.
- **It catches fewer IndicF5 and Chatterbox calls** (67% to 50%, 33% to 17%). Its threshold
  is set for no false alarms, and on these two tools its fake windows sit near it.
- **Spontaneous speech is rejected at the call level by both models.** A single short clip
  is rarely warned (2.8% final, 8.6% deployed), but over a minute the fake-leaning windows
  of spontaneous speech add up, and every genuine 60 s spontaneous call raises a warning.
  This is the real-call form of the 44% clip-level false alarm rate, and it is the clearest
  limit of both systems: neither should warn a spontaneous caller in deployment until it
  has seen spontaneous genuine speech in training.
- **The counts are small** (6 to 41 calls per cell): read them as directions, not rates.

## Using it

The live system takes its model and threshold from the environment. To run the final recipe:

```bash
set DFD_CHECKPOINT=<data>/checkpoints/final_s1234_best.pt
set DFD_THRESHOLD_FILE=configs/threshold_final.yaml
```

`configs/threshold.yaml` stays the deployed model's operating point, so the published demo
still reproduces.
