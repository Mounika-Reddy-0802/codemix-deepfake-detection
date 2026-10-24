# Spontaneous speech with the final recipe (v1)

Same test as `hiacc_false_alarm_v1.md`: all 3,318 genuine clips of HiACC's 24 adult
speakers (spontaneous, never in any training set), level-normalised and rendered through
the G.711 line at 20 dB, called fake when P(genuine) < 0.5, intervals from 1,000
resamples of whole speakers. Code: `src/training/false_alarm.py` (`summarise`,
`compare_paired`; tests in `tests/test_false_alarm.py`).

| Model (one training each) | False alarms on HiACC | Read lectures (eval pool) |
|---|---|---|
| Deployed S2, published checkpoint | 54.4% [44.4, 64.2] | 0.9% |
| Deployed recipe, retrained in run 1 (seed 1234) | 38.5% [28.2, 48.7] | - |
| 248 speakers, G.711 (run 1, seed 1234) | 46.0% [38.8, 53.3] | - |
| Final recipe: 248 speakers, G.711 + AMR-NB (run 4, seed 1234) | 43.9% [36.5, 51.2] | 1.1% |

Paired on identical clips, the final-recipe model is 10.5 points below the published
deployed checkpoint, [-17.1, -3.8], with 16 of 24 speakers lower and 8 higher
(`experiments/results/hiacc_false_alarm_v1/final_vs_deployed_paired.json`).

**Reading.** The paired interval says these two models differ; it does not say the recipe
helps. Two trainings of the same deployed recipe already differ by 16 points (38.5 and
54.4), and the final recipe sits inside that range. With one seed per recipe the effect of
the recipe on spontaneous speech cannot be separated from training noise, and every model
stays far above its 1% on read lectures. Neither more read voices nor a real codec teaches
the detector what spontaneous speech sounds like; that needs spontaneous training data.

Results: `experiments/results/hiacc_false_alarm_v1/final_s1234_channel.json` and its
per-clip scores. The larger encoder was not run here: its trained models were not kept.
