# How natural are the clones? Predicted MOS (v1)

UTMOS22 (strong) predicted mean opinion score, 1-5, on a seeded random sample of
clean, level-normalised clips per set; 95% bootstrap interval of the mean. UTMOS
was trained on English, so compare sets of the same speakers rather than reading
absolute values. Built by `src/reporting/naturalness.py`.

| Set | Clips | MOS mean [95% CI] | SD |
|---|---|---|---|
| genuine, new test speakers | 200 | 3.17 [3.07, 3.27] | 0.71 |
| CM05 XTTS-v2 (new speakers) | 200 | 2.66 [2.58, 2.75] | 0.63 |
| CM06 Tortoise (new speakers) | 200 | 2.48 [2.38, 2.59] | 0.76 |
| genuine, IndicF5 speakers (all Hindi) | 200 | 3.10 [3.01, 3.18] | 0.63 |
| CM07 IndicF5 | 200 | 3.07 [2.99, 3.15] | 0.57 |
| CM08 XTTS-v2 (adaptation speakers) | 200 | 2.49 [2.41, 2.57] | 0.59 |
