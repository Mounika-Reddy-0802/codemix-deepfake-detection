# CM04 held-out Tortoise generation -- mechanical QA (W4-T3)

src.data.generation_qa screened every clip generated for the held-out attack
(tool=tortoise, pool=eval). The screen tests the relationship between each
transcript and its audio, because a generator can return valid audio that is
completely wrong and raise nothing.

| | |
|---|---|
| Clips screened | 159 |
| Passed | 150 |
| Failed | 9 |
| Pass rate | 94.34% |
| Median speaking rate | 9.3 chars/s |

| Reason | Clips |
|---|---|
| speech too slow / stalled (5 chars/s) | 4 |
| near-silent (rms 0.0063) | 1 |
| near-silent (rms 0.0052) | 1 |
| near-silent (rms 0.0032) | 1 |
| near-silent (rms 0.0072) | 1 |
| near-silent (rms 0.0031); speech too fast / truncated (68 chars/s) | 1 |

Full per-clip report: docs/qa/heldout_generation_qa.csv
