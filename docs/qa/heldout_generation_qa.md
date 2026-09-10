# CM04 held-out Tortoise generation -- mechanical QA (W4-T3)

src.data.generation_qa screened every clip generated for the held-out attack
(tool=tortoise, pool=eval). The screen tests the relationship between each
transcript and its audio, because a generator can return valid audio that is
completely wrong and raise nothing.

| | |
|---|---|
| Clips screened | 500 |
| Passed | 458 |
| Failed | 42 |
| Pass rate | 91.6% |
| Median speaking rate | 9.45 chars/s |

| Reason | Clips |
|---|---|
| speech too slow / stalled (5 chars/s) | 13 |
| speech too slow / stalled (6 chars/s) | 11 |
| near-silent (rms 0.0024) | 3 |
| near-silent (rms 0.0072) | 2 |
| near-silent (rms 0.0063) | 1 |
| near-silent (rms 0.0032) | 1 |
| near-silent (rms 0.0031); speech too fast / truncated (68 chars/s) | 1 |
| near-silent (rms 0.0031) | 1 |
| near-silent (rms 0.0052) | 1 |
| near-silent (rms 0.0030) | 1 |
| near-silent (rms 0.0065) | 1 |
| near-silent (rms 0.0012); speech too fast / truncated (72 chars/s) | 1 |
| near-silent (rms 0.0020) | 1 |
| near-silent (rms 0.0029); speech too fast / truncated (35 chars/s) | 1 |
| near-silent (rms 0.0099) | 1 |
| near-silent (rms 0.0053) | 1 |
| near-silent (rms 0.0013) | 1 |

Full per-clip report: docs/qa/heldout_generation_qa.csv
