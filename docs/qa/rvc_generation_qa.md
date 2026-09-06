# RVC generation — mechanical QA (W4-T2)

`src.data.generation_qa` screened every converted clip. The screen exists because
a generator can return **valid audio that is completely wrong** and raise nothing:
in the 40-clip script pilot one job produced 0.83 s of audio from a 150-character
transcript, was logged as a success, and was counted. Nobody listens to
1500 clips, so this has to be mechanical.

| Check | Threshold | Clips flagged |
|---|---|---|
| Speaking rate | 6–30 chars/s | 89 |
| Near-silence | RMS < 0.01 | 7 |
| Clipping | > 1% of samples at full scale | 0 |
| Too short to hold a code-switch | < 1.0 s | 0 |
| Missing file | — | 0 |

**Verdict: the CM02 corpus does NOT clear the bar as generated.** 96 of 1500 clips fail the mechanical screen (6.4%), which is too many to leave inside a training corpus.

Median speaking rate is 11.1 chars/s. Full per-clip
report: `docs/qa/rvc_generation_qa.csv`.

| Reason | Clips |
|---|---|
| speech too slow / stalled (5 chars/s) | 31 |
| speech too slow / stalled (4 chars/s) | 25 |
| speech too slow / stalled (6 chars/s) | 20 |
| speech too slow / stalled (3 chars/s) | 7 |
| near-silent (rms 0.0001) | 2 |
| near-silent (rms 0.0003) | 2 |
| speech too slow / stalled (2 chars/s) | 2 |
| speech too fast / truncated (40 chars/s) | 2 |
| near-silent (rms 0.0002) | 2 |
| speech too slow / stalled (1 chars/s) | 2 |
| near-silent (rms 0.0004) | 1 |


## The objective measurement agrees

The pilot verdict for CM01 (`docs/qa/pilot_script_verdict.md`) paired the ratings
with intra-utterance pitch range, and that is what separated a legitimate attack
from an easy one. The same measurement, on this corpus:

| | f0 IQR |
|---|---|
| Real spontaneous speech (HiACC), P-019 | 42.2 Hz |
| Real read speech (MUCS), P-019 | 41.1 Hz |
| **Real MUCS source clips, re-measured in this run** | **21.8 Hz** |
| **RVC converted clips, this run** | **21.1 Hz** |
| XTTS-v2, every configuration tried (P-019) | 25–29 Hz |

The real column is re-measured rather than quoted: P-019's numbers were taken ad
hoc with no committed code, so quoting them against a different estimator would
compare tooling, not attacks. `src.data.f0_stats` now pins the method, and
measures both sides of this comparison in one pass.

> P-019's prediction HOLDS: converted clips keep 96.4% of the real pitch range (21.1 Hz vs 21.8 Hz), where XTTS-v2 kept only 25-29 Hz. RVC starts from real speech, so the contour is human and the CM01 compression does not occur.

## What this does and does not decide

**Decided — CM02 is the harder attack family the plan assumed.** Pitch-range
retention is 96.4% of the real speech. The CM01
flattening is a property of text-to-speech, not of cloning in general, so a
detector that has learned to spot a flat contour will not transfer to CM02.

**Decided — the corpus enters training as generated**, subject to the screen
above. Every clip carries the target speaker's id, and both endpoints are
train-pool, so the pool disjointness that makes CM01 auditable holds identically
here.

**Not decided — perceptual quality.** This is a mechanical screen and a pitch
statistic. Neither says whether a listener would be fooled; the CM01 pilot needed
human raters for that, and CM02 has not had them. Nothing here should be read as
a claim about how convincing the clips are.
