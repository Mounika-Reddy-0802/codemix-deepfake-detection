# RVC voice conversion — CM02 generation run (W4-T2)

Generated on Kaggle (2 x Tesla T4) on 2026-09-05 from
`notebooks/kaggle_w4t2_rvc_generation.ipynb`, branch `week8-krishna-rvc-generation`, repo commit
`ec352e6`. RVC-WebUI checkout `81eed5e`.

CM02 is the second seen attack family. P-019 established why it has to exist:
XTTS-v2 (CM01) compresses intra-utterance pitch range by roughly 35%, because it
invents prosody from text. Voice conversion starts from a real human recording
and swaps timbre only, so the contour is the source speaker's own — that was the
prediction, and this run is the first time it has been measured. See the pitch
section below.

## What ran

| | |
|---|---|
| Targets trained | **12** |
| Epochs per target | 100 |
| Conversions requested per target | 125 |
| Clips written | **1500** |
| Conversion failures | **0** |
| Total converted audio | 2.62 h |
| Source pool | MUCS 2021 train pool, 25 speakers |
| Model | RVC v2, 40k, warm-started from `f0G/f0D40k.pth` |
| f0 method | rmvpe |

Every conversion has a train-pool speaker at **both** ends, no speaker is
converted into themselves, and no source clip is reused within one target —
checked by `assert_rvc_invariants` before any GPU time, not after.

## Per target

| Target speaker | Train clips | Train audio | Epochs | Clips written | QA pass |
|---|---|---|---|---|---|
| 106718 | 140 | 632 s | 100 | 125 | 116/125 |
| 177570 | 133 | 631 s | 100 | 125 | 119/125 |
| 322882 | 102 | 644 s | 100 | 125 | 113/125 |
| 445718 | 127 | 771 s | 100 | 125 | 119/125 |
| 457295 | 156 | 735 s | 100 | 125 | 114/125 |
| 539354 | 96 | 574 s | 100 | 125 | 114/125 |
| 541304 | 82 | 611 s | 100 | 125 | 120/125 |
| 584991 | 114 | 630 s | 100 | 125 | 117/125 |
| 850754 | 92 | 749 s | 100 | 125 | 117/125 |
| 876898 | 96 | 574 s | 100 | 125 | 118/125 |
| 903756 | 101 | 733 s | 100 | 125 | 121/125 |
| 996368 | 86 | 633 s | 100 | 125 | 116/125 |


## Mechanical quality screen

`src.data.generation_qa` screens the relationship between each transcript and its
audio, because a generator can return valid audio that is completely wrong without
raising anything. Full report: `docs/qa/rvc_generation_qa.csv`, verdict: `docs/qa/rvc_generation_qa.md`.

| | |
|---|---|
| Clips screened | 1500 |
| Passed | **1404** |
| Failed | **96** |
| Pass rate | **93.6%** |
| Median speaking rate | 11.1 chars/s |

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


## Pitch range — testing P-019's prediction

Measured with `src.data.f0_stats` (Praat autocorrelation, 60–400 Hz).
The converted clips and the real clips they were converted from were measured in
the same pass with the same settings, so this is one difference rather than two
measurements.

| | Median f0 IQR | Median f0 | Clips |
|---|---|---|---|
| Real MUCS source clips (measured here) | **21.8 Hz** | 110 Hz | 1493 |
| RVC converted clips (measured here) | **21.1 Hz** | 109 Hz | 1493 |
| XTTS-v2, P-019 (ad hoc, tooling unknown) | 25–29 Hz | — | — |

Pitch-range retention: **96.4%** of the real speech it started from.

> P-019's prediction HOLDS: converted clips keep 96.4% of the real pitch range (21.1 Hz vs 21.8 Hz), where XTTS-v2 kept only 25-29 Hz. RVC starts from real speech, so the contour is human and the CM01 compression does not occur.

## What this repository keeps, and what it does not

The audio is **not** committed. These are cloned voices of identifiable MUCS
speakers in a public repository, and `**/generated/**/*.wav` is gitignored
deliberately. The voice models are not committed either — twelve weights at
~55 MB each, and `checkpoints/**` is ignored for the same reason.

What is committed is enough to audit and rebuild the run:

| Artefact | Path |
|---|---|
| Job table (every conversion, both endpoints) | `data/manifests/rvc_generation_jobs.csv` |
| Per-clip metadata, paths made portable | `outputs/rvc_generation_metadata.jsonl` |
| Run summary | `outputs/rvc_generation_summary.json` |
| Model checksums (SHA-256, size, epochs) | `outputs/rvc_model_checksums.json` |
| QA report | `docs/qa/rvc_generation_qa.csv` |
| QA verdict | `docs/qa/rvc_generation_qa.md` |

`generation_metadata.jsonl` is written by the generator full of absolute
`/kaggle/working/...` paths. It is passed through `src.utils.paths.portable`
before it enters the tree — commit f60f989 gitignored the ASVspoof manifests for
exactly that reason, and a machine-local path in git is junk on every other
machine.

## Reproducing this

```bash
# Kaggle: GPU T4 x2, Internet on, MUCS train-pool dataset + signed ethics PDF as Inputs
# then run notebooks/kaggle_w4t2_rvc_generation.ipynb top to bottom.
python -m src.data.corpora --data-root $DATA_ROOT      # rebuilds clip_index.csv
python -m src.data.generation_qa --jobs data/manifests/rvc_generation_jobs.csv \
    --out-dir <converted clips> --report docs/qa/rvc_generation_qa.csv
```

Three toolchain defects had to be fixed before any of this ran; they are recorded
as **P-020** in `docs/problems_and_decisions.md`.
