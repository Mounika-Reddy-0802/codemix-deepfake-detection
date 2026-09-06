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
| Real MUCS source clips | **21.8 Hz** | 110 Hz | 1493 |
| RVC converted clips | **21.1 Hz** | 109 Hz | 1493 |
| XTTS-v2 pilot clips (re-measured) | **27.8 Hz** | 109 Hz | 23 |
| XTTS-v2 **full scale run** (P-023) | **18.99 Hz** | 106 Hz | 3,991 |

Pitch-range retention: **96.4%** of the real speech it started from.

> **The CM02 half of P-019's prediction holds.** Converted clips keep 96.4% of the
> source's pitch range (21.1 Hz vs 21.8 Hz), measured in one pass with one
> estimator, so this is a single difference rather than two measurements compared.
> RVC starts from real speech and the contour survives.

**The CM01 half of P-019 does not survive re-measurement, and the error is in its
real-speech baseline.** P-019 reported 41.1 Hz for MUCS against 25–29 Hz for
XTTS-v2 and concluded XTTS compresses pitch range by ~35%. Re-measuring the same
> **Superseded in part by P-023.** The 27.8 Hz row below is 23 pilot clips. The
> full 4,000-clip CM01 run measures **18.99 Hz** — 13% *narrower* than real speech,
> not wider — so the "opposite directions" reading in this section does not hold.
> Both generators undershoot; RVC retains 96.4% of the source range and XTTS 86.9%.
> The rest of this section is left as written, because the log is append-only.

XTTS pilot clips with `src.data.f0_stats` returns **27.8 Hz**, which reproduces
P-019's XTTS figure exactly — but the same estimator puts real MUCS at **21.8 Hz**,
not 41.1 Hz. On one consistent estimator XTTS-v2 is **27% wider** than the speech
it clones, not 35% narrower. P-019's conclusion was an artefact of comparing two
different measurement methods; see **P-021**.

What this does *not* undo: CM02 is still a distinct attack family, and arguably a
cleaner story than the original. The two generators deviate from real speech in
**opposite directions** — RVC tracks it (96.4%), XTTS overshoots it (+27%) — so
pitch range still separates them, and a detector that has only seen CM01 has not
seen CM02's behaviour.

Caveat on the numbers above: the XTTS row is 23 usable clips from the W3-T5 pilot,
against 1493 for the other two rows. It is enough to show P-019's baseline is
wrong; it is not yet the number for the paper. The CM01 scale clips
(`data/generated/xtts_v2`, ~4,000) should be re-measured with the same call once
they are recovered from their own archive.

Reproduce the XTTS row:

```bash
python - <<'PY'
import glob
from src.data.f0_stats import clip_f0_stats, summarise
from src.utils.audio_utils import load_wav
rows = [clip_f0_stats(*load_wav(p)) for p in sorted(glob.glob("<xtts clips>/**/*.wav", recursive=True))]
print(summarise(rows))
PY
```

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

### Where the run itself lives

The audio and the weights are in a **private** Kaggle dataset,
`saikrishnareddy9/rvc-cm02-generation-archive` (2.19 GB): the 1500 converted
clips as `rvc_converted_wavs.zip`, the 12 `rvc_*.pth` weights, and the 12
`added_*.index` files. It is private and stays private — these are cloned voices
of identifiable speakers and the mentor sign-off covers generating them for
research, not publishing them.

Only the twelve **final** weights are archived. `assets/weights/` accumulates 36
`.pth` because training snapshots at `_e50` and `_e100` alongside the final, so
the archive is built from the twelve `weight_file` names in the manifest rather
than from a glob — a glob would add 1.3 GB and make "which weights produced these
clips?" ambiguous. The pretrained assets (`hubert_base`, `rmvpe`,
`pretrained_v2`) and the per-target `logs/<exp>/` extraction intermediates are
excluded: both regenerate, and together they were 5 GB of the 7.28 GB session
output.

Verify a copy of the archive against this repository:

```bash
python -m src.data.rvc_archive --root /kaggle/input/rvc-cm02-generation-archive
```

`src.data.rvc_archive` re-hashes every weight and index against
`outputs/rvc_model_checksums.json` and checks every clip named in the metadata is
present. It exits non-zero if anything is missing, truncated, or has a different
hash — the last of which is the only way to notice that a *re-run's* weights have
quietly replaced this run's.

First verification, 2026-09-06, against repo commit `f79619e`:

| | Result | Checked how |
|---|---|---|
| Weights | **12/12** | name + size + SHA-256 |
| Indexes | **12/12** | name + size + SHA-256 |
| Converted clips | **1500/1500** | present by name |
| **Total** | **1524/1524** | exit 0, `COMPLETE` |

The clip row is presence, not content: the metadata log carries no per-clip hash,
so a renamed or missing clip is caught and a corrupted one is not. Extracting the
archive is itself a CRC check of every member, and `src.data.generation_qa` is
what screens the audio — its report is in `docs/qa/`, 1404/1500 passing.

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
