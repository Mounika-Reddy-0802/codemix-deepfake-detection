# CM01 recovered: the 4,000-clip XTTS run is now described in git

**W4-T1 recovery, owner SK.** The CM01 corpus was generated in Week 4 on a GPU
laptop and then existed **only on that laptop**. Nothing in the repository recorded
what the run actually produced: `data/manifests/scale_generation_jobs.csv` says what
was *requested*, and `generate_batch` skips failures, so the jobs table is an upper
bound rather than a record. This closes that gap and re-measures the one number
P-021 left explicitly open.

## What was actually on the machine

| | |
|---|---|
| Metadata records | **4,000** (`generation_metadata.jsonl`, 1.65 MB) |
| Clips on disk | **4,000** |
| Usable after QA (L, W4-T6) | **3,998** (99.95%) |
| Total audio | 6.91 h over 25 train-pool speakers, 160 each |
| Archive | 842 MB zipped |

Metadata count and file count agree exactly, so nothing was generated-then-lost or
lost-then-recorded. The two clips separating 4,000 from 3,998 are the borderline-slow
rejects L recorded in Week 4; both their speakers still keep 159 clips.

## 1. The log is now in git, portable

`outputs/xtts_generation_metadata.jsonl` — 4,000 records, every path rewritten
through `src.utils.paths.portable` so no machine-local absolute path enters the
repository (precedent: commit `f60f989`). CM01's paths carry a `generated/` layout
anchor, so they rebase cleanly:

```
C:\dfdata\generated\xtts_v2\outputs\xtts_v2_106718_00001.wav
  -> ${DATA_ROOT}/generated/xtts_v2/outputs/xtts_v2_106718_00001.wav
```

This differs from CM02, whose conversions were written to `/kaggle/working` — no
anchor to rebase on, so those were reduced to basenames. Both are portable; CM01's
form simply carries more.

Checked mechanically before commit: no `C:/`, `C:\`, `/kaggle/` or `/content/`
substring survives anywhere in the file.

## 2. The audio is archived and verified

Uploaded to the private Kaggle dataset **`saikrishnareddy9/xtts-cm01-generation-archive`**,
the same reasoning as CM02: this is cloned speech of identifiable people and the
project repository is public, so the audio is never committed — only its description.

`src.data.rvc_archive` verified it. CM01 is zero-shot TTS, so unlike CM02 there are
**no per-speaker models to hash** — the module previously assumed a model manifest
always exists. It now accepts a clips-only archive:

```bash
python -m src.data.rvc_archive --root <archive> \
    --no-models --metadata outputs/xtts_generation_metadata.jsonl
```
```
clips-only archive check: .../xtts_v2/outputs
Models and indexes: none expected (zero-shot run, nothing to hash)
Converted clips: 4000/4000 present
COMPLETE -- every clip is present.
```

`expected_models(None)` and a missing manifest both return an empty list rather than
raising, because "this run trained no models" is a different condition from "the
manifest is missing" and only the first is normal. Four tests pin it.

## 3. Pitch, re-measured on the full run — and it changes the answer

P-021 closed with an explicit open item: its XTTS figure rested on **23 pilot clips**
against 1,493 for the others, and it said so — *"enough to retire the 41.1 Hz
baseline, not enough to be the paper's number."*

Measured with the identical call (`src.data.f0_stats`, Praat autocorrelation,
60–400 Hz) over all 4,000 clips:

| Family | Median f0 IQR | Median f0 | Clips | Retention vs real |
|---|---:|---:|---:|---:|
| Real MUCS source | 21.84 Hz | 109.6 Hz | 1,493 | — |
| CM02 (RVC conversions) | 21.06 Hz | 109.4 Hz | 1,493 | **96.4%** |
| **CM01 (XTTS, full run)** | **18.99 Hz** | 105.9 Hz | **3,991** | **86.9%** |
| *CM01 (XTTS, 23 pilot clips)* | *27.8 Hz* | *109 Hz* | *23* | *127% — superseded* |

**The pilot figure does not survive the scale run**, which is 173× larger.

Three measurements, three answers, and only the last is on the full corpus:

| | XTTS | Baseline | Reading |
|---|---|---|---|
| P-019 | 25–29 Hz | 41.1 Hz (ad hoc tooling) | 35% narrower — **wrong baseline** |
| P-021 | 27.8 Hz (23 clips) | 21.84 Hz | 27% wider — **unrepresentative sample** |
| **P-023** | **18.99 Hz (3,991 clips)** | 21.84 Hz | **13% narrower** |

### What this costs, stated plainly

**P-021's headline argument does not hold.** It claimed the two generators "deviate
from real speech in opposite directions — RVC tracks it, XTTS overshoots". At scale
both *undershoot*: RVC loses 3.6 pp of pitch range, XTTS loses 13.1 pp, so XTTS is
about 3.7× further from real speech but on the same side of it.

### What survives, on firmer ground

- **Pitch range still separates the two families**, now as a magnitude difference in
  one direction rather than a sign difference. A detector trained only on CM01 has
  still not seen CM02's pitch behaviour.
- **P-019's mechanism was right even though its numbers were not**: XTTS invents
  prosody from text and flattens it; RVC inherits it from real speech. The direction
  P-019 asserted is restored — only the magnitude and the baseline were wrong.
- The CM01 figure is now the paper's number rather than a pilot's.

### Caveat

The 21.84 Hz baseline is the CM02 run's own 1,493 source segments — the same corpus
and the same train pool CM01 clones, but **not the identical utterances**. This is a
like-for-like corpus comparison, not a paired one. A paired measurement would need
CM01 regenerated from the CM02 source transcripts, which is not worth a GPU session
for a 13% effect that is already consistent in direction across two estimators.

## Reproduce

```bash
export DATA_ROOT=/c/dfdata

# the portable log committed here, from the machine-local original
python -c "import json; from pathlib import Path; from src.data.rvc_report import portable_records; \
records=[json.loads(l) for l in Path('$DATA_ROOT/generated/xtts_v2/outputs/generation_metadata.jsonl') \
.read_text(encoding='utf-8').splitlines() if l.strip()]; print(len(portable_records(records)))"

# archive verification (clips-only)
python -m src.data.rvc_archive --root "$DATA_ROOT/generated/xtts_v2/outputs" \
    --no-models --metadata outputs/xtts_generation_metadata.jsonl

# pitch over the full run -> experiments/results/f0_cm01_scale.json
python -c "import glob, json; from src.data.f0_stats import clip_f0_stats, summarise; \
from src.utils.audio_utils import load_wav; \
rows=[clip_f0_stats(*load_wav(p)) for p in sorted(glob.glob('$DATA_ROOT/generated/xtts_v2/outputs/**/*.wav', recursive=True))]; \
print(json.dumps(summarise(rows), indent=2))"
```

## What is still open

- **CM01 has no QA artefact in the repo.** L's Week-4 screen found the 116 bad clips
  and the 987461 reference problem, but the numbers live in `docs/progress.md` rather
  than in a `docs/qa/` report the way CM02's does. Worth normalising.
- **The two rejected clips are not identified** in the committed metadata — the log
  has 4,000 records and the usable count is 3,998, but nothing marks *which* two.
