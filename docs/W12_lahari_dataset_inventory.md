# Week 12 — Lahari — dataset inventory, and the datasheet corrected

Two jobs: put a measured number against every corpus and every attack, and fix the
place where the datasheet had gone out of date and was saying something untrue.

## What I did, and why it is built this way

### The inventory is arithmetic, not a restatement

`src/reporting/datasets.py` counts the manifests and **measures** the audio.
Durations come from `clip_index.csv` where it has them, and from reading the audio
headers otherwise — 151,818 files, cached so a re-run is instant. The datasheet
stays the written record; this is the arithmetic behind it, regenerable on demand,
so a number that drifts shows up as a difference instead of being repeated.

Audio not present on the machine is counted and marked **unmeasured**. It is never
estimated. That rule caught a real mistake: the XTTS outputs looked absent and
2,217 clips were reported unmeasured until I found they were in a directory the
search had not been pointed at.

### ASVspoof's attacks are kept separate from ours

A first version mixed A01–A19 into the same table as CM01–CM04. They are English,
they are not ours, and putting them together makes both halves unreadable. They now
have their own section.

### "Generated" and "used in an experiment" are different numbers

Both are reported. The generator produced 4,025 XTTS clips; the manifests reference
7,497 rows of XTTS audio, because clips are reused across clean and channel copies.
Reporting one number and calling it the other would be wrong in either direction.

## Numbers

**Bonafide code-mixed corpora**

| Corpus | Clips | Hours | Speakers |
|---|---:|---:|---:|
| MUCS 2021 Hindi–English | 52,825 | 89.5 | 520 |
| HiACC (adult subset only) | 3,318 | 3.2 | 24 |
| **Total** | **56,143** | **92.8** | **544** |

**ASVspoof 2019 LA** — 121,461 clips, 109.6 hours; 12,483 real (11.8 h) against
108,978 fake (97.8 h), at 8.7 : 1.

**Attacks we generated**

| Attack | Tool | On disk | Hours | Used in manifests | Hours used |
|---|---|---:|---:|---:|---:|
| CM01 + CM03 | XTTS-v2 | 4,025 | 7.0 | 7,497 | 13.4 |
| CM02 | RVC v2 | 1,500 | 2.6 | 2,435 | 4.3 |
| CM04 (held out) | Tortoise-TTS | 500 | 1.2 | 458 | 1.1 |

### The datasheet correction

Section 2 and limitation 3 both said CM04 "has **not** been scored yet" and that the
shortcut question was "answerable in principle and unanswered in fact". It has been
scored against every system since. The datasheet now carries the table, and the
answer is more interesting than a simple yes:

| System | Clean (floor 31.21%) | Channel (floor 25.79%) |
|---|---:|---:|
| S1 baseline | 41.67% ✗ | 44.15% ✗ |
| S2 LoRA, XTTS | **5.44%** ✓ | 28.16% ✗ |
| S2 LoRA, XTTS+RVC (deployed) | **12.01%** ✓ | 30.99% ✗ |
| S3 native, XTTS | 27.72% ✓ | **13.71%** ✓ |
| S3 native, XTTS+RVC | 21.39% ✓ | 31.67% ✗ |

On clean audio the adapters generalise to a tool they have never seen. Over a
telephone line the deployed model sits **above its own shortcut floor**, so against
an unseen tool on a real phone line it has no demonstrated detection ability. All
ten values were checked programmatically against `systems_s1_s2_s3.json`.

## Honest limitations

- **One manifest's hours are not unique audio.** `gap_codemix_clean`'s bonafide rows
  point at whole MUCS lecture recordings, not segmented clips, and repeat the same
  recording many times: 359.5 h by row, **3.8 h** of unique audio across 25 files.
  Footnoted in the inventory rather than silently summed.
- **Speaker counts are per manifest**, so a speaker appearing in two manifests is
  counted in both. Only the corpus table gives a true distinct count.
- **The duration cache is not committed** (4.9 MB, gitignored). A fresh checkout
  re-measures, which takes about nine minutes.
- **The datasheet's own attack table still lists CM05–CM09 as planned.** They do not
  exist and the text says so, but the rows remain, which reads as roadmap rather
  than inventory.

## How to run it

```bash
DATA_ROOT=/path/to/dfdata python -m src.reporting.datasets
pytest tests/test_datasets_inventory.py
```

## What's next

- Fold the inventory tables into the report's dataset chapter.
- Decide whether the CM05–CM09 rows stay in the datasheet or move to future work.

---

# Later in week 12 — the data chapter and two review tests

The external review (`docs/external_review_response.md`) asked for more speakers and
speaking styles (comment 3) and for the cost of the phone-line adaptation on English
(comment 4). I took the data side of both, and wrote the data chapter.

## What I did, and why it is built this way

### The data chapter says who each model hears

`paper/sections/dataset.tex` now has a speaker table (`sec:data-speakers`) and a
section on speaking styles (`sec:data-styles`). Writing it turned up a wrong number:
the paper said the deployed model was adapted on 10 speakers. That is only the XTTS
pool. RVC adds 13 more, so the deployed model hears **21**, and it is tested on 51
speakers it has never heard. `methodology.tex` was corrected to match.

### A false-alarm test, because HiACC has no fakes

Every Hindi-English test set so far is MUCS lecture audio: read, prepared speech.
HiACC's adult partition is the only other Hindi-English source we may use, and only
for evaluation. It holds genuine speech only, so it cannot give an EER. What it can
give is the share of genuine speech called fake, which is the error a new caller
meets first. `src/training/false_alarm.py` measures that at the deployed threshold,
with an interval over whole speakers, and compares it against the same model's
genuine evaluation-pool clips so the number has a reference. The child partition was
not touched.

### English over the phone line, on a fixed sample

Scoring all 71,237 ASVspoof clips through the phone-line chain on the CPU would take
about five hours per model. `src/data/stratified_subset.py` draws a fixed sample
instead: 308 clips of each of the 13 attacks and 2,000 genuine clips, from all 67
speakers, seed 1234. Equal clips per attack keep one attack from dominating the EER.
The sample rebuilds byte-for-byte from the full manifest, so the copy with local
paths is gitignored and only the portable versions are committed.
`src/reporting/english_channel.py` reports, besides the EER, the share of genuine
English flagged and of fakes caught at the deployed threshold, and the EER per
attack, since EER alone hides what a caller would actually see.

## Numbers

| Test | Model | Result |
|---|---|---|
| HiACC adult, 24 speakers, spontaneous, phone line | deployed S2 | **54.37%** [44.43, 64.15] of genuine clips called fake |
| MUCS read lectures (reference), 15 speakers | deployed S2 | 0.89% [0.54, 1.29] called fake |
| HiACC per speaker | deployed S2 | 13.9% to 89.5%, all 24 above 10% |
| English sample, clean | deployed S2 | 2.30% EER [1.84, 2.85] (full set: 2.42%) |
| English sample, phone line | deployed S2 | **24.20%** EER [23.27, 25.30]; 7.45% of genuine flagged, 59.97% of fakes caught |
| English sample, phone line | S1 | 6.50% EER [5.92, 7.19]; 0.65% of genuine flagged |
| English sample, clean | S1 | 0.51% EER [0.35, 0.79]; 0.20% of genuine flagged (full set: 0.85%) |

Both tests point the same way: adaptation taught the model what genuine *lecture
recordings over a phone line* sound like, and genuine speech that sounds different
(other speakers talking freely, or English over the same line) is rejected.

## Honest limitations

- **S1 on the clean English sample took three attempts.** The first two crashed when
  the laptop ran out of memory while another job held it; the third completed
  (0.51%, close to the 0.85% on the full set).
- **One channel only** (G.711 at 20 dB) for both tests.
- **The English phone-line numbers come from a sample.** The clean check (2.30%
  against 2.42% on all clips) suggests it is representative; the phone-line numbers
  have not been checked against the full set.
- **HiACC's source manifest has machine-local paths** and no code to rebuild it; it is
  committed because it is the record of which 3,318 clips were used.
- **HiACC cannot be used to fix the false alarms**: it is evaluation-only by the
  project's dataset rules.

## How to run it

```bash
python -m src.data.stratified_subset --manifest data/manifests/asvspoof_eval.csv --out data/manifests/asvspoof_eval_subset6k.csv
python -m src.reporting.english_channel
pytest tests/test_stratified_subset.py tests/test_english_channel.py tests/test_false_alarm.py
```
The full commands for normalising, rendering and scoring both tests are in
`docs/results/hiacc_false_alarm_v1.md` and `docs/results/english_channel_v1.md`.

## What's next

- Run the HiACC test on a variant model (augmentation, layer mix), to see whether any
  of them lowers the false alarms. The Kaggle runs did not keep their checkpoints, so
  this needs one variant retrained with its checkpoint saved.
