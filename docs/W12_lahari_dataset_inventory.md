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
