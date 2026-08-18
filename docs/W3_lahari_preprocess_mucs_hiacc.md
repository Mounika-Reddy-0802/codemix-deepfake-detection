# W3 — Lahari: MUCS/HiACC preprocessing + child-quarantine verification

Branch: `week3-lahari-preprocess-mucs-hiacc` · Tasks: **W3-T2**, **W3-T4** (ranking half)

## What I built

**1. Closed a hole in the child-audio quarantine (the important one).**
`preprocess_dir` walked a corpus root with `rglob("*.wav")` and had no exclusion
for `_EXCLUDED_children/`. `scripts/02_preprocess_all.sh` printed
*"(HiACC child folder is quarantined and will not be processed)"* and then handed
the whole HiACC root to that walk. So the reassuring line was wrong: running
preprocessing today would have written child segments into
`data/processed/clean/hiacc/`. Fixed at the source — `preprocess.audio_files()`
now skips anything under `EXCLUDED_DIR_NAMES` at any depth, and the script refuses
to preprocess HiACC when the audit warns.

**2. `src/data/quarantine.py` — the audit that produces the W3-T2 evidence.**
The download script's `child|kid|minor` folder regex is a *guess* about HiACC's
layout. The audit reports what is actually on disk: child-looking directories and
files, whether each is inside quarantine, and — the case the plan warns about —
age-like manifest columns (`age`, `age_group`, `speaker_type`, …) when the folder
regex matched nothing, which means the split is by metadata and the automatic
quarantine is a no-op. It **never returns a pass**; it renders
`docs/qa/child_quarantine_check.md` for a human to sign.

Also fixed the pattern itself: `child(?:ren)?\b` never matches `child_07`, because
`_` is a word character so `\b` does not fire. Now bounded by letter lookarounds
(`(?<![a-z])child(?:ren)?(?![a-z])`), which matches `child_07` and `spk_child`
while still rejecting `kidney` and `childish`.

**3. `src/data/speaker_selection.py` — the scripted half of W3-T4.**
Per-clip SNR (90th-vs-10th-percentile frame energy), aggregated per speaker,
ranked **SNR first, then duration** so hours of noisy audio do not outrank half a
minute of clean audio. The shortlist returns only eligible speakers and is never
padded to reach 30 — `enough_speakers()` is how you find out the corpus came up
short. The clip index refuses to walk quarantined directories.

**4. `src/data/listening_test.py` — the channel-sim listening test (W3-T2).**
Samples 20 clips stratified across MUCS and HiACC, renders clean + channel-matched
pairs through the same `ChannelConfig` the eval pipeline uses, and writes a rating
sheet (one row per clip per rater). `summarise_ratings()` reports a partially
filled sheet as incomplete rather than averaging whatever is present.

## How to run it

```bash
# quarantine audit -> the report a human signs (safe with no corpus on disk)
python -m src.data.quarantine --root data/raw/hiacc --out docs/qa/child_quarantine_check.md

# after downloads: preprocessing now gated by that audit
bash scripts/02_preprocess_all.sh

# speaker ranking (writes clip index + ranking to data/manifests/)
python -m src.data.speaker_selection --root data/processed/clean/mucs2021 --source mucs2021

# listening test: sheet only (no audio) or full render
python -m src.data.listening_test --sheet-only
python -m src.data.listening_test
```

## Numbers — measured on the real corpora

Downloads ran on 12 Aug; everything below is from the corpora on disk, not from
synthetic trees.

| | |
|---|---|
| MUCS utterances / speakers | **52,825 / 520** (89.55 h) |
| HiACC adult clips / speakers | **3,318 / 24** (3.22 h) |
| Total indexed | **56,143** clips across 544 speakers |
| HiACC child files quarantined | **1,858** (of 5,176 total audio) |
| Child ids reachable from any manifest | **0** |
| Channel, G.711 @ 20 dB | 17.62 dB SNR, correlation 0.991, **20/20** |
| Channel, AMR-NB | 5.57 dB SNR, correlation 0.885, **20/20** |
| Listening test | telephony **4.0/5**, intelligibility **4.0/5**, 60/60 rows |
| Tests | 56 added in Week 3; suite now **410 passed**, ruff clean |

Both machines index to identical counts, which is the point — the eval box and the
generation box must agree or nothing transfers between them.

**Defects found and fixed this week:** the quarantine bypass in the preprocessing
walk (ethics-critical); the `\b`-vs-`_` child-pattern regex; the `grep -P` +
`(?i)`-under-`-E` double failure that made the download sweep report "moved 0
folder(s)" on a corpus with 1,858 live child files; the AMR-NB codec-delay
misalignment that failed 15/20 pairs on a working chain (**P-013**); and the audit
overwriting its own signed report.

## Verify

```bash
python -m src.data.corpora --data-root $DATA_ROOT     # 52,825/520 and 3,318/24
python -m src.data.quarantine --root $DATA_ROOT/raw/hiacc
python -m src.data.channel_qa                         # 20/20
```

## What the corpus turned out to be

Two findings that changed downstream work:

- **MUCS is a Kaldi corpus**, not a folder of clips — 521 long recordings with
  speakers in `utt2spk` and spans in `segments`. The original blind chunking
  destroyed speaker identity entirely, so `02_preprocess_all.sh` was superseded by
  corpus-aware indexing (`src.data.corpora`).
- **HiACC splits adult from child by top-level folder**, confirmed from
  `Corpus/readme.txt` — which is what the quarantine sweep keys on, so it acts on
  the documented layout rather than a guess. Corroborated by metadata: `AD*` ages
  19–42, `CH*` ages 10–14, disjoint id sets.

## Honest limitations

- **AMR-NB is ear-unchecked.** Verified by measurement only (20/20); the listening
  pass at `<DATA_ROOT>/processed/listening_test_amr/` is outstanding, so that column
  must not be reported as listened-to.
- **The speaker shortlist rests on a proxy.** Ranking used a dynamic-range measure
  (24.5–92.4 dB spread), *not* calibrated SNR, and the eligibility gate passed all
  520 speakers — so the ordering is a hint, not a quality guarantee. Spot-check the
  top of `speaker_shortlist.csv` by ear before Week-4 scale.
- **`AD63` is undocumented** — audio but no metadata row (and `AD65` the reverse).
  Adult-prefixed, so no child audio enters the pipeline, but its age is unverified:
  it must not be a cloning reference until HiACC confirms it.
- **`processed/` is not used downstream.** Nothing after `corpora.py` reads it; it
  is regenerable and deliberately not copied between machines.

## What's next

- **AMR-NB listening pass** (L, M, SK) — the last open item on W3-T2.
- **Confirm the top of the shortlist by ear** before generation at scale.
- **Email HiACC** about the `AD63`/`AD65` metadata mismatch.
- **Generation at scale** (W4-T1) once the pilot rating lands.
