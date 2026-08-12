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

## Numbers

- **56 tests** added/passing across `test_preprocess_quarantine.py` (7),
  `test_quarantine.py` (15), `test_speaker_selection.py` (20),
  `test_listening_test.py` (14). Full suite green, `ruff` clean.
- **1 ethics-critical defect found and fixed** (quarantine bypass in the
  preprocessing walk), **1 regex defect fixed** (`\b` vs `_` in the child pattern).
- **0 corpus numbers.** MUCS and HiACC are not downloaded — no speaker counts, no
  SNR distribution, no hours. Everything above is verified on synthetic trees.

## What's next (blocked on humans)

- **Downloads have not run** — waiting on the explicit "run downloads now". Until
  then the quarantine audit reports `Root exists: True, Audio files: 0` and proves
  nothing about HiACC.
- **`docs/qa/child_quarantine_check.md` is unsigned.** The regex-vs-real-layout
  question can only be closed by reading the HiACC documentation.
- **Listening test unrated** — needs L, M, SK to actually listen to 20 pairs.
- **Speaker pools not frozen** — the ranking is ready; the selection needs the team
  listening pass, and the carve/freeze half is SK's (`week3-krishna-xtts-rvc-pilot`).
