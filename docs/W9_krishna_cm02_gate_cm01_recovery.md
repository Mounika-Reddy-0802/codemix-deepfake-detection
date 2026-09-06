# W9 — Krishna: the CM02 shortcut gate, CM01 recovered, CM04 staged

Branch: `week9-krishna-cm01-recovery` · Tasks: **W5-T4** (CM02 half), **W4-T1 recovery**, **W4-T3** (prepared)

**Outcome:** the shortcut gate was asked of CM02 and answered — RVC's tell is
spectral, not level, and 16× weaker than CM01's. The 4,000-clip CM01 corpus, which
existed on one laptop and nowhere in git, is now described, archived and verified —
and re-measuring its pitch **overturned P-021**. CM04 is staged but not generated:
the job table is committed and the notebook is written, and it still needs a GPU
session.

---

## What I did

### 1. The shortcut gate, asked of CM02 (W5-T4)

`lowlevel_cue_check_v1.md` failed at **1.39% EER** on CM01, where chance is 50%. CM02
allows a much sharper question, because every RVC job records the source segment it
converted: bonafide = the 1,345 unique MUCS source recordings, spoof = the 1,500
conversions **of those same recordings**. Same words, same room, same microphone —
so anything the gate finds is the conversion and nothing else. Fit/score disjoint by
source speaker, 13/12 of 25.

| Condition | EER | AUC | Verdict |
|---|---:|---:|---|
| raw | **22.42%** | 0.8537 | ❌ FAIL |
| normalised | **22.28%** | 0.8599 | ❌ FAIL |
| *CM01 clean, for comparison* | *1.39%* | *0.9990* | *FAIL* |

Both fail the near-chance band and both are **16× further from a shortcut than
CM01**. Neither half of that should be dropped when it is quoted.

**The mechanism is the finding.** Normalisation moved the CM01 gate 3.7×; here it
buys **0.14 pp**, because the tell was never in the levels. RMS mean and std
equalise to within 1% and the gate does not move. What carries it is scale-invariant:
`zero_crossing_rate` 0.1324 → 0.0798 (40% lower, top coefficient in both conditions)
and `hf_energy_ratio` 0.0047 → 0.0023 (half). RVC's decoder reconstructs from pitch
and content features and does not reproduce the source's high-frequency detail —
room noise, breath, fricatives. It keeps the prosody and loses the top of the
spectrum. Recorded as **P-022**.

### 2. CM01 recovered from one laptop into git (W4-T1 recovery)

The corpus was generated in Week 4 on a GPU laptop and lived **only there**.
`scale_generation_jobs.csv` records what was *requested*, and `generate_batch` skips
failures, so the jobs table is an upper bound rather than a record of what exists.

- **Metadata:** 4,000 records, 4,000 clips on disk — the two agree exactly, so
  nothing was generated-then-lost or lost-then-recorded. Committed as
  `outputs/xtts_generation_metadata.jsonl` with every path rewritten through
  `paths.portable`, checked mechanically for `C:/`, `C:\`, `/kaggle/`, `/content/`.
- **Archive:** 842 MB zipped to the private Kaggle dataset
  `saikrishnareddy9/xtts-cm01-generation-archive`. Audio is never committed — it is
  cloned speech of identifiable people and this repo is public.
- **Verified:** `rvc_archive` previously assumed a model manifest always exists. CM01
  is zero-shot: there are no per-speaker checkpoints to hash. It now takes
  `--no-models`, and `expected_models(None)` returns an empty list rather than
  raising, because "this run trained no models" is a different condition from "the
  manifest is missing". Result: **4,000/4,000 clips present**.

### 3. Pitch re-measured on the full run — and it changes the answer (P-023)

P-021 closed with an explicit open item: its XTTS figure rested on **23 pilot clips**
against 1,493 for the others, and it said so. Measured with the identical call over
all 4,000:

| Family | Median f0 IQR | Clips | Retention |
|---|---:|---:|---:|
| Real MUCS source | 21.84 Hz | 1,493 | — |
| CM02 (RVC) | 21.06 Hz | 1,493 | 96.4% |
| **CM01 (XTTS, full run)** | **18.99 Hz** | **3,991** | **86.9%** |
| *CM01 (23 pilot clips)* | *27.8 Hz* | *23* | *superseded* |

**P-021's headline does not hold.** It claimed the two generators deviate from real
speech in *opposite* directions — RVC tracks it, XTTS overshoots. At scale both
undershoot: XTTS is about 3.7× further from real speech but on the same side of it.
What survives is that pitch range still separates the families, now as a magnitude
difference in one direction, and the CM01 number is the corpus's rather than a
pilot's. P-019's *direction* is restored; only its 41.1 Hz baseline stays retired.

### 4. CM04 staged, not generated (W4-T3)

500 Tortoise jobs over the 15 eval-pool speakers, 33–34 each, 500 unique
transcripts, committed as `data/manifests/heldout_generation_jobs.csv` **before** any
GPU is booked. `pool_jobs` already built eval-pool tables; it only lacked a `--tool`
flag, so CM04 reuses that builder rather than adding a parallel one.

Firewall verified on the committed table: `tool == tortoise` and `pool == eval` on
every row, **zero** train-pool speakers, no duplicate (speaker, transcript) pairs.

`notebooks/kaggle_w4t3_heldout_tortoise.ipynb` drives it. Writing it caught two API
drifts in my own first draft: `generate_heldout_batch` takes no `failures` argument,
and `generation_qa` exposes `screen`/`summarise` rather than a `screen_clips`. Both
corrected against the real modules before commit.

---

## Numbers

| | |
|---|---|
| CM02 shortcut gate | **22.42% raw / 22.28% normalised**, 1,369 scored clips, 0 missing |
| CM01 metadata recovered | **4,000** records = **4,000** clips on disk |
| CM01 archive | 842 MB, **4,000/4,000** verified present |
| CM01 pitch (full run) | **18.99 Hz** median f0 IQR, 3,991 usable of 4,000 |
| CM04 job table | **500** jobs, 15 eval-pool speakers, 0 train-pool leakage |
| Problems recorded | **P-022**, **P-023** |

## Honest limitations

- **CM04 does not exist yet.** Everything up to the GPU session is done; the clips,
  the QA report and the results doc are not. Until then the "not shortcut artefacts"
  claim still has no held-out-tool evidence.
- **The CM01 pitch comparison is not paired.** The 21.84 Hz baseline is the CM02
  run's own source segments — same corpus and train pool, but different utterances.
  A paired measurement needs CM01 regenerated from the CM02 transcripts, which is not
  worth a GPU session for a 13% effect that is consistent in direction.
- **CM01 has no QA artefact in the repo.** L's Week-4 screen found the 116 bad clips
  and the 987461 reference problem, but those numbers live in `docs/progress.md`
  rather than a `docs/qa/` report the way CM02's does.
- **The two rejected CM01 clips are not identified.** The log has 4,000 records and
  the usable count is 3,998; nothing marks *which* two.
- **This is my third branch in week 9**, which breaks the one-branch-per-person-per-week
  rule. The other two were already merged before the rule was clarified; everything
  since is consolidated here.

## How to run it

```bash
export DATA_ROOT=/c/dfdata

# the CM02 gate
python -m src.data.rvc_gate --clips <1,500 rvc_*.wav> --data-root "$DATA_ROOT"

# CM01 archive verification (clips-only: zero-shot, no models to hash)
python -m src.data.rvc_archive --root "$DATA_ROOT/generated/xtts_v2/outputs" \
    --no-models --metadata outputs/xtts_generation_metadata.jsonl

# CM04 job table (already committed; this regenerates it identically)
python -m src.data.pool_jobs --pool eval --tool tortoise --n 500 --no-refs
```

## What's next

1. **Run CM04 on a Kaggle GPU** — the last genuinely unseen attack.
2. **Score a detector on CM02.** No model has been run against it yet, and it now
   has a 22.3% shortcut baseline to beat rather than CM01's 1.39%.
3. **Run the gate on the channel-matched condition.** Band-limiting moved CM01 from
   1.39% to 9.25%; CM02's tell *is* high-frequency, so it should compress harder.
