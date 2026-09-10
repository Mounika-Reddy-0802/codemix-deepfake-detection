# W4-T3 — CM04: the held-out Tortoise attack

**Status: done.** 500 of 500 committed job rows generated, screened, archived and
logged. This closes the last structural gap in the attack table: until now every
generated clip came from a tool the models were allowed to see.

Generation ran on Kaggle (GPU T4 ×2) from `notebook835bf0c1f3`, resuming a partial
first session. Metadata and QA are committed in `9766b03`; the audio is not, and
will not be.

---

## 1. What was generated

| | |
|---|---|
| Job table | `data/manifests/heldout_generation_jobs.csv`, committed **before** the run |
| Clips requested | 500 |
| Clips produced | **500** (100% of the table) |
| Tool | Tortoise-TTS 3.0.0, preset `fast` |
| Speaker pool | **eval only** — 15 speakers, 33–34 clips each |
| Language tag | `hi` (code-mixed Hindi-English transcripts, Latin transliteration) |
| Total audio | 4,272 s = **1.19 h** |
| Archive | `tortoise_cm04_clips.zip`, **175.17 MB** |
| Transcripts | 500 distinct — no (speaker, transcript) pair repeats |
| Seeds | 500 distinct, taken from the committed table |

Because the job table was committed first, *what* got generated was decided in git
and auditable in review, not chosen on a GPU at 2 a.m. Clip names and seeds come
from the table rather than from a runtime counter — see §4.

## 2. The two firewalls, verified on the committed metadata

Both are mechanical, and both were re-checked against
`outputs/heldout_generation_metadata.jsonl` after the run:

| Invariant | Result |
|---|---|
| `tool == "tortoise"` on every row | ✅ 500/500 |
| `pool == "eval"` on every row | ✅ 500/500 |
| Speakers drawn only from the 15 eval-pool ids | ✅ zero overlap with the 25 train-pool speakers cloned by CM01/CM02 |
| Unique `output_path` per row | ✅ 500 distinct |
| Unique `seed` per row | ✅ 500 distinct |
| No absolute paths in the committed log | ✅ zero `/kaggle/`, zero `C:/` |

`tests/test_splits.py` fails the build if a Tortoise clip ever appears in a
training manifest. CM04 is test-only, permanently, by rule.

## 3. QA screen

`src.data.generation_qa` tests the relationship between each transcript and its
audio, because a generator can return valid audio that is completely wrong and
raise nothing.

| | |
|---|---|
| Clips screened | 500 |
| Passed | **458** |
| Failed | **42** |
| **Pass rate** | **91.6%** |
| Median speaking rate | 9.45 chars/s (passed clips) |
| Clip duration | median 7.13 s, range 1.34–22.75 s |
| Usable audio | 3,832 s = **1.06 h** |
| Clipping | **zero** clips had any clipped samples |

Failure breakdown:

| Reason | Clips |
|---|---|
| speech too slow / stalled | 24 |
| near-silent | 18 |
| speech too fast / truncated | 3 |

(Three clips carry two reasons, so the reasons sum to more than 42.)

**The failures are not spread evenly.** One speaker, `136325`, accounts for **19 of
the 42** — it drops from 34 clips to 15 usable, while every other speaker keeps at
least 30. That is a per-speaker reference-quality problem, not a Tortoise-wide one,
and it should be treated as such: if CM04 is ever rebalanced, `136325`'s reference
clip is the thing to re-cut, not the preset.

For comparison, CM01 (XTTS) screened at 99.95% and CM02 (RVC) at 93.6%. Tortoise at
91.6% is the least reliable of the three generators — which is worth stating plainly
in the paper rather than hiding, since the whole point of CM04 is that it is a
different tool with different failure modes.

## 4. Two defects found and fixed during this task

**Clip names did not match the job table.** `build_heldout_jobs` re-derived each
clip name from a 0-based `enumerate`, so job 1 became `tortoise_<spk>_00000.wav`
while the committed table said `..._00001.wav`. Zero of 500 names matched, and the
QA screen — which looks clips up by the table's basename — found no file it was
looking for. Fixed by taking names and seeds from the committed table, which is the
truth, rather than re-deriving them at runtime.

**`glob("/kaggle/input/**/mucs2021")` matched the wrong directory.** Once the
previous run's output was attached as an Input (that is what carries finished clips
into the next session), `/kaggle/input` contained *two* paths ending in `mucs2021`:
the real dataset, and a leftover `codemix-deepfake-detection/data/raw/mucs2021`
inside the notebook output. `mucs[0]` picked the leftover, which has no Kaldi
tables, and section 4 died with `CorpusError: missing Kaldi tables`. Fixed by
filtering candidates to the one that actually contains
`train/transcripts/segments`:

```python
mucs = [p for p in mucs if os.path.isfile(p + "/train/transcripts/segments")]
```

Both are the same class of bug as P-020 defect 1: a path that was inferred at
runtime instead of being pinned.

## 5. Session economics

The run does not fit in one Kaggle session and was never expected to. Tortoise is
far slower than XTTS: measured here at roughly 85–90 s/clip on a T4 with the KV
cache enabled, so 500 clips is ~12 GPU-hours against a 12-hour wall clock.

The notebook therefore carries a `GENERATE_BUDGET_HOURS` stop and a resume path:
generation halts cleanly well short of the wall, the remaining sections still run,
and section 8's zip is what carries finished clips into the next session as an
Input. In practice:

| Session | Clips | Runtime |
|---|---:|---|
| 1 | 159 | 4 h 02 m |
| 2 (resume) | 341 → **500 total** | **7 h 37 m** |

Without the budget-and-resume design, a session killed at the wall saves no output
at all and the work is simply lost.

## 6. Where everything lives

| Artefact | Location |
|---|---|
| Job table | `data/manifests/heldout_generation_jobs.csv` (committed before the run) |
| Per-clip metadata | `outputs/heldout_generation_metadata.jsonl` — 500 records, portable paths |
| QA report | `docs/qa/heldout_generation_qa.csv` (per clip) and `.md` (summary) |
| Audio | **private Kaggle dataset** — cloned voices of identifiable MUCS speakers, never in a public repo |
| Notebook | `notebooks/kaggle_w4t3_heldout_tortoise.ipynb` |

Verify the archive against the log with:

```bash
python -m src.data.rvc_archive --root <archive> --no-models \
    --metadata outputs/heldout_generation_metadata.jsonl
```

## 7. What this unblocks, and what it does not

**Unblocks.** The claim "these results are not shortcut artefacts" now has
held-out-tool evidence available to it. A detector that scores near-zero EER on
CM01/CM02 and collapses on CM04 is doing shortcut learning, and we can finally tell.

**Does not.** CM04 exists; it has **not been scored**. No checkpoint has been
evaluated against it. Generating the attack and measuring against it are two
different tasks, and only the first is done. Until a checkpoint is run over these
500 clips, the shortcut question is answerable in principle and unanswered in fact.
