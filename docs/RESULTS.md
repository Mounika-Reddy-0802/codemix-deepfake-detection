# Results — what is actually measured

One page for anyone asking "what does this project *know* so far".

**There are no detection results yet.** No EER, no gap matrix, no LoRA number.
Stage-1 has not trained, because ASVspoof 2019 LA is not on the GPU machine. What
exists is a validated pipeline and four measured milestones, each reproducible from
[`REPRODUCE.md`](../REPRODUCE.md).

Read this alongside [`problems_and_decisions.md`](problems_and_decisions.md), which
records *why* each of these came out the way it did.

---

## M1 — Both corpora index identically on two machines

| | dev laptop | GPU laptop |
|---|---|---|
| MUCS utterances / speakers | 52,825 / 520 | **52,825 / 520** |
| HiACC adult clips / speakers | 3,318 / 24 | **3,318 / 24** |
| HiACC child files quarantined | 1,858 | **1,858** |
| Total indexed | 56,143 | **56,143** |

Why it matters: the two machines are the eval box and the generation box. If they
indexed the corpus differently, nothing measured on one would transfer to the other.

The number that took work is 52,825. MUCS is a **Kaldi corpus** — 521 long
recordings with speakers in `utt2spk` and utterance spans in `segments` — so the
original blind chunking destroyed speaker identity entirely. Chunking was replaced
by corpus-aware indexing against the real Kaldi tables (**P-009** territory; see the
Week-3 notes).

**Status:** ✅ verified on both machines.

---

## M2 — Child audio is excluded, and that is now evidenced rather than asserted

| Check | Result |
|---|---|
| Child wavs in quarantine | **1,858** |
| Adult clips reachable | **3,318** |
| Child-looking items outside quarantine | **0** |
| `CH*` speaker ids in any manifest | **0** |
| `clip_index.csv` rows pointing into quarantine | **0** |

HiACC's `readme.txt` states the split is by top-level folder (`Adult/`, `Children/`),
which is exactly what the quarantine sweep acts on. Corroborated by metadata: 24
adult speakers (`AD*`, ages 19–42) and 20 child speakers (`CH*`, ages 10–14), with
disjoint id sets and non-overlapping ages.

Two things this milestone is honest about:

- **A sweep that reports success is not evidence.** On 12 Aug the quarantine
  reported "moved 0 folder(s)" on a corpus whose `Corpus/children/` directory was
  sitting right there — `grep -P` fails on Git Bash and `(?i)` matches nothing under
  POSIX `-E`. Two silent bugs, and either alone is indistinguishable from a clean
  corpus. The fix that matters is not the regex; it is the **post-condition check**
  that now aborts if anything child-looking survives.
- **An empty set proves nothing.** A later re-check appeared to pass while testing
  nothing: the id column is `PID`, the lookup guessed `speaker`/`id`, and every
  intersection against the resulting empty set was empty. "No overlap" is only
  evidence when the set being intersected is non-empty — it is 20 ids.

**Open:** `AD63` has audio but no metadata row (and `AD65` the reverse). Both are
adult-prefixed, so no child audio enters the pipeline, but `AD63`'s age is
unverified — it must not be used as a cloning reference until HiACC confirms it.

**Status:** ✅ signed 12 Aug, re-verified on the GPU laptop 17 Aug.
→ [`qa/child_quarantine_evidence.md`](qa/child_quarantine_evidence.md)

---

## M3 — The telephony channel is real, by measurement *and* by ear

| Condition | SNR | Correlation | Objective pass |
|---|---|---|---|
| G.711 μ-law @ 20 dB | 17.62 dB | 0.991 | 20/20 |
| AMR-NB | 5.57 dB | 0.885 | 20/20 |

High-frequency energy drops from 0.6 % (clean) to 0.0001 % (channel) — the audio is
genuinely band-limited, not merely quieter. AMR-NB is materially harsher than
G.711, which is why the eval matrix carries both rather than assuming one stands in
for the other.

**Ear check:** L, M and SK each rated all 20 clean/channel pairs (60/60 rows) —
telephony **4.0/5**, intelligibility **4.0/5**, in both languages.

The measurement nearly failed for the wrong reason: AMR-NB delays its output by
~4.8 ms (76–80 samples at 16 kHz) and G.711 does not. Measured at offset 0 that
shift read as correlation ≈ 0 and negative SNR, failing 15 of 20 pairs on a chain
that was working perfectly (**P-013**). Alignment is now bounded to ±100 ms so it
can never rescue genuinely unrelated audio.

**Open:** the AMR-NB **listening** pass. That column is verified by measurement
only and must not be reported as ear-checked.

**Status:** ✅ G.711 complete. ⬜ AMR-NB ears outstanding.
→ [`qa/channel_sim_verdict.md`](qa/channel_sim_verdict.md)

---

## M4 — Spoof generation works end to end, and the script question has an answer

**40 clips, 0 failures** on an RTX 3050 (6 GB), CUDA torch 2.8.0. A matched A/B —
same speaker, same sentence, same language tag, differing only in script.

| | Devanagari | Romanised |
|---|---|---|
| Clips | 20 | 20 |
| Failures | 0 | 0 |
| Median speech rate | 13.9 chars/sec | **14.4 chars/sec** |
| Median clip duration | 9.42 s | 10.44 s |
| Paired duration ratio (median) | — | 1.09 |

Speech rate is indistinguishable between the two and durations track the ~13 % text
expansion, so **romanisation does not rush or truncate speech**. This is a
pre-screen that says the A/B is worth listening to — it does not say which side wins.

### Why the corpus is romanised at all

| | Count |
|---|---|
| MUCS transcripts | 52,825 |
| …containing no Devanagari | 957 (1.8 %) |
| …usable, inside the frozen train pool | **17** |

A Latin-only corpus could not be *selected*, only *produced*. The transliterator is
hand-rolled (no dependency, so it runs in CI and its spellings stay frozen) and
verified across all 56,143 transcripts: **0 unmapped characters, 0 rows leaking
Devanagari** (**P-014**).

### Two traps this milestone walked into

- **Romanisation inflates length ~13 %, up to 58 %.** Fourteen of the 20 pilot
  transcripts crossed XTTS's 150-character Hindi limit once transliterated. XTTS
  truncates *silently*. Filtering on the Devanagari length would have handed the
  model text it quietly cut off — the P-010 failure re-entering by the back door
  (**P-015**).
- **The pack was not reproducible.** MUCS segment durations are whole seconds, so
  ties are the rule; with duration as the only sort key the winner fell out of
  filesystem walk order. The two laptops built the *same* pilot with speakers 4 and
  5 swapped in every cell, so `deva_hi_04` was a different person on each and a
  position-keyed rating sheet would have mis-attributed every score (**P-016**).

### The defect worth carrying into Week 4

One clip of 40 produced **0.83 s of audio from a 150-character transcript** — about
180 chars/sec — and raised no exception. The batch runner catches crashes, not
clips that come out empty. At Week-4 scale that is roughly **200 dead files** nobody
would notice. The duration / chars-per-second auto-filter (W4-T6) has to land
*before* the 4,000-clip run.

**Status:** ✅ generated. ⬜ **unrated — this blocks Week 4.**
→ [`qa/pilot_script_rating_sheet.csv`](qa/pilot_script_rating_sheet.csv)

---

## M5 — RVC: viability answered, and the answer is "not here"

W3-T5 asks for four decisions — script, language tag, quality bar, RVC viability.
The fourth: **blocked on toolchain, not on GPU or ethics.**

`rvc-python` depends on `fairseq`, whose build needs an old numpy calling
`pkgutil.ImpImporter` — removed in Python 3.12. The install dies at
`Failed to build 'numpy'` before fetching anything. This machine has only Python
3.12.7 and no conda.

Three ways forward, none free: a side-by-side Python 3.10 with its own venv; a
fairseq-free RVC fork (re-validates the whole conversion path); or dropping RVC and
letting Tortoise be the only second attack family — which narrows the scope the
mentor signed off on, so it is a team decision rather than a quiet omission.

**Do not attempt the install in the project venv** — the resolver downgrades torch
and would break the XTTS stack that currently generates (**P-017**).

**Status:** ⬜ blocked, deferred to W4-T2.

---

## What has to happen before there are detection numbers

| Blocker | Needs |
|---|---|
| ASVspoof 2019 LA absent from the GPU machine | A download — it is Stage-1's *only* training corpus |
| The 40-clip A/B unrated | Three people, ~20 minutes each |
| AMR-NB listening pass | Same session |
| Duration auto-filter | W4-T6, before the 4,000-clip run |

The Week-5 baseline checkpoint is the first point at which a cross-lingual gap
number from this repo means anything.
