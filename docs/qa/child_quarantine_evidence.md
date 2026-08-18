# HiACC child quarantine — re-verification on the GPU laptop

**This does not re-open a closed gate.** The adult/child split was read from the
HiACC documentation and the manual verification was completed and recorded on
12 August 2026 — see the ticked checklist and the incident narrative in
`child_quarantine_check.md`, which remain the primary record.

This file exists because the corpus was extracted a second time, on a second
machine, and an ethics exclusion is worth re-checking on every machine that holds
the data rather than assumed to have travelled with it.

Machine: GPU laptop, `DATA_ROOT=C:\dfdata`. Extracted 17 Aug 2026 by
`scripts/01_download_data.sh --extract-only`.

## What was re-checked here, and what it showed

| Check | 12 Aug (dev laptop) | 17 Aug (GPU laptop) |
|---|---|---|
| Adult clips | 3,318 | **3,318** |
| Child wavs quarantined | 1,858 | **1,858** |
| Total audio | 5,176 | **5,176** |
| Adult speakers | 24 | **24** |
| Child-looking items outside quarantine | 0 | **0** |
| `clip_index.csv` rows inside the quarantine | — | **0** |

The extraction script's post-condition check (which is what the 12 Aug incident
added) passed: it re-scans after the sweep and exits non-zero if anything
child-looking survives outside quarantine. It exited 0.

## Manifest cross-check against the real child id set

The 12 Aug check confirmed no `CH*` id reaches a manifest. Repeated here against
the manifests as they now stand, including the two new pilot job tables:

| manifest | rows | child ids present |
|---|---|---|
| `clip_index.csv` | 56,143 | none |
| `speaker_pools.csv` | 50 | none |
| `speaker_ranking.csv` | 520 | none |
| `speaker_shortlist.csv` | 50 | none |
| `pilot_generation_jobs.csv` | 20 | none |
| `pilot_generation_jobs_roman.csv` | 20 | none |

Read from `Children/metadata/speaker_info.csv`, column **`PID`**: 20 child
speakers, ages 10–14. Adults are `AD*`, ages 19–42; the id sets are disjoint.

> Method note, because an earlier pass of this check proved nothing. The id
> column is `PID`; a lookup guessing `speaker`/`id`/`spk` returned an empty set,
> and every intersection against an empty set is empty — so the check "passed"
> without testing anything. "No overlap" is evidence only when the set being
> intersected is non-empty. It is 20 ids here.

## New finding: an undocumented speaker

Not visible on 12 Aug because it surfaces from indexing, which is new:

- `AD63` — has audio, but **no row** in `Corpus/adult/metadata/speaker_info.csv`.
- `AD65` — has a metadata row, but **no audio**.

Both carry the `AD` prefix and every child speaker in the corpus is `CH`-prefixed
with a recorded age of 10–14, so this does not put child audio into the pipeline.
It looks like a mislabelled folder or an off-by-one in the corpus's own metadata.

**Action:** `AD63`'s age is unverified, so it must not be used as a cloning
reference until the HiACC authors confirm it. It is not in the frozen train pool,
so nothing generated so far depends on it. Worth an email alongside the
IITG-HingCoS one.

## Process fix

Running `python -m src.data.quarantine` on this machine **overwrote**
`child_quarantine_check.md` and destroyed both the incident narrative and the
completed checklist — the run prints `wrote ...` either way, so nothing signalled
it. The file is generated, but people write into it, and that content is exactly
the part a tool cannot reproduce.

The audit now refuses to overwrite a report carrying hand-written sections and
names what would be lost, unless `--force` is passed. Guarded by
`test_a_fresh_report_over_a_signed_one_is_refused`.

## Sign-off

The 12 Aug sign-off stands and is not superseded. This page needs a signature
only as a record that the exclusion was re-verified on the second machine:

Re-checked by: ______________________  Date: ____________
