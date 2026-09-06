# W9 — Lahari: the dataset datasheet, measured rather than planned

Branch: `week9/lahari-dataset-datasheet` · Task: **W5-T2**

**Outcome:** `docs/datasheet.md` exists — the AffectDF-style corpus description the
paper needs as supplementary material. The rule I held to while writing it: **every
number comes from a committed manifest, and anything not generated says so instead
of carrying the plan's target.**

---

## Why that rule mattered

`docs/attack_taxonomy.md` was written in Week 3 with a banner saying *"Every sample
count is a TARGET, not a measurement."* Nine weeks later those targets are still
sitting there, and it would have been easy to copy them into a datasheet and let a
reviewer read them as counts. So the datasheet is built the other way round: I
recomputed each figure from `data/manifests/` and the generation metadata, and where
an attack has not been generated the row reads `0 — not generated`.

That single decision is what surfaced how much of the taxonomy is still aspirational:
**four of the nine attack IDs do not exist yet**, including the only genuinely unseen
one.

## What the corpus actually is

| Source | Clips | Speakers | Hours | Style |
|---|---:|---:|---:|---|
| MUCS 2021 Hindi-English | 52,825 | 520 | 89.6 | read / lecture |
| HiACC (adult only) | 3,318 | 24 | 3.2 | spontaneous |
| **Total** | **56,143** | **544** | **92.8** | |

| Attack | Model | Clips | Usable | Split |
|---|---|---:|---:|---|
| CM01 | XTTS-v2 | 4,000 | **3,998** (99.95%) | train |
| CM02 | RVC v2 | 1,500 | **1,404** (93.6%) | train |
| CM03 | XTTS-v2 | 1,600 | 1,600 * | adaptation |
| **CM04–CM09** | Tortoise / kNN-VC / external | **0** | — | test |

\* CM03's usable figure is what reached the adaptation manifests, footnoted as such —
it was never screened the way CM01 and CM02 were.

**spoof:real per manifest** turned out to be worth stating: the code-mixed sets run
about 1.5 : 1 while ASVspoof runs 8.8 : 1, so the same detector confidence produces
different operating points on the two, and EERs are not directly comparable across
them.

## The exclusions section is the one I care about most

HiACC ships 5,176 files, **1,858 of them children**. They are quarantined and never
used — not as bonafide, not as a cloning reference.

I wrote that up including the part that reflects badly on us: on 12 August the
automatic quarantine **silently failed**, `Corpus/children/` was live in the corpus,
and it was caught by hand-verification against the real folder layout rather than by
the code. After the fix `audio_files()` reaches exactly 3,318 and no `CH*` id appears
in any manifest.

A datasheet that only records the controls that worked is a marketing document. The
lesson — a quarantine nobody audits is a quarantine that is not running — belongs in
the same file as the count.

## Two corrections found while writing it

- **`--report` is not a real flag** on the quarantine CLI. The reproduce block I
  first drafted would have failed for anyone who ran it; corrected to `--root`. I ran
  every command in section 8 before committing, which is how it surfaced.
- **CM03's "usable" count was an inference**, not a measurement — it is the number
  that reached the manifests. Footnoted rather than quietly presented alongside
  CM01's and CM02's screened figures.

## Numbers

| | |
|---|---|
| Bonafide clips documented | **56,143** over 544 speakers, 92.8 h |
| Attacks with real counts | **3** (CM01, CM02, CM03) |
| Attacks recorded as not generated | **6** |
| Limitations stated | **6** |
| Reproduce commands, all run before commit | **5** |

## Honest limitations

- **The datasheet is a snapshot**, dated in its header. CM04 landing changes three
  tables and nothing automatically tells the file that.
- **No gender balance is reported** for the cloned voices — MUCS has no gender
  metadata for the train pool, so the datasheet says that rather than estimating.
- **HiACC is small** (3,318 clips, 24 speakers, 3.2 h) and is the only spontaneous
  source, so the read-vs-spontaneous contrast the plan wants rests on a narrow base.
- **I did not re-verify the corpus counts against the raw audio this week** — they
  come from `clip_index.csv`, which was itself verified in Week 3 on both machines.

## What's next

1. **W8-T3, the read-vs-spontaneous style contrast** — the HiACC adult subset is
   indexed and the manifests exist; this is analysis, not new data.
2. **W6-T3 eval-set QA report** (`docs/qa_report.md`) — speaker overlap, uniform
   channel simulation, label balance per column. Much of the checking already exists
   inside `test_splits.py`; it needs to become a document a reviewer can read.
3. **Update the datasheet when CM04 lands** — three tables and the limitations list.
