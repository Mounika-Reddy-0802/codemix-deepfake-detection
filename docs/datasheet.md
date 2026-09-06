# Dataset datasheet — code-mixed audio deepfake detection corpus

**W5-T2, owner L.** Written in the shape AffectDF (Mahapatra et al.,
arXiv:2608.05507) uses for its Table 1 and its data appendix, because that is the
format a reviewer of this literature already knows how to read: every attack has an
ID, a generation type, a named model, a base corpus, a real sample count, a speaker
count, and the split it belongs to.

**Every number here is measured, not planned.** Counts come from the committed
manifests and generation metadata; the commands that reproduce them are at the end.
Where a row is not yet generated it says so instead of carrying a target figure —
`docs/attack_taxonomy.md` holds the targets, this file holds what exists.

Last measured: 2026-09-06.

---

## 1. Motivation and composition

The corpus supports one question: **how much worse is an English-trained deepfake
detector on Hinglish code-mixed speech carried over a telephone channel, and can
that gap be closed cheaply?**

It has three parts:

1. **An English anchor** — ASVspoof 2019 LA, used unmodified. Stage-1 trains on this
   and nothing else, which is what makes the gap a measurement of generalisation
   rather than an artefact of mixed training data.
2. **Code-mixed bonafide speech** — MUCS 2021 Hindi-English (NPTEL lecture audio) and
   HiACC adult (spontaneous), indexed to real speaker identities.
3. **Code-mixed spoofs we generated** — XTTS-v2 zero-shot clones (CM01) and RVC
   per-speaker voice conversions (CM02), both built only from train-pool speakers.

### 1.1 Bonafide sources, as indexed

| Source | Clips | Speakers | Hours | Style | Role |
|---|---:|---:|---:|---|---|
| MUCS 2021 Hindi-English | 52,825 | 520 | 89.6 | read / lecture | code-mixed bonafide, cloning references |
| HiACC (adult subset only) | 3,318 | 24 | 3.2 | spontaneous | code-mixed bonafide, eval only |
| **Total (`clip_index.csv`)** | **56,143** | **544** | **92.8** | | |

ASVspoof 2019 LA is indexed separately because it is a different language and a
different role:

| Split | Clips | Speakers | Bonafide | Spoof | Spoof:real |
|---|---:|---:|---:|---:|---:|
| train | 25,380 | 20 | 2,580 | 22,800 | 8.8 : 1 |
| dev | 24,844 | 20 | 2,548 | 22,296 | 8.8 : 1 |
| eval | 71,237 | 67 | 7,355 | 63,882 | 8.7 : 1 |

Speaker sets are disjoint across the three splits; the counts match the published
protocol figures, which is checked rather than assumed
(`tests/test_splits.py`, and the assertion in the retention notebook).

### 1.2 Speaker pools — carved once, before any audio was generated

50 MUCS speakers were shortlisted on SNR and effective speech duration, then split
into three **disjoint** pools and frozen:

| Pool | Speakers | Used for |
|---|---:|---|
| train | 25 | S3 native training; CM01 + CM02 generation |
| adaptation | 10 | S2 LoRA adaptation only (CM03) |
| eval | 15 | every evaluation column; held-out attacks |

Frozen at `data/manifests/speaker_pools.csv`,
SHA-256 `f57e0d85dbd3c8f5b96ad6af59edae7218104b0167807eae6433314947063e7b`.

The freeze is the point. **A cloned voice carries the identity of the speaker it
clones**, so if a speaker moved pool after generation, a fake of an evaluation voice
would be sitting in training data and every generalisation number in the project
would be void. `speaker_pools.verify_frozen()` re-checks the hash before use, and
`test_verify_fails_when_a_speaker_moved_pool` is exactly that scenario.

---

## 2. Attacks — what has actually been generated

| ID | Type | Model | Base data | Clips | Usable | Targets | Split | Seen by |
|---|---|---|---|---:|---:|---:|---|---|
| **CM01** | TTS (zero-shot clone) | XTTS-v2 | MUCS train pool | 4,000 | **3,998** (99.95%) | 25 spk × 160 | train | S3 |
| **CM02** | VC (per-speaker models) | RVC v2 | MUCS train pool | 1,500 | **1,404** (93.6%) | 12 spk × 125 | train | S3 |
| **CM03** | TTS (zero-shot clone) | XTTS-v2 | MUCS adaptation pool | 1,600 | 1,600 \* | 10 spk | adaptation | S2 (LoRA) |
| CM04 | TTS | Tortoise-TTS | MUCS/HiACC eval pool | **0 — not generated** | — | — | test | nothing |
| CM05 | VC | kNN-VC *(optional)* | eval pool | **0 — not generated** | — | — | test | nothing |
| CM06–CM08 | TTS | IndicTTS-Deepfake / IndicSynth | mono-Hindi, mono-Tamil | **0 — not indexed yet** | — | — | test | nothing |
| CM09 | mixed | AffectDF subset | ESD / MSP-Podcast | **0 — not fetched** | — | — | test (external) | nothing |

\* CM03's usable count is the number that reached the adaptation manifests (1,280 train + 320 dev); it was not screened separately the way CM01 and CM02 were.

**CM04 is the gap that matters most.** It is the only genuinely unseen attack, and
until it exists the claim "these results are not shortcut artefacts" has no
held-out-tool evidence behind it. It is W4-T3 and it is still open.

### 2.1 CM01 — XTTS-v2, the seen TTS attack

4,000 clips over the 25 train-pool speakers, 160 each, cross-paired against 1,404
unique synthesisable transcripts so no (speaker, transcript) pair repeats — a
detector that learned "this sentence = spoof" would be learning the transcript, not
the artefact. 6.91 hours of audio, zero generator exceptions.

Two QA passes were needed. The first screen caught **116 clips (2.9%)** that raised
no error but were unusable — stalled at 3–6 chars/s, truncated at 31–92 chars/s, or
under one second. **95 of the 116 came from a single speaker (987461)** whose
reference was 25 s of audio at 89% silence, i.e. 2.65 s of actual voice. References
were re-ranked by *effective speech seconds* rather than raw duration (5 of 25 were
below threshold), that speaker was re-cut to 12.34 s, and the affected clips were
regenerated. After the re-cut: **3,998 of 4,000 usable (99.95%, up from 97.1%)**. The
two remaining rejects are borderline-slow and both speakers still keep 159 clips.

**Known property, not a defect (P-019 / P-021):** measured with `src.data.f0_stats`
(Praat autocorrelation, 60–400 Hz), XTTS-v2 clips have a median intra-utterance f0
IQR of 27.8 Hz against 21.8 Hz for the real MUCS speech they clone — the generator
*overshoots* prosodic variation. P-019 originally reported this as a 35% compression;
that comparison used two different estimators and P-021 retired it. The corrected
figure is based on 23 pilot clips and needs re-measuring across all ~4,000.

### 2.2 CM02 — RVC, the seen VC attack

12 per-speaker RVC v2 voice models (rmvpe f0, 40k, 100 epochs, seed 4200), each
converting 125 real utterances **from other train-pool speakers** into that target's
voice — 1,500 conversions drawing on 1,345 unique source clips across all 25 source
speakers. A target is never converted from its own speech; that would not be an
impersonation.

Quality screen: **96 of 1,500 (6.4%) flagged** — near-silence (7 clips, RMS ≤ 0.0004)
and stall/truncation by speaking rate. Usable: 1,404.

Pitch, same estimator as CM01: median f0 IQR **21.06 Hz** over 1,493 measurable
clips against 21.8 Hz for the real source speech — **96.4% retention**. This is the
designed contrast with CM01: voice conversion starts from real speech so the contour
is human, while XTTS invents it.

Every conversion's metadata records source clip, source speaker, target speaker and
the **SHA-256 of the exact model weights** that produced it. All 24 model checksums
(12 `.pth` + 12 `.index`) were verified against the archived copies.

### 2.3 CM03 — XTTS-v2 on the adaptation pool

The same generator as CM01 pointed at the 10 adaptation-pool speakers, so S2's LoRA
adaptation trains on a *seen tool* over *unseen speakers*. Disjointness from CM01 is
by speaker pool, which the frozen pool file guarantees.

---

## 3. Splits and spoof:real ratio, per manifest

| Manifest | Clips | Speakers | Bonafide | Spoof | Spoof:real | Attack |
|---|---:|---:|---:|---:|---:|---|
| `codemix_adapt_train` | 2,154 | 8 | 874 | 1,280 | 1.5 : 1 | CM03 |
| `codemix_adapt_dev` | 506 | 2 | 186 | 320 | 1.7 : 1 | CM03 |
| `codemix_eval` | 3,966 | 15 | 1,566 | 2,400 | 1.5 : 1 | CM01 |
| `gap_codemix_clean` | 4,434 | 25 | 2,217 | 2,217 | **1 : 1** | CM01 |

The code-mixed manifests are far closer to balanced than ASVspoof's 8.8:1, which
matters when reading EERs across the two: the same detector confidence produces
different operating points on differently balanced sets.

Each evaluation manifest exists in **clean** and **channel-matched** form
(`*_clean.csv` / `*_channel20.csv`), the same clips passed through 8 kHz G.711 μ-law
with additive noise at 20 dB SNR. The clean copies are retained, so the channel
effect stays separable from the language effect.

---

## 4. Exclusions — what is deliberately absent

**HiACC child audio, entirely.** The corpus ships 5,176 audio files, of which
**1,858 are children**. They are quarantined into `_EXCLUDED_children/` on
extraction and are never bonafide, never a cloning reference, never anything.

This is recorded honestly because it did not work the first time: on 12 August 2026
the automatic quarantine **silently failed** and `Corpus/children/` was sitting live
in the corpus. It was caught by hand-verification against the real folder layout, not
by the code. After the fix, `src.data.preprocess.audio_files()` reaches exactly
**3,318** files and no `CH*` identifier appears in any manifest. Archive md5
`dd6cc9354e1dee5e2f25bc5243df88ac`, verified against Zenodo. The lesson recorded in
`docs/qa/child_quarantine_check.md` is that a quarantine nobody audits is a
quarantine that is not running.

**Also excluded:**

- **ASVspoof 2021 DF** — skipped for storage (34.5 GB).
- **MUCS Bengali-English** — out of scope; Hindi-English only.
- **Tortoise clones from any training manifest** — permanently, by rule. Enforced by
  `tests/test_splits.py`, which fails the build rather than warning.
- **Eval-only corpora from any training manifest** — IndicSynth, IndicTTS-Deepfake,
  IndicVoices, AffectDF.

---

## 5. Licensing and redistribution

Full table with usage conditions: [`docs/licences.md`](licences.md). The four that
constrain what this project may release:

| Corpus | Licence | Consequence for us |
|---|---|---|
| MUCS 2021 | **CC BY-SA 4.0** | **Attribution + ShareAlike.** Our spoofs are derivatives of MUCS references, so a released code-mixed spoof set inherits BY-SA. |
| HiACC | CC BY 4.0 | Attribution. Adult subset only. |
| IndicSynth | **CC BY-NC 4.0** | **Non-commercial only** — the restriction must be stated explicitly in the paper. |
| ASVspoof 2019 LA | ASVspoof research licence | Research use, cite the dataset paper. LA partition only. |
| IndicVoices | AI4Bharat research | Gated: accept terms, use a token. |

**The generated audio is not in this repository and will not be.** It is cloned
speech of identifiable people; the repo is public. CM01 and CM02 audio live in
private Kaggle datasets, with per-clip metadata, job tables and SHA-256 checksums
committed here so the corpus is auditable without the audio being public.

---

## 6. Collection and generation process

1. **Index** — MUCS is read from its own Kaldi tables (`text`, `wav.scp`,
   `segments`) so speaker ids are the corpus's real ones, not invented from
   filenames; HiACC uses its documented id prefixes. 56,143 rows.
2. **Shortlist** — speakers ranked on SNR and effective speech seconds; top 50 kept.
3. **Freeze pools** — 25/10/15, hashed, committed. Nothing generates before this.
4. **Ethics gate** — no generation of any kind, including pilots, until a signed
   mentor note exists in `docs/ethics/`. Mechanical, with no override flag
   (`src/data/ethics_gate.py`); every generation entry point calls it before a model
   loads.
5. **Job tables** — every clip to be generated is a committed row (tool, speaker,
   pool, source/transcript, seed, output path) *before* the GPU is booked, so
   generation is reproducible, resumable and auditable after the fact.
6. **Generate** — CM01 on a GPU laptop, CM02 on a Kaggle T4.
7. **QA screen** — automatic filters for near-silence, stalling and truncation, with
   the failure rate reported rather than the failures quietly dropped.
8. **Channel simulation** — every eval clip through 8 kHz G.711 μ-law + noise at a
   controlled SNR; clean copies retained.

### Transcript handling

MUCS is 98.2% Devanagari-bearing, and only 17 Devanagari-free transcripts exist
inside the train pool — so Hinglish had to be **produced**, not selected. Transcripts
are transliterated to Latin by a hand-rolled transliterator (`src/data/transliteration.py`)
rather than a library, because the spelling is an *input to the generator* and a
library upgrade changing one vowel would silently regenerate half the corpus from a
different romanisation. 0 unmapped characters across all 56,143 rows.

---

## 7. Known limitations

1. **The corpus fails its own shortcut gate.** Eight cheap signal statistics separate
   CM01 from MUCS bonafide at **1.39% EER** where chance is 50%. Two causes: a real
   bug (`portable_bundle.build` never level-normalises, though `preprocess.py` and
   `channel_sim.yaml` both specify −23 dBFS) and a genuine recording-domain gap
   (lecture audio vs vocoder). CM02, asked the stricter question of conversions
   against *their own source recordings*, reaches **22.3%** — 16× better, and its
   residual is spectral rather than level-based. See
   [`lowlevel_cue_check_v1.md`](results/lowlevel_cue_check_v1.md) and
   [`lowlevel_cue_check_cm02_v1.md`](results/lowlevel_cue_check_cm02_v1.md).
   **Consequence: the clean 1.34% gap-closure figure is not quotable.**
2. **Language and recording domain are confounded in the gap matrix.** ASVspoof is
   studio read speech; MUCS is NPTEL lecture audio. The 51× gap measures *English
   studio → Hinglish lecture*, not code-mixing alone. The channel-matched columns and
   a monolingual-Hindi column are the two ways to separate them; the mono-lingual
   columns are not built yet.
3. **No held-out attack tool exists yet (CM04).** Every generated clip so far comes
   from a tool the models are allowed to see.
4. **Spoof quality is uneven.** CM01 rated 1.5/5 "sounds human" and 1/5
   "code-switch natural" by the team. These are easy fakes, and detector numbers
   against them are optimistic.
5. **HiACC is small** — 3,318 adult clips, 24 speakers, 3.2 hours. It is the only
   spontaneous-speech source, so the read/spontaneous contrast rests on a narrow base.
6. **Gender metadata is absent** for the MUCS train pool, so no gender balance can be
   reported for the cloned voices.

---

## 8. Reproduce every number in this file

```bash
export DATA_ROOT=/c/dfdata

# 1.1 bonafide composition
python -c "import pandas as pd; d=pd.read_csv('data/manifests/clip_index.csv'); \
print(d.groupby('source').agg(clips=('utt_id','size'), speakers=('speaker','nunique'), \
hours=('duration_seconds', lambda s: round(s.sum()/3600,1))))"

# 1.1 ASVspoof splits / 3. spoof:real per manifest
python -c "import pandas as pd; [print(n, len(d), d.speaker.nunique(), \
(d.label=='bonafide').sum(), (d.label=='spoof').sum()) for n,d in \
[(n, pd.read_csv(f'data/manifests/{n}.csv')) for n in \
['asvspoof_train','asvspoof_dev','asvspoof_eval','codemix_adapt_train', \
'codemix_adapt_dev','codemix_eval','gap_codemix_clean']]]"

# 1.2 pools + freeze hash
python -m src.data.speaker_pools --verify f57e0d85dbd3c8f5b96ad6af59edae7218104b0167807eae6433314947063e7b

# 2.2 CM02 counts, QA and pitch
python -c "import json; d=json.load(open('outputs/rvc_generation_summary.json')); \
print(d['stats']); print(d['qa']['clips'], d['qa']['failed']); print(d['pitch']['converted'])"

# 4. quarantine
python -m src.data.quarantine --root "$DATA_ROOT/raw/hiacc"
```
