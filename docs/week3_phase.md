# Week 3 — corpora on disk, pools frozen, generation proven

**Goal:** turn the Week-2 scaffolding into something that has actually run on real
data, so Week 4 can generate the corpus and start baseline training.

**Outcome:** both corpora indexed on two machines to identical counts, the child
exclusion evidenced rather than asserted, the telephony channel verified by
measurement *and* by ear, the speaker pools frozen, and 40 spoof clips generated on
a real GPU with zero failures. Stage-1 training did **not** start — ASVspoof 2019 LA
is still missing.

## What shipped

| Owner | Area | Deliverable |
|---|---|---|
| **L** | corpora + ethics | Corpus-aware indexing (MUCS Kaldi tables, HiACC prefix ids) → 56,143 clips; child quarantine fixed, verified, and now protected from being overwritten; channel-sim listening test |
| **L** | channel | G.711 + AMR-NB verified objectively (20/20) and by ear (60/60 rows) |
| **M** | positioning | AffectDF related-work anchor + precise gap statement; attack taxonomy in Table-1 format; env verification |
| **M** | environment | GPU-laptop runbook, XTTS weight pre-fetch, dependency pins made installable |
| **SK** | generation | Ethics gate; three-pool carve + SHA-256 freeze; XTTS pilot generated (40 clips); Devanagari→Latin transliteration; blind A/B rating sheet |

## Numbers

| | |
|---|---|
| Clips indexed | **56,143** (MUCS 52,825 / 520 spk, HiACC 3,318 / 24 spk) |
| Child files quarantined | **1,858** — 0 reachable from any manifest |
| Channel, G.711 @ 20 dB | 17.62 dB SNR, correlation 0.991, 20/20 |
| Channel, AMR-NB | 5.57 dB SNR, correlation 0.885, 20/20 |
| Listening test | telephony 4.0/5, intelligibility 4.0/5, 60/60 rows |
| Speaker pools | 25 train / 10 adaptation / 15 eval, SHA-256 `f57e0d85…` |
| Spoof clips generated | **40**, 0 failures, RTX 3050 6 GB |
| Test suite | **410 passed**, 2 skipped, ruff clean |

## Verify

```bash
ruff check . && ruff format --check . && pytest -q     # 410 passed
python -m src.data.corpora --data-root $DATA_ROOT      # 52,825 / 520 and 3,318 / 24
python -m src.data.quarantine --root $DATA_ROOT/raw/hiacc
python -m src.data.ethics_gate                         # exits 0
```

Full command list with the expected output of each: [`../REPRODUCE.md`](../REPRODUCE.md).

## Model architecture (unchanged from Week 3's scaffolding)

```
waveform (16 kHz) → SSLEncoder (wav2vec2/WavLM/XLSR, CNN frozen)
                  → AttentiveStatsPooling (mean ⊕ std)
                  → MLPHead (2-layer) → logits {bonafide, spoof}
```

Still validated by lint, compile and shape tests only — it has not seen a training
batch, because Stage-1 has no corpus yet.

## Honest limitations

- **No detection numbers.** No EER, no gap matrix. Stage-1 cannot train:
  **ASVspoof 2019 LA is absent** from the GPU machine and is its only training corpus.
- **The 40-clip A/B is unrated.** It blocks Week 4. The objective pre-screen (13.9 vs
  14.4 chars/sec) says romanisation does not rush or truncate speech, but the words
  themselves need ears.
- **AMR-NB is ear-unchecked.** Verified by measurement only; that column must not be
  reported as listened-to.
- **Generation can fail silently** — 1 clip of 40 produced 0.83 s from a 150-char
  transcript with no exception. Needs the W4-T6 duration filter before scale.
- **RVC never ran** (P-017): `fairseq` will not build on Python 3.12, so the second
  attack family is unavailable until that is resolved.
- **The speaker shortlist rests on a proxy** — a dynamic-range measure, not
  calibrated SNR, and the eligibility gate passed all 520 speakers. The ordering is
  a hint, not a guarantee; confirm the top of the list by ear before Week-4 scale.

## Decisions taken this week

| | |
|---|---|
| **P-013** | `channel_qa` aligns on codec delay before measuring — AMR-NB's ~4.8 ms shift failed 15/20 pairs on a working chain |
| **P-014** | The corpus is written in one script, and it is Latin — Hinglish is produced, not selected |
| **P-015** | The XTTS character limit applies to the **romanised** text, not the source |
| **P-016** | Selection ties break on `utt_id`, so two machines build the same pack |
| **P-017** | RVC is toolchain-blocked and deferred to W4-T2 |

Full log with the problem behind each: [`problems_and_decisions.md`](problems_and_decisions.md).

## Still open at week's end

Everything remaining is human, not engineering: rate the 40 clips, the AMR-NB
listening pass, three PR reviews, Week-3 log-book entries, and the ASVspoof
download. Tracked in [`../Works updates.md`](../Works%20updates.md).
