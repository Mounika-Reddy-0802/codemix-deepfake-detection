# Attack taxonomy (AffectDF Table-1 format)

**W3-T3, owner M.** Our analogue of AffectDF's Table 1 (Mahapatra et al.,
arXiv:2608.05507). Their table is the reason their splits are auditable at a
glance: every attack has an ID, a generation type, a named model, its base corpus,
a sample count, a speaker count, and the split it belongs to — with **disjoint
speakers AND disjoint attack systems** across splits. We adopt the same shape so
our dataset can be read the same way.

> **Attack IDs below are OURS, not AffectDF's.** Their A01–A21 are a different
> dataset. Ours are prefixed `CM` (code-mixed) to keep the two apart in any table
> that cites both.

> **Every sample count is a TARGET, not a measurement.** No spoof has been
> generated yet — generation is blocked behind the ethics sign-off (W3-T1) and
> begins in Week 4. This file is re-committed with real counts at the Week-5
> dataset freeze.

---

## 1. Bonafide sources

| Source | Language | Role | Split(s) | Notes |
|---|---|---|---|---|
| ASVspoof 2019 LA | en | S1 anchor training + English eval column | Train / Dev / Eval | The **only** S1 training corpus |
| MUCS 2021 subtask 2 | hi-en | Code-mixed bonafide; cloning references | Train / Adaptation / Eval | OpenSLR 104, Hindi-English only (Bengali-English skipped) |
| HiACC (adult subset) | hi-en | Code-mixed bonafide, spontaneous | Eval | Child folders quarantined, never used — see `docs/qa/child_quarantine_check.md` |
| IndicVoices | hi, ta | Mono-Hindi / mono-Tamil bonafide | Eval only | Gated on HF; streaming only |

## 2. Attacks

| Attack ID | Generation Type | Model | Base Data | Samples (target) | No. Spks | Split | Seen by |
|---|---|---|---|---|---|---|---|
| **TRAIN POOL** ||||||||
| CM01 | TTS (zero-shot clone) | XTTS-v2 | MUCS train pool | ~4,000 | 20–25 | Train | S3 |
| CM02 | VC | RVC (per-speaker models) | MUCS train pool | ~1,500 | 10–15 | Train | S3 |
| **ADAPTATION POOL** ||||||||
| CM03 | TTS (zero-shot clone) | XTTS-v2 | MUCS adaptation pool | ~800 | 8–10 | Adaptation | S2 (LoRA) |
| **TEST — HELD-OUT TOOLS** ||||||||
| CM04 | TTS | Tortoise-TTS | MUCS/HiACC eval pool | 400–600 | eval pool | Test | **nothing** |
| CM05 | VC | kNN-VC *(optional)* | eval pool | ~300 | eval pool | Test | **nothing** |
| **TEST — EXTERNAL, EVAL-ONLY** ||||||||
| CM06 | TTS | IndicTTS-Deepfake | mono-Hindi | as released | — | Test | nothing |
| CM07 | TTS | IndicSynth | mono-Hindi | as released | — | Test | nothing |
| CM08 | TTS | IndicSynth | mono-Tamil | as released | — | Test | nothing |
| CM09 | mixed (21 attacks) | AffectDF test subset | ESD / MSP-Podcast | sampled | — | Test (external) | nothing |

## 3. The disjointness rules this table encodes

1. **Speaker pools are carved before any generation.** Train / adaptation / eval
   pools are disjoint sets of MUCS speakers, frozen in
   `data/manifests/speaker_pools.csv`. A cloned voice of speaker X carries
   speaker id X, so disjointness holds across bonafide *and* spoof.
2. **Tool firewall.** CM04–CM09 never appear in any training manifest, in any
   stage. Tortoise is the headline unseen attack: if S3 scores near 0% EER on
   CM01/CM02 and collapses on CM04, that is shortcut learning, not detection.
3. **Split disjointness of attack systems** (AffectDF's move): no generation model
   appears in both a training split and a test split. CM01/CM03 share XTTS-v2 but
   sit in train and adaptation respectively, on disjoint speakers — S2's whole
   point is cheap adaptation to a *seen* tool on *unseen* speakers.
4. **No child audio anywhere** — not as bonafide, not as a cloning reference.
5. **Eval-only corpora** (IndicSynth, IndicTTS-Deepfake, IndicVoices, AffectDF)
   never enter a training manifest.

Enforced mechanically by `tests/test_splits.py`.

## 4. Gap-matrix columns this taxonomy feeds

| Column | Bonafide | Spoof | What it measures |
|---|---|---|---|
| english | ASVspoof 2019 LA | ASVspoof 2019 LA | in-domain anchor (S1 was trained here) |
| hindi | IndicVoices hi | CM06, CM07 | mono-lingual shift, no code-switching |
| tamil | IndicVoices ta | CM08 | mono-lingual shift, unrelated language family |
| codemixed | MUCS + HiACC adult | CM01–CM05 | **intra-utterance code-switching — the headline** |
| affectdf (external) | AffectDF bonafide | CM09 | emotional shift; cross-validates against the base paper |

Every column is evaluated **twice** — `clean` and `channel_matched` (8 kHz,
G.711/AMR-NB, SNR noise) — with the clean copy retained, so the channel effect
stays separable from the language effect.

## 5. Open decisions

- **kNN-VC (CM05) is optional.** Include only if Week-4 generation hours allow;
  Tortoise alone satisfies the held-out-tool requirement.
- **AffectDF subset size (CM09)** depends on what is practical to download under
  CC BY-NC 4.0. Fallback in the plan: a pretrained-AASIST comparison row carries
  the external-validity claim alone.
- **Sample counts** are targets pending the Week-4 generation run.
