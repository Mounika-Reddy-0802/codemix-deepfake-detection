# Licence & data-use register

Per-corpus licences, roles, and the usage conditions the project must honour.
This table feeds the ethics/licence note (`docs/ethics/`) that the mentor signs
off **before** any spoof generation. All links verified 21 Jul 2026.

> **Golden rule reminder:** Stage-1 trains on ASVspoof 2019 LA (English) only.
> Everything Indic/code-mixed is evaluation-only. See the team rules doc §6.

## Corpora

| # | Corpus | Licence | Role | Key usage conditions |
|---|--------|---------|------|----------------------|
| 1 | **ASVspoof 2019 LA** ([Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/3336)) | Research (ASVspoof licence) | **TRAIN (Stage-1)** + English eval | Research use; cite the ASVspoof 2019 dataset paper. LA partition only. |
| 2 | **ASVspoof 2021 DF** ([Zenodo](https://zenodo.org/records/4835108)) | ODC-ODbL | **SKIPPED** (optional) | Not downloaded (34.5 GB busts free storage). If ever added: attribution + share-alike under ODbL. |
| 3 | **IndicSynth** ([HF](https://huggingface.co/datasets/vdivyasharma/IndicSynth)) | **CC-BY-NC-4.0** | **EVAL ONLY** | **Non-commercial** research use; **attribution required**. Fine for an academic paper — state the NC restriction explicitly. Stream Hindi+Tamil only. |
| 4 | **IndicTTS-Deepfake** ([HF](https://huggingface.co/datasets/SherryT997/IndicTTS-Deepfake-Challenge-Data)) | Research | **EVAL ONLY** | Research use; cite the challenge. Stream Hindi/Tamil/English subsets. |
| 5 | **MUCS 2021 Hindi-English** ([OpenSLR 104](https://www.openslr.org/104/)) | **CC BY-SA 4.0** | **TRAIN (Stage-3 slice) + EVAL + spoof refs** | **Attribution + ShareAlike**: derivatives (e.g. a released code-mixed spoof set built from MUCS refs) inherit BY-SA. Bengali-English skipped. |
| 6 | **HiACC** ([Zenodo 15551669](https://zenodo.org/records/15551669)) | CC-BY-4.0 | **EVAL ONLY** (adult subset) | Attribution required. **HARD RULE: the 1,858 child utterances are excluded from the pipeline entirely** — never bonafide, never a cloning reference. State this in the ethics section. |
| 7 | **IndicVoices** ([HF](https://huggingface.co/datasets/ai4bharat/indicvoices)) | Research (AI4Bharat) | **EVAL ONLY** | Gated — accept terms on HF, use a token. Research use; cite AI4Bharat. Stream filtered Tamil + code-switched subsets only. |
| 8 | **Project XTTS-v2 clones** (generated Wks 2–3) | Own data (derivative) | **TRAIN (Stage-3) + EVAL** | Built from MUCS/HiACC-adult **adult** speaker refs. Not released standalone; if released, respect source BY-SA (MUCS). Speaker-disjoint train/eval. |
| 9 | **Project Tortoise-TTS clones** (generated Wk 3) | Own data (derivative) | **EVAL ONLY** (unseen attack) | Tool firewalled from training — **zero** Tortoise audio in any training manifest, ever. |
| 10 | **IITG-HingCoS** ([arXiv:1810.00662](https://arxiv.org/abs/1810.00662)) | Author permission | **BONUS** (eval-only if granted) | On-request; use only under the authors' agreement. Not on the critical path. See `docs/outreach/iitg_hingcos_request.md`. |

## Tools / models (for completeness)

| Tool | Licence | Use |
|------|---------|-----|
| wav2vec2-base / wav2vec2-large-xlsr-53 (Meta) | Apache-2.0 (MIT for XLSR weights on HF) | Encoder — research/commercial OK |
| WavLM-base (Microsoft) | MIT | Encoder — research/commercial OK |
| Coqui XTTS-v2 | Coqui Public Model Licence (CPML) — **non-commercial** | Spoof generation only; academic use. Note the NC restriction. |
| Tortoise-TTS | Apache-2.0 | Held-out unseen-attack generation |

## Obligations checklist (carry into the paper's ethics/licence statement)

- [ ] Cite every corpus and model as its licence requires.
- [ ] State **IndicSynth is CC-BY-NC** and **XTTS-v2 is CPML (non-commercial)** — the
      work is non-commercial academic research, which both permit.
- [ ] State the **HiACC child-data exclusion** explicitly.
- [ ] Note **MUCS BY-SA share-alike** implications for any released derivative spoof set.
- [ ] Confirm no eval-only corpus and no Tortoise audio appears in any training
      manifest (enforced by `tests/test_splits.py`).
- [ ] Mentor sign-off on this note **before** any spoof generation (Week 2+).
