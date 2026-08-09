# Related-work notes (becomes the paper's Related Work section)

17 references from the proposal, grouped into **four survey strands**. Each member
deep-reads ~6 papers and writes a **3–4 sentence summary** in the block under each
reference. Keep summaries factual: (1) what the paper does, (2) its key
method/result, (3) how it relates to *our* code-mixing generalisation-gap study.

> **Verify each citation** (exact venue, year, DOI/arXiv id) before it goes into
> the paper. The ids below are starting points, not final bibliography entries.

## Reading assignments (~6 papers each)

| Member | Strand focus | Papers |
|--------|--------------|--------|
| **M** (Model Lead) | Detection systems + SSL representations | 1, 2, 3, 6, 7, 8 |
| **L** (Data Lead) | Voice cloning/TTS (attacks) + Indic/code-switch corpora | 10, 11, 12, 13, 14, 15 |
| **SK** (Systems Lead) | Detection architectures + generalisation/robustness | 4, 5, 9, 16, 17 |

Each reference names the member who owns its summary. Cross-review at the end:
every member reads the other two strands.

---

## Strand 1 — Spoofing detection & the ASVspoof benchmarks

**[1] Wu et al., "ASVspoof 2015: the first automatic speaker verification spoofing
and countermeasures challenge," Interspeech 2015.**

> _Summary (3–4 sentences) — M:_
> TODO.

**[2] Todisco et al., "ASVspoof 2019: Future Horizons in Spoofed and Fake Audio
Detection," Interspeech 2019 (arXiv:1904.05441).**

> _Summary (3–4 sentences) — M:_
> TODO. (This is our Stage-1 training corpus — connect it to the golden rule.)

**[3] Yamagishi et al., "ASVspoof 2021: accelerating progress in spoofed and deepfake
speech detection," ASVspoof Workshop 2021 (arXiv:2109.00537).**

> _Summary (3–4 sentences) — M:_
> TODO. (Note the DF track we deliberately skip for storage, and why.)

**[4] Tak et al., "End-to-end anti-spoofing with RawNet2," ICASSP 2021
(arXiv:2011.01108).**

> _Summary (3–4 sentences) — SK:_
> TODO.

**[5] Jung et al., "AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal
Graph Attention Networks," ICASSP 2022 (arXiv:2110.01200).**

> _Summary (3–4 sentences) — SK:_
> TODO.

---

## Strand 2 — Self-supervised speech representations for anti-spoofing

**[6] Baevski et al., "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech
Representations," NeurIPS 2020 (arXiv:2006.11477).**

> _Summary (3–4 sentences) — M:_
> TODO. (Our baseline encoder.)

**[7] Chen et al., "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack
Speech Processing," IEEE JSTSP 2022 (arXiv:2110.13900).**

> _Summary (3–4 sentences) — M:_
> TODO. (Our secondary encoder for the ablation.)

**[8] Conneau et al., "Unsupervised Cross-lingual Representation Learning for Speech
Recognition (XLSR-53)," Interspeech 2021 (arXiv:2006.13979).**

> _Summary (3–4 sentences) — M:_
> TODO. (Cross-lingual pre-training — directly relevant to our multilingual gap.)

**[9] Tak et al., "Automatic speaker verification spoofing and deepfake detection
using wav2vec 2.0 and data augmentation," Odyssey 2022 (arXiv:2202.12233).**

> _Summary (3–4 sentences) — SK:_
> TODO. (SSL front-end + augmentation — a template for our pipeline.)

---

## Strand 3 — Voice cloning / TTS (the attack side)

**[10] Casanova et al., "YourTTS: Towards Zero-Shot Multi-Speaker TTS and Zero-Shot
Voice Conversion for Everyone," ICML 2022 (arXiv:2112.02418).**

> _Summary (3–4 sentences) — L:_
> TODO.

**[11] Kim et al., "Conditional Variational Autoencoder with Adversarial Learning for
End-to-End Text-to-Speech (VITS)," ICML 2021 (arXiv:2106.06103).**

> _Summary (3–4 sentences) — L:_
> TODO.

**[12] Wang et al., "Neural Codec Language Models are Zero-Shot Text-to-Speech
Synthesizers (VALL-E)," 2023 (arXiv:2301.02111).**

> _Summary (3–4 sentences) — L:_
> TODO. (Context for how easy high-quality cloning has become.)

**[13] Casanova et al., "XTTS: a Massively Multilingual Zero-Shot Text-to-Speech
Model," 2024 (arXiv:2406.04904).**

> _Summary (3–4 sentences) — L:_
> TODO. (Our seen-attack generator — XTTS-v2. Connect to the spoof corpus.)

---

## Strand 4 — Code-switching / Indic speech + generalisation & robustness

**[14] Diwan et al., "MUCS 2021: Multilingual and Code-Switching ASR Challenges for
Low-Resource Indian Languages," Interspeech 2021 (arXiv:2104.00235).**

> _Summary (3–4 sentences) — L:_
> TODO. (Our primary code-mixed bonafide + Stage-3 adaptation source.)

**[15] Javed et al. (AI4Bharat), "IndicVoices: Towards Building an Inclusive
Multilingual Speech Dataset for Indian Languages," 2024 (arXiv:2403.01926).**

> _Summary (3–4 sentences) — L:_
> TODO. (Tamil + code-switched eval columns.)

**[16] Müller et al., "Does Audio Deepfake Detection Generalize?," Interspeech 2022
(arXiv:2203.16263).**

> _Summary (3–4 sentences) — SK:_
> TODO. (The closest prior work on the generalisation question we quantify.)

**[17] Yi et al., "ADD 2022: The First Audio Deep Synthesis Detection Challenge,"
ICASSP 2022 (arXiv:2202.08433).**

> _Summary (3–4 sentences) — SK:_
> TODO. (Real-world/low-quality detection — motivates our channel-matched protocol.)

---

## After summaries are written

- Each member cross-reviews the other two strands (edit for accuracy).
- Deduplicate overlapping claims; agree the narrative arc: benchmarks → SSL
  front-ends → cheap cloning attacks → **the unmeasured code-mixed + telephony
  gap our work fills**.
- Port the finalised summaries into `paper/sections/related_work.tex` (SK owns
  the Related Work section per the Week-9 split).
