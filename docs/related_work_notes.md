# Related-work notes (becomes the paper's Related Work section)

17 references from the proposal, grouped into **four survey strands**, plus
**Strand 0 — the base paper**, which the whole Related Work section is now
anchored on. Each member deep-reads ~6 papers and writes a **3–4 sentence
summary** in the block under each reference. Keep summaries factual: (1) what the
paper does, (2) its key method/result, (3) how it relates to *our* code-mixing
generalisation-gap study.

---

## Strand 0 — the base paper (W3-T3, owner M)

**[0] Mahapatra, Zhao, Chandra, Zhang, Du, Ulgen, Lee, Andrews, Busso, Sisman,
"AffectDF: The Most Comprehensive Benchmark for Speech Deepfake Detection against
Emotionally Expressive Attacks," arXiv:2608.05507v1 [eess.AS], 6 Aug 2026.**
(Johns Hopkins / HK PolyU / CMU. Local copy: `Research/2608.05507v1.pdf`.)

> _Summary — M:_
> AffectDF is a ~260-hour benchmark of emotionally expressive speech deepfakes
> built from 21 spoofing attacks over five emotional states, spanning TTS, VC,
> emotional VC, and — new relative to prior emotional-spoofing sets — large
> audio-language-model (LALM) attacks such as Qwen2.5-Omni, Kimi-Audio and
> MiniCPM. Its train/dev/test partitions are constructed with **disjoint speakers
> and disjoint attack systems** (train A01–A05, dev A06–A07, test A08–A21), which
> is what makes its cross-attack generalisation claims readable. Benchmarking
> state-of-the-art detectors trained on conventional data shows severe collapse
> under emotional shift: AASIST goes from **0.83% EER on ASVspoof 2019 to 56.40%
> on AffectDF**, and RawNet2 from 4.60% to **59.71%**; SSL-based systems degrade
> less but still substantially (XLSR-Mamba 0.20% → 29.78%). Crucially, retraining
> *on* AffectDF does not fix it — it damages conventional performance instead
> (AASIST trained on AffectDF scores 44.52% EER back on ASVspoof 2019), so the
> paper documents a failure mode and stops there without testing a mitigation.

### The precise gap we claim

AffectDF covers, explicitly and well:

- **emotional/prosodic shift** (five emotional states, acted vs spontaneous),
- **attack-family shift** (TTS / VC / EVC / LALM, 21 systems),
- **speaker shift** (disjoint speakers per split).

Prior work it situates itself against covers **multilingual** shift (Müller et
al. 2024; Ali et al. 2025), **channel** shift (Liu et al. 2023), **environmental**
shift (Cheng et al. 2024) and **in-the-wild** conditions (Müller et al. 2022).

What is examined nowhere in that literature is **intra-utterance code-switching**.
AffectDF is entirely English (ESD + MSP-Podcast). Multilingual anti-spoofing work
tests one language per utterance; it does not test a speaker alternating between
Hindi and English *within a single utterance*, which is the ordinary register of
~600 million speakers and the register a fraud call in India actually arrives in.

> **Gap statement (paper wording draft):** "Recent benchmarks quantify detector
> collapse under emotional, multilingual, channel and in-the-wild shift. Shift
> arising from intra-utterance code-switching remains unexamined, and no prior
> work tests whether such a gap can be closed. We measure the code-mixing gap
> under channel-matched telephony conditions and evaluate a parameter-efficient
> mitigation against a native-training ceiling."

### Methodology we adopt from it (and where)

| AffectDF element | Where it lands in our work |
|---|---|
| Table 1 attack-ID taxonomy with disjoint speakers *and* attack systems | `docs/attack_taxonomy.md` (W3-T3) |
| Rich per-clip protocol files | generation metadata, W4-T1/T2/T3 |
| Appendix G low-level-cue logistic regression (8 features: RMS mean, RMS std, peak amplitude, clipping ratio, DC offset, ZCR, spectral rolloff, HF energy ratio) — theirs reaches **53.16% EER**, i.e. near chance, proving the split is not a low-level shortcut | W5-T4, target near-chance on our data |
| min t-DCF reported alongside EER | W6/W7 result tables |
| Fixed-generator acted-vs-spontaneous contrast (their Table 15) | our read-vs-spontaneous HiACC contrast, W8-T3 |
| Emotion-wise slicing (their Table 3) | our switch-density slicing, W7-T3 |
| The dataset itself, CC BY-NC 4.0, research/non-commercial only | external cross-eval column, W8-T2 |

### Where we differ deliberately

We do not compete on scale (260 h, 21 attacks, A100/H100 clusters). We contribute
a different axis (code-mixing), a **channel-matched** protocol they do not have
(they evaluate clean audio), a **mitigation** they never test (S2 LoRA vs S3
native), and a deployment (live call detection). Their finding that domain
retraining wrecks conventional performance is the one we replicate on our axis:
W7-T2 asks whether S3 loses English performance where S2 preserves it.

---

## Verification checklist for this entry

- [x] Numbers read from `Research/2608.05507v1.pdf` (Tables 2, 13, 14), not from
      the project plan's summary.
- [ ] Confirm the arXiv id and version are still current at submission time
      (v1 dated 6 Aug 2026; check for a v2 before the paper goes out).
- [ ] Confirm the venue if it is published between now and Week 11.
- [ ] Re-check the AffectDF licence terms before downloading any subset (W8-T2).

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
