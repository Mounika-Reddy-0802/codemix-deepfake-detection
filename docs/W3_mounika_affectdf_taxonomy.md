# W3 — Mounika: AffectDF anchor + attack taxonomy + environment verification

Branch: `week3-mounika-affectdf-taxonomy` · Tasks: **W3-T3**, **W3-T6** (code half)

## What I built

**1. AffectDF related-work anchor (`docs/related_work_notes.md`, Strand 0).**
Read `Research/2608.05507v1.pdf` and wrote the summary from the paper's own
tables, not from our plan's paraphrase. The numbers that matter to us:
AASIST **0.83% EER on ASVspoof 2019 → 56.40% on AffectDF**; RawNet2 4.60% →
59.71%; XLSR-Mamba degrades least (0.20% → 29.78%). The finding we replicate on
our axis is the reverse direction — **AASIST retrained on AffectDF scores 44.52%
EER back on ASVspoof 2019** (their Table 13), which is exactly the trade-off
W7-T2 tests for S3 vs S2.

**2. The precise gap statement.** AffectDF covers emotional, attack-family and
speaker shift; the literature it cites covers multilingual, channel,
environmental and in-the-wild shift. **Intra-utterance code-switching is
examined nowhere** — multilingual anti-spoofing tests one language per utterance,
not a speaker alternating Hindi and English inside one sentence. Draft paper
wording is in the notes so Related Work can be written straight from it in W10.

**3. Attack taxonomy (`docs/attack_taxonomy.md`) in their Table-1 format.**
Attack ID / generation type / model / base data / samples / speakers / split, plus
a "seen by" column making the tool firewall explicit. Our IDs are prefixed `CM` so
they can never be confused with AffectDF's A01–A21 in a table that cites both.
Encodes the four disjointness rules and maps each attack to its gap-matrix column.

**4. `src/utils/env_check.py` (W3-T6 code half).** Verifies what a process can
verify and *labels the rest `manual`* rather than counting it as a pass. Secrets
are never echoed — `describe_secret()` reports length and a prefix class only, so
the output is safe to paste into a PR. It also flags a `REPLACE_WITH_*`
placeholder as an **error**, not as "missing", because a placeholder looks
configured and is not. README gained a resource table for the links.

## How to run it

```bash
python -m src.utils.env_check              # offline
python -m src.utils.env_check --network    # + HF token authentication
python -m src.utils.env_check --encoders   # + load the three SSL encoders
```

## Numbers

- **17 tests** in `test_env_check.py`, all green; `ruff` clean.
- Environment on the **GPU laptop** (17 Aug): **ok=8, missing=2, error=0, manual=8**
  — up from `ok=3, missing=7` on the dev laptop. The full ML stack now imports
  (`torch 2.8.0+cu128`, `transformers 4.57.6`, `librosa 0.11.0`, `soundfile`,
  `pandas`, `numpy`) and CUDA resolves to an RTX 3050 6 GB. The 2 remaining
  blockers are `WANDB_API_KEY` and `HF_TOKEN`, both unset.
- Getting from 3 to 8 needed three dependency pins corrected — `librosa` 0.11.0
  (coqui-tts floors it), `huggingface-hub` 0.36.2 (transformers 4.57 needs ≥0.34)
  and `websockets` 12.0 (gradio-client caps it below 13). Before that,
  `pip install -r requirements.txt` failed outright with `ResolutionImpossible`.
- AffectDF figures cited above are read from the PDF's Tables 2, 13 and 14.
- Appendix-G target for W5-T4: their low-level logistic regression reaches
  **53.16% EER** on AffectDF test (near chance). Ours must land near chance too,
  or the pipeline is leaking a shortcut.

## The compute plan changed under us

The v2 plan sizes Weeks 5–9 around *3 Kaggle accounts × 30 GPU-hours/week*, with S1
and S3 training in parallel on separate accounts (W5-T3). Two things broke that:

- The pilot platform moved Kaggle → Colab → **the team's own GPU laptop**. Colab has
  no comparable weekly quota and disconnects idle sessions; the laptop has one
  RTX 3050, so there is no parallelism at all.
- **6 GB of VRAM is tight for training.** Generation fits comfortably; wav2vec2 /
  WavLM fine-tuning at a useful batch size may not.

**Revisit the Week-5 training plan before launching S1/S3** — buy Colab Pro, keep
Kaggle purely for the long runs, or serialise S1 then S3. This is a decision, not a
detail: it sets how long Weeks 5–9 actually take.

## What's next (blocked on humans)

- **`WANDB_API_KEY` and `HF_TOKEN` unset** — the 2 machine-checkable blockers.
- **Kaggle GPU on all three accounts unverified** — needs a login per account.
- **IndicVoices / IndicTTS-Deepfake terms not accepted** (gated, browser-only).
- **ASVspoof 2019 LA still not downloaded** — the S1 anchor, and Stage-1's *only*
  training corpus. The local pull stalled at 25 % (1.75 GB of 7.11 GB) because
  `datashare.ed.ac.uk` resets on sustained transfers. **Nothing trains until this
  lands**, which makes it the single highest-value item on this list.
- Sample counts in the taxonomy are **targets**; real counts land at the Week-5
  freeze after Week-4 generation.
