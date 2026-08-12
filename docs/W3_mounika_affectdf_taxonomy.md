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
- Current environment on this machine: **ok=3, missing=7, error=0, manual=8** —
  i.e. 7 blocking issues before Week-4 generation and 8 items only a human can
  confirm. `WANDB_API_KEY` and `HF_TOKEN` are unset; `torch`/`pandas`/`numpy`
  import, `transformers`/`librosa`/`soundfile`/`torchaudio` do not.
- AffectDF figures cited above are read from the PDF's Tables 2, 13 and 14.
- Appendix-G target for W5-T4: their low-level logistic regression reaches
  **53.16% EER** on AffectDF test (near chance). Ours must land near chance too,
  or the pipeline is leaking a shortcut.

## What's next (blocked on humans)

- **Kaggle GPU on all three accounts unverified** — needs a login per account.
- **W&B project and HF token not created/set**; IndicVoices and IndicTTS-Deepfake
  terms not accepted (both gated, browser-only).
- **ASVspoof 2019 LA not downloaded** — the S1 anchor. Blocked on "run downloads
  now".
- Sample counts in the taxonomy are **targets**; real counts land at the Week-5
  freeze after Week-4 generation.
