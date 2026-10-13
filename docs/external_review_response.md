# External review: response and progress

Reviewer: external reviewer, IIT Madras. Six comments, received October 2026. Every
comment is being addressed. This file records, for each one, what we checked, what we
changed, the evidence, and what is still to run. Status: **done**, **in progress**,
**planned**.

Rule for all new experiments: every choice (augmentation, layer mix, learning rate,
epoch) is made on the **dev set**; CM04 (Tortoise) and the RVC test set are scored
**once** per final system. Nothing is tuned on a test set.

---

## 1. Multiple random seeds and confidence intervals

> Evaluate the channel-matched LoRA model across multiple random seeds and report
> confidence intervals to establish the stability and reproducibility of the
> phone-line detection results.

**Checked.** Correct. Every config uses `seed: 1234`; the channel-matched model was
never retrained with another seed (`channel_matched_v1.md:135`). Our intervals
resampled clips, which covers test-set sampling but not training randomness, and is
too narrow with 12-15 test speakers. Run-to-run variation is real: one same-seed rerun
moved clean Hinglish from 13.92% to 10.36%, and one LoRA recipe gave 39.11% vs 48.67%
on CM04 on two machines.

**Done.** Speaker-level paired bootstrap implemented (`src/training/floor_compare.py`)
and applied to all eight main-table systems (`docs/results/shortcut_floor_v2.md`).

**Planned.** Retrain the deployed model, the XTTS-only channel model and S3 XTTS-only
channel with seeds 2025, 7, 42, 99 (published recipe unchanged); report mean ± sd
across the five seeds alongside the speaker-level intervals. About 20 T4-minutes per
S2 seed, 45 per S3 seed.

## 2. Generalisation to unseen tools (Tortoise)

> Investigate methods to improve generalization to unseen voice-cloning tools,
> particularly Tortoise-TTS (phone-line EER 30.99% vs shortcut floor 25.79%).

**Checked.** The numbers are right, and the problem is narrower than it looks.
Several systems already generalise to Tortoise: S2 trained on XTTS (clean) reaches
5.44%, and S3 trained on XTTS only reaches 13.71% over the phone line. The failure
appears when RVC is added to phone-line training (S2 28.16 → 30.99; S3 13.71 → 31.67).
Tried without gain: XLSR-53 (LoRA 58.50%; full fine-tune 32.08%,
`results/ablation_encoder/kaggle_xlsr_run/`). Against the matched floor (comment 5),
the deployed model's 30.99% is 6.80 points *below* its floor, inconclusive at the
speaker level, and S3 XTTS-only beats its floor clearly (Δ −19.93 [−32.95, −8.91]).

Found during the check: no augmentation anywhere in training; the classifier reads
only the encoder's last layer although the decision forms by layer 3 (P-034); and a
bug, the training crop of each clip is identical every epoch.

**Done (code, tested).** Opt-in training options, all off by default so published
configs train exactly as before:
- `crop_per_epoch` — a fresh random crop each epoch (fixes the bug);
- `augment: shortcut_v1 | channel_v1` — on-the-fly randomisation of the recording
  cues the floor found (level, spectral tilt, coloured noise) and, for phone-line
  systems, a second G.711 pass at a random 5-30 dB SNR (`src/data/augment.py`);
- `layer_weighting` — a learned mix of every encoder layer (`LayerWeightedSum`).

**Planned.** Train the deployed recipe with these options, select on dev only, then
score CM04/RVC once; combine with 5 seeds (comment 1). Score fusion of the deployed
model with S3 XTTS-only, weights set on dev.

## 3. More speakers and speaking styles

> Expand the Hinglish adaptation dataset beyond the current 10-speaker pool and
> evaluate performance on additional unseen speakers.

**Checked.** Partly correct. "10" is the XTTS adaptation pool (8 train + 2 dev); the
deployed model trains on 21 speakers (RVC adds 13), which the paper does not say.
Unseen-speaker testing already exists (15 eval speakers, 12 RVC-test speakers, no
overlap, enforced by `config_guard`). The real gaps: one speaking style (MUCS read
lectures), no spontaneous speech, no per-speaker results.

**Done.** Per-speaker EERs for every system and test set
(`experiments/results/floor_v2/per_speaker_eer.csv`). The deployed model's CM04 EER
ranges 6.4-45.5% across speakers (median 28.1%).

**Planned.** (a) Score HiACC adult speech (24 speakers, spontaneous; evaluation-only
use is permitted) as a false-alarm test. (b) Add the existing CM01 XTTS clones of the
13 RVC-train speakers to training (8 → 21 cloned voices, no new generation), subject to
`config_guard`. (c) A new frozen speaker pool from unused MUCS speakers for a larger
unseen-speaker test.

## 4. Hinglish phone-line vs clean-English trade-off

> Quantify the trade-off between improved Hinglish phone-line detection and degraded
> clean-English performance after channel-matched adaptation; discuss deployment.

**Checked.** Partly addressed already. English EER: S1 0.85% → deployed 2.42%; S3
collapses to 50.9-56.4%. Channel training *protects* English relative to clean
training (2.42% vs 3.75%; 2.55% vs 15.83%). At the deployed threshold the deployed
model catches 77.4% of English fakes against 93.5% for S1
(`classification_metrics_v1.md`). Not yet measured: channel models on clean Hinglish
after normalisation, and English over the phone line.

**Planned.** Score those two missing conditions (about 1 T4-hour plus about 50 minutes
per model for English); add a trade-off table and a deployment paragraph.

## 5. Interpreting the shortcut floor

> Explain more clearly how the shortcut-floor results should be interpreted ... how
> residual recording-domain differences affect the reliability of comparisons.

**Checked.** Correct, and it exposed a methods error: the CM04 floor was fitted on
Tortoise clips the detector never sees, and scored on different clips from the
detector. On the same clips the deployed model is level with that floor (26.78% vs
25.79%).

**Done.** Matched floors (fitted on each detector's own training data, scored on the
detector's exact clips), paired speaker-level test for all 24 system × set cells,
interpretation paragraph and residual-cue list: `docs/results/shortcut_floor_v2.md`.

**Planned.** Move the v2 table, the reading paragraph and the oracle/matched
distinction into `paper/sections/methodology.tex` and `results.tex`.

## 6. Bibliography and positioning

> Provide complete bibliographic details and clarify how the channel-matched approach
> differs from existing multilingual and audio deepfake detection methods.

**Checked.** Correct. `refs.bib` had 10 entries and no DOIs; AdamW, G.711, Opus, the
bootstrap and a layer-wise claim were named without citations; related work excluded
multilingual detection and cited no channel/codec-robustness or adapter-based
anti-spoofing work.

**Done.** `paper/refs.bib` rebuilt: 38 entries (the 10 cited ones completed, 28
added), every field checked against the publisher, Crossref or the ISCA Archive;
sources per entry in `docs/references_verification.md`. One error corrected in an
existing entry (wav2vec 2.0 second author). New references cover ASVspoof 2021,
AASIST, RawNet2, RawBoost, In-the-Wild, ADD 2022, MLAAD, XLSR-53, XLS-R, WavLM,
layer-wise SSL analysis, AdamW, G.711, Opus, the bootstrap, codec/channel
robustness, parameter-efficient anti-spoofing, and multilingual/Indic deepfake
datasets (IndicSynth, SpeechFake, Indic-CodecFake).

**Planned.** Related-work subsections and the "how this work differs" paragraph.
The verified literature shows each ingredient has precedent separately (real
telephone/VoIP channel adaptation, Odyssey 2026; intra-utterance code-switched
deepfakes, Arabic-English, ICLR 2026; LoRA for detectors, 2023 and ICASSP 2026), so
the contribution is stated as the combination: Hinglish code-switching within an
utterance, trained and tested over the same telephone channel, judged against a
matched shortcut floor per test set and by English retention, and deployed on live
calls.

**Also fixed.** The `tab:gap` rows in `results.tex` ended in `\` instead of `\\`.
