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

**Done (Mounika, `docs/results/seed_stability_v1.md`).** Five seeds each for the
deployed model, S2 XTTS-only and S3 XTTS-only, phone line, published configs
unchanged (`notebooks/kaggle_w12_review_runs.ipynb`; `src/reporting/seeds.py`). The
deployed seed-1234 re-run reproduces the published run (CM04 30.14 vs 30.99).
- Deployed: eval pool 3.27 ± 1.17 (beats floor in 3 of 5 seeds), **CM04 35.59 ± 3.35
  (0 of 5)**, RVC 5.33 ± 1.27 (5 of 5). The published 30.99 on Tortoise was a
  favourable draw.
- S3 XTTS-only: **CM04 19.72 ± 5.36, below its floor in every seed (3 of 5
  significant)**, but worse than its RVC floor in 4 of 5.
- Seed spread on CM04 is 3-5 points for every system.

**Done in the paper.** `results.tex`: a seed table (`tab:seeds`, `sec:seeds`), and
Findings 1, 3, 5 and the summary quote seed means. Abstract, introduction and
conclusion now state the Tortoise result from the seeds, not the single run.

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

**Done (Krishna, `docs/results/variants_v1.md`).** The deployed recipe trained with
four variants (crop; crop + augment; crop + layer mix; all three), chosen by the rule
fixed in advance (dev EER only, ties within 0.5 points to fewer changes). The
published recipe stays: no variant improves dev EER. The two layer-mix variants first
crashed (strict load of the Stage-1 checkpoint, fixed in `load_initial_weights`) and
were re-run.

**Exploratory.** Crop + augment + layer mix, singled out after seeing its CM04 score
(26.43), was retrained with four more seeds: **CM04 26.63 ± 1.66 against 35.59 ± 3.35
for the published recipe**, every variant seed below every recipe seed (exact
Mann-Whitney p = 0.008; 0.016 on the four seeds trained after the choice). RVC rises
5.33 → 7.72; against the Tortoise floor it stays inconclusive in every seed. Its dev
EER equals the recipe's, so the in-domain dev set cannot see unseen-tool gains.
Reported in the paper as a lead to confirm on a second unseen tool
(`results.tex`, `sec:variants`; conclusion, future work).

**Not done.** Score fusion of the deployed model with S3 XTTS-only: it would need
weights set on a dev set that, as above, cannot see the unseen tool.

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

**Done (new speakers, new style).** HiACC adult speech, 24 speakers no model has heard,
talking spontaneously, scored over the phone line as a false-alarm test (evaluation
only; the child partition untouched). The deployed model calls **54.37%** [44.43,
64.15] of their genuine speech fake, against 0.89% for the read lectures of the
evaluation pool; every speaker is above 10% (13.9-89.5%), and clip length does not
explain it (`docs/results/hiacc_false_alarm_v1.md`, Lahari). The data chapter now
states who each model hears (21 speakers for the deployed model) and who it is tested
on (51 unseen speakers), with a speaker table (`dataset.tex`, `sec:data-speakers`).

**Planned.** (b) Add the existing CM01 XTTS clones of the 13 RVC-train speakers to
training (8 → 21 cloned voices, no new generation), subject to `config_guard`. (c) A
new frozen speaker pool from unused MUCS speakers for a larger unseen-speaker test.
(d) The HiACC test on the Kaggle variant models, to see whether augmentation lowers
the false alarms.

## 4. Hinglish phone-line vs clean-English trade-off

> Quantify the trade-off between improved Hinglish phone-line detection and degraded
> clean-English performance after channel-matched adaptation; discuss deployment.

**Checked.** Partly addressed already. English EER: S1 0.85% → deployed 2.42%; S3
collapses to 50.9-56.4%. Channel training *protects* English relative to clean
training (2.42% vs 3.75%; 2.55% vs 15.83%). At the deployed threshold the deployed
model catches 77.4% of English fakes against 93.5% for S1
(`classification_metrics_v1.md`). Not yet measured: channel models on clean Hinglish
after normalisation, and English over the phone line.

**Done.** Both missing conditions measured on the laptop CPU.
- *Other condition* (Mounika, `docs/results/tradeoff_v1.md`): all four S2 adapters on
  the three Hinglish sets in the condition they were not trained in, 11 of 12 cells.
  Training over the phone line costs little on clean audio (deployed, evaluation pool:
  2.75% → 6.71%); clean training costs a lot over the line (1.99% → 22.97%). Tortoise
  is harder over the line whatever the training condition.
- *English over the phone line* (Lahari, `docs/results/english_channel_v1.md`): a fixed
  6,004-clip ASVspoof sample (all 13 attacks, all 67 speakers) through the same G.711
  chain. The deployed model goes from 2.30% to **24.20%**, flagging 7.45% of genuine
  English clips at its threshold; S1, never trained on phone audio, holds 6.50%.

**Done in the paper.** `results.tex` has a new subsection, "The deployment trade-off"
(`sec:tradeoff`), with the other-condition table and a deployment paragraph: the
deployed model suits Hindi-English calls but should not judge English calls; the
summary now says "clean English".

**Still to do.** Two cells stopped when the laptop ran out of memory (the deployed
adapter on the clean RVC set, about 2 minutes; S1 on the clean English sample, about
17 minutes).

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

**Done in the paper.** `methodology.tex` now defines the matched floor, the paired
speaker-level test, a "reading the floor" paragraph, and the oracle floor (disclosed,
kept for reference). `results.tex` has a new floors table (matched and oracle), the
main table marked by the paired test (bold = beats, italic = inconclusive,
dagger = worse), and Findings 1, 2 and 5 and the summary rewritten to match. The
abstract, introduction (a new "matched shortcut floors" contribution) and conclusion
no longer say the deployed model *loses* to the Tortoise floor: it is below its
matched floor (30.99% vs 37.79%) but inconclusive at the speaker level.

**Done.** Both floor figures (`systems_s2_s3.png`, `ablation_rvc.png`) were redrawn
with the matched floor per bar (`src/reporting/figures.py`).

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

**Done in the paper.** `related_work.tex` rewritten: benchmarks and generalisation;
detectors and self-supervised front-ends; channel and codec robustness; parameter-
efficient adaptation; multilingual and code-switched detection; and a "how this work
differs" subsection. The old disclaimer that multilingual detection was out of scope
is gone. AdamW, G.711, Opus, the bootstrap and the layer-wise claim are now cited.
All 38 references are cited and every citation resolves.
The verified literature shows each ingredient has precedent separately (real
telephone/VoIP channel adaptation, Odyssey 2026; intra-utterance code-switched
deepfakes, Arabic-English, ICLR 2026; LoRA for detectors, 2023 and ICASSP 2026), so
the contribution is stated as the combination: Hinglish code-switching within an
utterance, trained and tested over the same telephone channel, judged against a
matched shortcut floor per test set and by English retention, and deployed on live
calls.

**Also fixed.** The `tab:gap` rows in `results.tex` ended in `\` instead of `\\`.
