# Encoder ablation v1 — is XLSR-53 a better front-end than wav2vec2-base?

**Week 12.** Prompted by a direct question: would rebuilding S3 on XLSR-53, a
multilingual encoder, give a significant improvement on code-mixed speech? This
records what was measured, what was tried and did not work, and what is still open.

## Short answer

**XLSR-53's frozen representations are significantly better for this task, especially
on the unseen tool, but the advantage does not survive training.** On a 16 GB T4
(section 3b): under LoRA, XLSR-53 does not learn at all; under the full S3 fine-tune it
learns but ties wav2vec2-base on the unseen tool (32.08% vs 31.67%) and is far worse on
RVC (41.03% vs 5.98%). A bigger multilingual encoder does not fix unseen-tool
generalisation here.

## 1. Representation quality: linear probe on frozen features

Both encoders frozen. Each clip's last hidden layer is mean+std pooled, features are
standardised on the training set, and the same class-balanced logistic regression is
fitted on the adaptation training set (`codemix_adapt_train_rvc_channel20`). EER on
the channel-condition test sets, with a **paired** bootstrap (1,000 stratified
resamples, the same clips drawn for both encoders) so the interval is on the
difference itself.

| Set (channel) | Floor | wav2vec2-base | XLSR-53 | XLSR − base, 95% CI |
|---|---:|---:|---:|---:|
| Eval pool (XTTS, seen) | 10.01 | 9.18 [8.24, 10.09] | **5.93** [5.12, 6.61] | **−3.25** [−4.19, −2.40] |
| CM04 (Tortoise, unseen) | 25.79 | 51.77 [48.88, 54.63] | **38.20** [34.73, 41.20] | **−13.57** [−17.62, −10.03] |
| RVC test | 22.46 | 21.49 [18.42, 25.20] | **16.96** [14.38, 20.20] | **−4.53** [−8.73, −0.16] |

Every interval on the difference excludes zero. The unseen-tool gap is the largest
and the most important, since unseen-tool-over-channel is the weakest result in the
paper.

**This overturned a prediction.** The interpretability probes (P-034) show the
decision forming by transformer layer 3, before linguistic content, which suggested a
multilingual encoder would add little. The probe says otherwise. The likely source is
scale and pretraining-data diversity giving more robust low-level features, not
multilinguality as such — but this ablation cannot separate the two.

**Caveats.** Frozen features, not an adapted model: absolute numbers sit well above
the adapted systems, and adaptation could narrow or widen the gap. XLSR's 38.20% on
CM04 is still above that set's floor, so it is a better starting point, not a solution.

## 2. What the English Stage-1 start contributes (a by-product)

The same LoRA recipe as the deployed S2, run from the *pretrained* encoder instead of
from the English Stage-1 checkpoint, isolates what that stage buys:

| Set (channel) | Floor | S2, from English S1 | Same recipe, from pretrained |
|---|---:|---:|---:|
| Eval pool (XTTS) | 10.01 | **2.75** | 4.59 [3.91, 5.09] |
| CM04 (Tortoise) | 25.79 | **30.99** | 39.11 [36.61, 41.62] |
| RVC test | 22.46 | **5.01** | 9.86 [7.27, 12.77] |

Starting from the English anti-spoofing checkpoint roughly halves the error on every
set. This is direct evidence for the two-stage design, which the paper previously
justified only by argument.

## 3. Adapting XLSR — what was tried, and why it is not done

**Hardware.** Measured on the RTX 3050 (6 GB) at the adapters' real settings
(batch 8, 4 s clips, fp16):

| Recipe | Encoder | Peak GPU | Step time |
|---|---|---:|---:|
| LoRA | wav2vec2-base | 1.37 GB | 0.29 s |
| LoRA | XLSR-53 | 3.99 GB | 0.45 s |
| Full fine-tune (S3) | wav2vec2-base | 2.38 GB | 0.34 s |
| Full fine-tune (S3) | XLSR-53 | **6.45 GB** | **38.9 s** |

S3 on XLSR-53 exceeds the card and spills to system memory — about 39 hours for the
10-epoch recipe. It needs a 16 GB GPU.

**XLSR does not train under the shared LoRA recipe.** Training loss stays at
ln 2 = 0.693, chance, where wav2vec2-base falls from 0.689 to 0.537 over the same 200
steps. Diagnosed and tested:

| Hypothesis | Test | Result |
|---|---|---|
| fp16 saturation | encode identical clips in fp16, bf16, fp32 | **ruled out** — output 100% finite and identical in all three |
| feature scale | measure pooled-feature spread | XLSR's spread is ~40× smaller (0.002 vs 0.085) |
| fix: LayerNorm before pooling | 200 training steps | **no effect** — 0.699 → 0.687 |
| fix: BatchNorm before pooling | 200 training steps | **no effect** — 0.700 → 0.682 |

Both normalisation attempts were reverted; none is in the codebase.

The linear probe finds XLSR's information because it solves a convex problem to its
optimum. Short SGD with LoRA does not find it. That points at the training recipe for
a model three times the size — learning rate, pooling, schedule — which needs a proper
search on a stable GPU, not further one-line changes.

The laptop GPU also faulted twice during XLSR runs (`CUDA error: unknown error`, and a
checkpoint write failing when the page file grew). This is the driver issue already
documented in `src/training/evaluate.py`, not the experiment.

## 3b. On a 16 GB GPU (Kaggle T4, 11 Oct)

Run with `notebooks/kaggle_w12_xlsr_encoder.ipynb` at commit `a83d36a`, every step
completed without error. Per-clip scores and the summary are in
`results/ablation_encoder/kaggle_xlsr_run/xlsr_results/`.

**Stage 1, learning-rate search** (XLSR-53 + LoRA, 300 steps per rate):

| Rate | Last loss | Dev EER | Score std | Outcome |
|---|---:|---:|---:|---|
| 1e-4 | 0.689 | 0.493 | 0.0002 | stuck at chance |
| 3e-4 | 0.690 | 0.478 | 0.0004 | stuck at chance |
| 1e-3 | 0.695 | 0.699 | 0.0000 | collapsed (constant output) |
| 3e-3 | 0.764 | 0.382 | 0.0000 | collapsed (constant output) |

A collapsed model returns the same score for every clip, so its dev EER is an artefact
of tie-breaking, not a measurement. No rate learned, so stage 2 gave XLSR the full 5
epochs at 1e-4, the wav2vec2-base rate.

**Stages 2 and 3**, EER (%), channel condition, 95% bootstrap intervals over clips:

| System | Eval pool (XTTS) | CM04 (Tortoise) | RVC test |
|---|---:|---:|---:|
| Floor (v1) | 10.01 | 25.79 | 22.46 |
| wav2vec2-base + LoRA, from pretrained | 6.40 [5.72, 7.16] | 48.67 [46.29, 51.52] | 16.96 [13.90, 20.04] |
| XLSR-53 + LoRA, from pretrained | 42.23 [40.77, 43.74] | 58.50 [55.02, 60.57] | 55.25 [50.41, 59.45] |
| S3 native on XLSR-53 (lr 1e-5, 10 epochs) | 5.17 [4.36, 6.03] | 32.08 [29.04, 34.77] | 41.03 [37.34, 45.39] |
| S3 native on wav2vec2-base (published) | 1.79 | 31.67 | 5.98 |

- **XLSR-53 + LoRA never learns.** Its scores barely vary between clips (std ≈ 0.001),
  and it sits near chance even on the seen tool.
- **S3 on XLSR-53 learns but gains nothing.** Its CM04 interval contains the
  wav2vec2-base figure, it is worse on the seen tool, and on RVC it is worse than the
  floor. The frozen-probe advantage of section 1 does not carry into a trained model.
- **The same recipe varies a lot between runs.** wav2vec2-base + LoRA from pretrained
  gave 39.11% on CM04 on the laptop (section 2) and 48.67% on Kaggle. Single-run CM04
  numbers cannot carry a claim; this is why the external review's request for seed
  repeats matters (`docs/external_review_response.md`).

The laptop XLSR checkpoints were deleted on 11 Oct: one never learned (dev EER 57.3%,
constant output) and the other was a truncated write that could not be loaded. The
wav2vec2-base checkpoint of section 2 is kept.

## 4. Also found

**WavLM is incompatible with the project's LoRA wrapper.** WavLM's attention reads
`q_proj.bias` directly, which `LoRALinear` does not expose, so any WavLM run with LoRA
crashes on the first forward pass. A one-line property on `LoRALinear` would fix it.

## What to do next

1. Done on 11 Oct (section 3b): the learning-rate search ran on a T4 and found no rate
   at which XLSR-53 + LoRA learns; S3 on XLSR-53 ties wav2vec2-base on CM04.
2. For the paper: report the probe result, the Stage-1 contribution and the negative
   XLSR result together. The last-layer readout is the likely reason the probe's
   advantage does not transfer (the decision forms by layer 3, P-034); the opt-in
   `layer_weighting` option now tests that directly.
3. Repeat the section-2 comparison over seeds before stating the Stage-1 effect as a
   number.

## Reproduce

Configs: `configs/ablation_encoder_w2v2base_lora.yaml`,
`configs/ablation_encoder_xlsr53_lora.yaml` (both pass `src.training.config_guard`).
The wav2vec2-base checkpoint and per-clip scores are in
`checkpoints/ablation_encoder/run_w2v2base_lora/` and `results/ablation_encoder/`.
