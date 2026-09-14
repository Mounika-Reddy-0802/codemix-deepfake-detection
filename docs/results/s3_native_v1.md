# S3 native training: the three systems side by side

**W5-T3 / W6-T2 / W7-T2 / W9-T2, owner M.** Four S3 models fine-tune the pretrained
wav2vec2 encoder directly on code-mixed audio, with no English spoofing data and no
start from the English checkpoint (P-028). Each reads exactly the train and dev
manifests of one W10 LoRA adapter, so **S2 against S3 is the training method and
nothing else**. All eight models were then scored on the 71,237-clip ASVspoof 2019
LA eval partition in the same Kaggle session.

Raw results: `docs/results/s3_native/` (pooled JSONs, per-clip scores). Built from
them: `experiments/results/systems_s1_s2_s3.json` and
`experiments/results/s3_reverse_degradation.json` (`python -m src.reporting.systems`).
Figure: `experiments/figures/systems_s2_s3.png`.

## Result

EER %, 95% bootstrap CI in brackets. Code-mixed sets are scored in each model's own
condition. **Bold** = the CI upper bound is below that set's shortcut floor.
~~Struck through~~ = not below its floor. English has no floor.

| Training data | Condition | System | Eval pool (XTTS) | CM04 (unseen Tortoise) | RVC test | English |
|---|---|---|---:|---:|---:|---:|
| XTTS | clean | S2 LoRA | **1.46** [1.01, 1.94] | **5.44** [4.05, 6.57] | **23.91** [20.33, 27.64] | 15.83 [15.54, 16.04] |
| | | S3 native | **1.46** [1.08, 1.82] | **27.72** [25.77, 30.14] | ~~45.07~~ [40.87, 50.24] | 56.42 [55.81, 56.93] |
| XTTS | channel | S2 LoRA | **4.41** [3.78, 5.19] | ~~28.16~~ [25.44, 30.20] | ~~29.56~~ [26.49, 32.79] | 2.55 [2.43, 2.67] |
| | | S3 native | **1.92** [1.43, 2.40] | **13.71** [12.30, 15.23] | ~~32.15~~ [28.43, 35.55] | 55.83 [55.31, 56.41] |
| XTTS + RVC | clean | S2 LoRA | **1.99** [1.47, 2.45] | **12.01** [10.42, 13.64] | **5.01** [3.55, 6.95] | 3.75 [3.63, 3.91] |
| | | S3 native | **0.83** [0.53, 1.14] | **21.39** [18.39, 24.58] | **8.24** [5.66, 10.51] | 50.90 [50.45, 51.35] |
| XTTS + RVC | channel | S2 LoRA | **2.75** [2.22, 3.37] | ~~30.99~~ [28.72, 33.84] | **5.01** [3.39, 6.95] | 2.42 [2.31, 2.54] |
| | | S3 native | **1.79** [1.44, 2.26] | ~~31.67~~ [28.20, 34.57] | **5.98** [4.19, 8.09] | 52.25 [51.58, 52.86] |

Floors: eval pool 5.17 / 10.01, CM04 31.21 / 25.79, RVC test 28.76 / 22.46
(clean / channel; `w10_norm_rvc.md`). **S1 English-only: 0.85% on English**
(W8-T1). S1 was never scored on the normalised code-mixed sets.

One check: S2 and S3 clean both score exactly 1.46% on the eval pool. Their
per-clip scores share no identical value (correlation 0.918). Both land on 23 missed
real clips of 1,566 and 35 passed fakes of 2,400 at the equal-error point.

## What it means

### 1. S3 is the ceiling on the tool it trained on
S3 matches or beats S2 on the seen tool in every configuration, down to **0.83%**
clean with RVC, and every one of those cells clears its floor. Native training on
code-mixed audio is the best detector of that audio.

### 2. It pays for that with English: reverse degradation, answered (W7-T2)
Every S3 model scores English at **50.90-56.42% EER**, which is chance. Its English
AUCs are 0.40-0.51, so the ranking is no better than random and in two models
slightly inverted. The S2 adapters stay at 2.42-3.75% (15.83% for the clean XTTS
adapter), against S1's 0.85%. Starting from the English checkpoint and adapting a
small set of weights keeps the conventional task. Training the encoder natively
does not. This is AffectDF's key finding (domain retraining wrecks conventional
performance), reproduced on our axis.

### 3. Neither method generalises cleanly to the unseen tool
- **Clean:** S2 is far better on unseen Tortoise (5.44% against S3's 27.72%).
- **Channel:** S3 is the only model that clears the CM04 channel floor (13.71%
  against 25.79%); S2 does not (28.16%).
- **With RVC added:** both get worse on Tortoise, and the S3 channel model loses
  its floor clearance (31.67%).

P-027's trade-off (more attack families, less unseen-tool generalisation) holds for
S3 as well as S2.

### 4. RVC needs RVC in training, whatever the method
Without RVC, both systems fail the RVC floor in the channel condition. With the
speaker-held-out RVC half, both reach 5-8%.

## The claims this supports

| Claim (plan, section 5) | Status |
|---|---|
| English-trained detectors collapse on code-mixed speech | Supported earlier (S1 53.71% / 54.92%, pre-normalisation) |
| LoRA adaptation closes the gap cheaply, English preserved | **Supported**: 1.46-4.41% seen tool, English 2.42-3.75% (clean XTTS 15.83%) |
| Native code-mixed training is the ceiling, at a cost | **Supported**: best seen-tool EER (0.83%), English at chance |
| Detection generalises to an unseen tool | **Partly**: clean S2 5.44%; channel only S3 XTTS-only 13.71% |

**For the live demo:** the phone-line S2 adapter trained with RVC
(`lora_norm_rvc_channel`) is the deployable model. It clears its floors on the seen
tool (2.75%) and RVC (5.01%) and keeps English (2.42%). It does **not** clear the
unseen-tool floor over a phone line, and the demo must say so.
