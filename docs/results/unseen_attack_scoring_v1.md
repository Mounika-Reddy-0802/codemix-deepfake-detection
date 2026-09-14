# Scoring the two attacks no detector had seen: CM02 and CM04

**W8-T1 (all columns, seen + unseen), owner M.** Every detector number before this
was measured against XTTS-v2, the tool the LoRA adapter was trained on. CM02 (RVC
voice conversion) and CM04 (Tortoise, the held-out tool) existed and were
QA-screened, but nothing had been scored on them. This does it, for all three
checkpoints, beside the shortcut floor each attack carries.

## Design

The real side of each comparison comes from **the same speakers as the fakes**, so
the EER cannot be earned by telling speakers apart (`src/data/scoring_manifests.py`).

| Attack | Real (bonafide) | Fake (spoof, QA-passed only) | Speakers |
|---|---|---|---|
| **CM02** RVC | 1,345 — the exact source segments the conversions were made from | 1,404 | 25 train-pool |
| **CM04** Tortoise | 1,566 — eval-pool bonafide from `codemix_eval.csv` | 458 | 15 eval-pool |

Clean condition, 4 s crops, `src.training.evaluate`. Run on **CPU**: this laptop
bluescreens (`0x133 DPC_WATCHDOG_VIOLATION`) within minutes of GPU load, and the CPU
runs finished without a single crash — see "How this was run".

## Result

| Checkpoint | **CM02** (RVC) | **CM04** (Tortoise, unseen) |
|---|---:|---:|
| Stage-1 baseline | 42.45% [40.70, 44.49] | 41.09% [38.92, 43.50] |
| LoRA, clean-trained | 31.90% [30.23, 33.80] | **5.03%** [4.15, 6.16] |
| LoRA, channel-matched | 28.48% [26.74, 30.08] | 13.10% [11.17, 14.76] |
| *Shortcut floor (8 cheap statistics)* | *22.28% – 22.42%* | *22.21%* |

EER with 95% bootstrap CI. AUC: CM04 0.9831 (clean LoRA) / 0.9416 (channel LoRA);
CM02 0.7491 / 0.7910.

## What it means

### 1. The adapter generalises to a tool it never saw — CM04

Trained on XTTS alone, the clean adapter separates Tortoise from real speech at
**5.03% EER, four times below the 22.21% shortcut floor**. The channel-matched
adapter reaches 13.10%, also below it. This is the first evidence behind the claim
"these results are not shortcut artefacts": an unseen generator, beaten by a margin
the eight low-level statistics cannot explain.

### 2. Voice conversion is a blind spot — CM02

Both adapters score **worse than the shortcut floor** on RVC (31.90% and 28.48%
against 22.3%). The error breakdown says why:

| | real called fake (@0.5) | fake called real (@0.5) |
|---|---:|---:|
| CM02, clean LoRA | 1.0% | **83.0%** |
| CM02, channel LoRA | 2.5% | **75.8%** |

The adapters wave most RVC fakes through as genuine. That is the attack family
doing what it was chosen to do: RVC starts from real speech and keeps its prosody
(P-021: 96.4% pitch-range retention), so a detector trained on text-to-speech
artefacts does not recognise it. **A detector trained on CM01 alone has not learned
voice conversion.** This is the strongest argument yet for W9-T1's XTTS+RVC
training ablation.

### 3. A caveat that was tested and did not hold

CM04's real clips come from the same bundle the adapter trained on, so a low EER
could have meant "familiar audio is real, everything else is fake". CM02 tests that
directly: its real clips are **fresh raw MUCS segments, not bundle audio**, and the
clean adapter still calls **99.0% of them real**. It has not learned "unfamiliar =
fake", so the CM04 result is not inflated that way.

### 4. EER hides the operating point

EER assumes the threshold is tuned. At the default **0.5**, the clean adapter lets
**47.2% of Tortoise fakes through** — its median P(real) on fakes is 0.28, well
separated from real (0.9996) in ranking but not by the default cut. The
channel-matched adapter lets 28.8% through at 0.5 despite its worse EER.
**The live demo must not use 0.5**; its threshold has to come from a DET curve
(W8-T5).

### 5. The baseline fails the way it always has

On both attacks Stage-1 calls **91.5–93.8% of real code-mixed speech fake**
(median P(real) 0.0005 on both classes) — the same collapse `gap_matrix_v1.md`
recorded. It is not detecting either attack; it rejects Hinglish outright.

## Limitations

- **Clean condition only.** No telephony (G.711) render of these two sets exists
  yet; the channel-matched column, where the project's primary result lives, is
  unmeasured for CM02 and CM04.
- **The normalisation bug is still unfixed** (`lowlevel_cue_check_v1.md`), so every
  number here is reported beside its shortcut floor rather than alone.
- **CM04 is 458 fakes against 1,566 real** (3.4 : 1 real-heavy) — the CI is honest
  about it, but the fake side is small.
- **CPU and GPU scores were mixed in one cache.** CM02's first 1,600 baseline scores
  came from the GPU run before it crashed; the rest from CPU. The difference is
  float-level and does not move an EER.

## How this was run

```bash
export DATA_ROOT=C:/dfdata
python -m src.data.scoring_manifests --attack cm02 --data-root "$DATA_ROOT"
python -m src.data.scoring_manifests --attack cm04 --data-root "$DATA_ROOT" \
    --spoof-dir cm04/cm04_outputs --gate-split

# per checkpoint x attack (baseline / lora_codemix / lora_codemix_channel)
python -m src.training.evaluate --checkpoint checkpoints/lora_codemix/best.pt \
    --manifest data/manifests/score_cm04.csv --data-root "$DATA_ROOT" \
    --device cpu --batch-size 16 --num-workers 2 --max-seconds 4.0 \
    --partial experiments/_partial_cm04_lora_clean.csv \
    --scores-out experiments/cm04_lora_clean_scores.csv --out experiments/cm04_lora_clean.json

# CM04 shortcut floor
python -m src.data.lowlevel_cue --train data/manifests/score_cm04_fit.csv --train-root "$DATA_ROOT" \
    --test data/manifests/score_cm04_score.csv --test-root "$DATA_ROOT" \
    --out experiments/results/lowlevel_cue_check_cm04_raw.json
```

CM04 audio: private Kaggle dataset `saikrishnareddy9/cm04-tortoise-clips-only`.
CPU throughput was about 6 clips/s; all six runs took about 55 minutes.

## CM04 shortcut floor, in brief

22.21% EER (AUC 0.8166), fit on 8 speakers and scored on the other 7. It fails the
near-chance band like CM01 and CM02, but on **different features**: the top
coefficients are `clipping_ratio` (+3.26), `hf_energy_ratio` (+1.73) and
`peak_amplitude` (+1.58) — level cues, where CM02's tell was spectral
(zero-crossing rate). Each generator leaves its own cheap fingerprint.
