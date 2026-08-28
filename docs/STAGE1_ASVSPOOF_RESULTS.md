# Stage-1 ASVspoof 2019 LA Results

**First completed Stage-1 baseline measurement.** The English-only Stage-1
detector was trained on ASVspoof 2019 LA and evaluated on the official
ASVspoof 2019 LA evaluation set.

Checkpoint: `checkpoints/baseline/best.pt`  
Encoder: `wav2vec2-base`  
Maximum clip duration: 4.0 s  
Evaluation manifest: `data/manifests/asvspoof_eval.csv`

## Result

| Metric | Result |
|---|---:|
| **EER** | **0.5843%** |
| EER 95% CI | 0.4527% – 0.7059% |
| **AUC** | **0.999752** |
| AUC 95% CI | 0.999668 – 0.999819 |
| **F1** | **0.9723** |
| Threshold | 0.997581 |
| Evaluation clips | **71,237** |
| Bonafide clips | 7,355 |
| Spoof clips | 63,882 |

The pooled evaluation result shows very strong discrimination on the
ASVspoof 2019 LA evaluation set, with an EER below 1% and an AUC above 0.999.

## Per-attack result

| Attack | Spoof clips | EER |
|---|---:|---:|
| A17 | 4,914 | 0.9948% |
| A18 | 4,914 | 0.7266% |
| A10 | 4,914 | 0.6689% |
| A15 | 4,914 | 0.5874% |
| A19 | 4,914 | 0.4652% |
| A16 | 4,914 | 0.4074% |
| A11 | 4,914 | 0.2614% |
| A07 | 4,914 | 0.1222% |
| A12 | 4,914 | 0.0815% |
| A08 | 4,914 | 0.0407% |
| A09 | 4,914 | 0.0170% |
| A14 | 4,914 | 0.0170% |
| A13 | 4,914 | 0.0000% |

## Interpretation

This establishes the Stage-1 English ASVspoof baseline. The model performs
very strongly on the same benchmark family used for Stage-1 evaluation.

The result should not be interpreted as evidence of code-mixed or
telephony-domain generalisation. Those are separate evaluation conditions
and remain part of the project's gap measurement.

## Platform

Trained and evaluated on a Kaggle GPU notebook, using this repository's Stage-1
training and evaluation pipeline (`src/training/train.py`, `src/training/evaluate.py`)
unchanged. `tests/test_splits.py` (the anti-leakage checklist) passed before the
run started.

**Note for anyone reproducing this locally:** the checkpoint that produced this
result lives only on Kaggle's output for that run. If `checkpoints/baseline/best.pt`
on your machine came from an earlier local run, re-evaluating it will reproduce
*that* run's number, not this one — pull down the Kaggle notebook's output
checkpoint first.

## Reproduction

```bash
python -m src.training.evaluate \
  --checkpoint checkpoints/baseline/best.pt \
  --manifest data/manifests/asvspoof_eval.csv \
  --device cuda \
  --out experiments/stage1_eval_results_kaggle.json
```

Re-training from scratch uses the same config family (`configs/train_stage1_run.yaml`
or the fuller `configs/train_baseline.yaml`) via `src/training/train.py` — see
`README.md`'s reproduce section for the full command.