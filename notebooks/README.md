# Notebooks

Exploration and environment checks only. **No pipeline logic lives here** — anything
that produces a paper number lives in `src/` behind a config (repo rule 1), so
results stay reproducible.

| Notebook | Purpose | Week |
|----------|---------|------|
| `env_check.ipynb` | Verify GPU + load wav2vec2-base / XLSR-53 / WavLM-base + init W&B | 1 |
| `eda_corpora.ipynb` | Durations, sample rates, speaker counts, label balance | 2+ |
| `spoof_quality_audit.ipynb` | Manual listening audit of generated spoofs | 3 |
| `error_analysis.ipynb` | Which utterances fool the detector / false alarms | 6+ |

## Running `env_check.ipynb`

1. Enable a GPU (Kaggle: Settings → Accelerator → GPU; Colab: Runtime → GPU).
2. Provide secrets **outside** the notebook — Kaggle *Secrets* or Colab *userdata*
   (or a local `.env`): `WANDB_API_KEY`, optionally `HF_TOKEN`. Never hardcode keys.
3. Run all cells. Success = each of the three encoders prints an `OK` line with a
   `last_hidden_state` shape, and (if `WANDB_API_KEY` is set) a W&B run appears.
4. Log the outcome (GPU model, VRAM, any load failure) in this week's log book.
