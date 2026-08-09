# Week 3 — spoof generation at scale + training scaffolding

**Goal:** stand up the project's core data contribution (the code-mixed spoof
corpus) and the Stage-1 training loop, so Week 4 can freeze the dataset and start
baseline training.

## What shipped

| Owner | Area | Deliverable |
|-------|------|-------------|
| L | generation at scale | `build_clone_jobs` (transcript × speaker-ref × pool → jobs), `generation_stats` (audit report), `configs/generation.yaml` |
| SK | held-out attack | `heldout_tts.py` — Tortoise driver for the unseen-attack split, tool+pool firewalled to eval-only |
| M | training scaffolding | `SSLEncoder`, `AttentiveStatsPooling`, `MLPHead`, `Detector`, AMP training loop with checkpointing + W&B, `configs/train_baseline.yaml`, `seed` util |

## Verify

```bash
ruff check . && ruff format --check . && pytest      # green
python -m src.training.train --config configs/train_baseline.yaml --smoke
```

`--smoke` runs 1% of the train manifest for ~20 steps to confirm the loop learns
before spending GPU hours.

## Model architecture

```
waveform (16 kHz) → SSLEncoder (wav2vec2/WavLM/XLSR, CNN frozen)
                  → AttentiveStatsPooling (mean ⊕ std)
                  → MLPHead (2-layer) → logits {bonafide, spoof}
```

## Honest limitations

- **Not yet run on real data/GPU here.** The audio- and model-heavy paths
  (XTTS-v2, Tortoise, wav2vec2 forward/backward) are validated by lint + compile +
  logic unit tests; the pure-numpy/pandas logic (clone-job assembly, generation
  stats, pooling shapes via torch when present, split checks) is unit-tested. The
  actual generation, the smoke-train, and the 50-clip listening audit are human
  steps (see `Works updates.md`), pending the ethics sign-off + downloads.
- **Generation counts are targets, not yet realised** (2,000 XTTS + ~500 Tortoise).
- Tortoise clones are **eval-only**; the firewall is enforced in code and by
  `tests/test_splits.py`, but the manifests themselves are built in Week 4.

## Decisions (see `problems_and_decisions.md`)

- P-003 held-out tool lives in a separate module (`heldout_tts.py`) so L's and SK's
  Week-3 edits never touch the same file.
- P-004 heavy imports kept lazy so CI stays light and green.
