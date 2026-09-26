# Week 11 — Mounika — interpretability probes and S1 measured on the normalised sets

Two things were unfinished going into the paper: the headline gap still rested on
S1 numbers measured *before* level normalisation, and we had no answer to "what is
the detector actually responding to?" beyond the shortcut-floor argument. Both are
now measured, and one exploratory finding had to be withdrawn.

## What I did, and why it is built this way

### S1 scored on all six normalised code-mixed sets (P-032)

P-030 left S1's code-mixed cells unmeasured, so the comparison table carried
pre-normalisation figures for one system and normalised figures for the other two.
S1 — `checkpoints/baseline/best.pt`, the exact checkpoint every adapter starts from,
verified by comparing frozen encoder weights against the deployed adapter — was
scored on all six normalised sets through `src.training.evaluate`.

`src.reporting.systems` now reads `results/s1_norm/`. When that directory is
incomplete the cells are **omitted and reported as unmeasured**, never backfilled
with the retired numbers; a test covers that.

### Perturbation probes, as a module rather than a scratch script (P-033, P-034)

`src/inference/interpretability.py` behind `configs/interpretability.yaml`, so the
probes are reproducible and a paper number never comes out of a notebook. Three
probes: chunk shuffling, band-limiting, and phase scrambling at fixed magnitude
spectrum. Run for S1 and S2 on byte-identical audio through two extra configs.

## Numbers

S1 on the normalised sets, against the same per-set floors S2 and S3 are judged by:

| Condition | S1 EER | Floor | Genuine clips called fake |
|---|---|---|---|
| Clean, code-mixed | 41.67–48.21% | 5.17–31.21% | 94.8% |
| Telephone channel | 44.15–58.42% | 10.01–25.79% | — |

AUC is 0.546–0.597 clean and **0.381–0.561 over the channel**: below 0.5 means the
clones look more genuine to S1 than the real speakers do. On English the same
checkpoint gives 0.90% EER (AUC 0.9940) locally, 0.85% on a T4. The paired contrast
at threshold 0.5: **0.37% of genuine English rejected against 94.8% of genuine
Hindi–English — a 256× rise in the false-alarm rate on real speech.**

Probes:

| Probe | S1 | S2 (deployed) |
|---|---|---|
| Layer at which the decision forms | AUC 0.95 at conv output, 1.00 by layer 3 | same |
| Keep 0–1 kHz only, genuine clip | 0.999 → 0.080 | 0.999 → 0.964 |
| Phase scrambled, fixed magnitude | genuine → 0.000 | genuine → 0.000 |
| Real English through the Hinglish adapter | — | 1.000 |

Both models decide in the first three layers, before linguistic representation
exists. What channel-matched adaptation moved is the **band**: S1's evidence for
real speech sits largely above 1 kHz, S2's in the pitch-harmonic band that survives
a narrowband codec — which is why the clean-trained adapter dies under G.711 (P-024).

## The finding I withdrew

An exploratory script reported that time-stretching a clone raised its score from
0.000 to about 0.95 — a one-line evasion of the deployed model, and it would have
gone into the paper. It was a bug. The script shuffled 5 ms chunks with
`rng.shuffle` on a **view** of the clip (`y[:k*n].reshape(k, n)`), so each row
overwrote the audio and every later row ran on wreckage; stretching that scores
about 0.95 for any clip, real or cloned. Rebuilt properly the same probe gives
0.101 and 0.000: the clones stay detected.

Consequences: no evasion claim enters the paper or the limitations section, the
numbers to quote are `experiments/results/interpretability.json`, and
`tests/test_interpretability.py` fails if any perturbation mutates its input.

## Honest limitations

- **S3 could not be probed** — no checkpoint on the demo machine, so the layer and
  band results cover S1 and S2 only.
- **A claim from the first probe run was corrected.** "A 100 ms shuffle leaves the
  verdict unchanged" holds in the deployment condition only; on clean studio English
  both models drop the genuine verdict at 100 ms, because there is no codec or room
  noise to mask the discontinuities. Both survive 500 ms. The language conclusion
  now rests on the RVC pair (identical words, opposite verdicts) and on real English
  scoring 1.000 through the Hinglish adapter.
- **Five clips is a small probe set.** The band and layer results are consistent
  across them but are not a statistical claim.
- The S1 English figure in `docs/STAGE1_ASVSPOOF_RESULTS.md` (0.5843%) belongs to a
  *different* Kaggle retrain written to the same path and is **not** the S1 the
  adapters descend from. The paper reports 0.85%. The stale file is annotated, not
  deleted.

## How to run it

```bash
python -m src.training.evaluate --checkpoint checkpoints/baseline/best.pt \
  --manifest data/manifests/asvspoof_eval.csv --device cuda \
  --out results/s1_norm/stage1__asvspoof_eval.json
python -m src.inference.interpretability --config configs/interpretability.yaml
python -m src.reporting.systems
pytest tests/test_interpretability.py tests/test_systems.py
```

## What's next

- Probe S3 once a checkpoint is on a machine with a GPU.
- The paper's limitations section takes the corrected shuffle result, not the
  withdrawn one.
