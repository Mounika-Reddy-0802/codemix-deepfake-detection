# Gap closure v1 — LoRA adaptation on code-mixed Hinglish

**The other half of [gap_matrix_v1.md](gap_matrix_v1.md).** That document measured
how badly the English-trained Stage-1 detector fails on code-mixed speech. This one
measures how much of that failure a 1.13%-parameter LoRA adapter removes.

Adapter: `checkpoints/lora_codemix/best.pt` (r=8, alpha=16, q/k/v/out_proj on all
12 encoder layers, 1,082,371 / 95,454,083 trainable). Trained 5 epochs on the
**adaptation pool only**; best dev EER 1.01%. 2.2 minutes on one RTX 4500 Ada.

## Result

Both rows are the **same 3,966 clips** — `data/manifests/codemix_eval.csv`, 1,566
MUCS bonafide + 2,400 XTTS-v2 spoof over the **15 eval-pool speakers**. The only
difference between the rows is the adapter.

| Model | EER | 95% CI | AUC |
|---|---|---|---|
| Stage-1 baseline, unadapted | **53.71%** | [52.1%, 55.5%] | 0.459 |
| **+ LoRA code-mix adapter** | **1.34%** | [0.80%, 1.66%] | **0.9997** |

**A 40× reduction in EER**, from worse-than-chance to near-solved, by training
1.13% of the parameters for 2.2 minutes on one RTX 4500 Ada.

The confidence intervals do not come close to overlapping, so this is not
sampling noise.

### The same adapter on the column the paper already published

`gap_matrix_v1.md`'s code-mixed row is a *different* set — 4,434 clips over the 25
**train**-pool speakers. Those speakers are also disjoint from the 10 adaptation
speakers (verified), so the adapter can be scored on it directly and the published
row restated without re-deriving anything:

| `data/manifests/gap_codemix_portable.csv` — 2,217 bonafide / 2,217 spoof | EER | AUC |
|---|---|---|
| Stage-1 baseline (as published in `gap_matrix_v1.md`) | 44.65% | 0.533 |
| **+ LoRA code-mix adapter** | **1.40%** | **0.9992** |

Same clips, same speakers, same protocol as the published number — **a 32×
reduction on the paper's own headline row**.

### Seed variance

The configuration was trained three times, changing only `seed`. Nothing was tuned
against either eval set.

| Seed | best dev EER | held-out eval pool | published gap column |
|---|---|---|---|
| 1234 | 1.01% | 1.34% | 1.40% |
| 2025 | 0.58% | 1.46% | 1.26% |
| 7 | 1.01% | 1.54% | 1.76% |
| **mean ± sd** | | **1.45% ± 0.10pp** | **1.47% ± 0.26pp** |

The spread is a fifth of a percentage point against a gap of ~52 points, so seed
choice is not carrying this result. All three runs held score std near 0.48 and
none collapsed.

## The failure mode that closed

`gap_matrix_v1.md` argued the headline EER understates the damage: the detector
was not failing to catch fakes so much as **rejecting real human speech**. That is
the part that had to move, and it did.

| Set | mean P(bonafide) | classified "bonafide" |
|---|---|---|
| MUCS bonafide — baseline | 0.0397 | **2.6%** |
| MUCS bonafide — adapted | **0.9981** | **99.9%** |
| XTTS spoof — baseline | 0.0329 | 1.7% |
| XTTS spoof — adapted | 0.0291 | 2.8% |

The baseline calls **97.4% of genuine Hinglish speech a deepfake** on these
speakers — worse than the 87.9% the gap matrix measured on the train pool, and the
reason its AUC sits *below* 0.5: both classes are crushed against the spoof end,
with the real clips slightly further down. The adapter lifts the bonafide class to
0.998 mean while leaving the spoof class where it was. It learned what real Hindi
speech sounds like, which is precisely what Stage-1 never saw.

## Why this number is honest

- **Speaker firewall.** Adaptation used the 10-speaker adaptation pool; this eval
  uses the 15-speaker eval pool. Verified disjoint (`FIREWALL overlap: set()`), so
  the result is generalisation to unheard voices, not memorisation.
- **Same manifest for both rows.** The 44.65% figure in `gap_matrix_v1.md` was
  measured on a *different* column (4,434 clips, 25 train-pool speakers) and is
  therefore not the right "before" for this "after". The baseline row above was
  produced by running the unadapted Stage-1 checkpoint over this exact manifest.
- **Zero-init adapters.** `lora_B` starts at zero, so the adapted model begins
  numerically identical to Stage-1 (`tests/test_lora.py::test_lora_starts_as_a_no_op`).
  Any delta is the adaptation, not a different starting point.
- **No label-adjacent shortcut.** Class durations are comparable (bonafide 5.87 s
  vs spoof 6.11 s mean), so clip length is not separating the classes.

## What this does NOT establish

**It does not survive a phone line as trained.** Measured under G.711 at 20 dB, this
adapter goes from 1.34% to **38.58% EER** — it calls 98.9% of spoofs genuine once the
codec removes the band its spoof cue lived in. Retraining channel-matched recovers it
(3.89%), so the gap *is* closable under telephony, but the number on this page is a
clean-audio number and must not be quoted as a deployment result. Full 2×2 and the
mechanism: [channel_matched_v1.md](channel_matched_v1.md).


**The spoof side is still weak, and that caps how much this proves.** Per
`gap_matrix_v1.md`, the XTTS corpus rated 1.5/5 sounds-human with a compressed
pitch range — these are easy fakes, and the eval spoof set is one attack family
(`xtts_v2`) generated by the same tool the adapter trained against. A 1.34% EER
means the adapter separates *this* generator from real Hindi speech. It is not
evidence of robustness to an unseen attack family; RVC and the held-out Tortoise
split are what would test that.

**Language and recording domain are still confounded**, exactly as in v1. The
adapter closed a gap that is part language shift and part MUCS-lecture domain
shift, and this result cannot apportion it between the two.

**One configuration, three seeds.** `batch_size` and `lr` were fixed in advance
(`configs/train_lora_codemix.yaml`) and never tuned against either eval set, which
protects the number from selection. The seed table above bounds run-to-run
variance, but no sweep over `r`, `lr` or the target set was run, so nothing here
says this configuration is the best available — only that it is stable.

## The open question this does NOT answer: what did English cost?

AffectDF's Table 13 is the warning — AASIST retrained on their data scores **44.52%
EER back on ASVspoof 2019**, having started at 0.83%. Domain adaptation bought them
the new domain and destroyed the old one. **W7-T2 exists to test whether Stage-3
does the same thing to us**, and it is not answered here.

Stage-1 scores **0.87% EER on ASVspoof 2019 eval** (`gap_matrix_v1.md`). The adapted
model's number on that same 71,237-clip set is the missing cell, and it is the one a
reviewer will ask for first: a detector that solves Hinglish by forgetting English is
not a mitigation, it is a trade.

**It is blocked on data, not on compute.** ASVspoof 2019 LA is not staged on this
machine — `D:\dfdata` holds only `colab_bundle`, `lora_bundle`, `generated` and
`raw/mucs2021`. The corpus lives on the GPU laptop. Once it is present:

```powershell
.venv\Scripts\python.exe -m src.training.evaluate `
  --checkpoint checkpoints\lora_codemix\best.pt `
  --manifest data\manifests\asvspoof_eval.csv `
  --device cuda --batch-size 32 --num-workers 4 `
  --out experiments\lora_asvspoof_eval.json
```

There is reason for cautious optimism that the damage is small — LoRA freezes the
base encoder and only 1.13% of the parameters moved — but that is an argument, not a
measurement, and it must not be reported as one.

## Reproduce

```powershell
$env:DATA_ROOT = "D:\dfdata"

.venv\Scripts\python.exe -m src.training.train `
  --config configs\train_lora_codemix.yaml `
  --device cuda --data-root "D:\dfdata\lora_bundle"

# adapted
.venv\Scripts\python.exe -m src.training.evaluate `
  --checkpoint checkpoints\lora_codemix\best.pt `
  --manifest data\manifests\codemix_eval.csv `
  --device cuda --batch-size 32 --num-workers 4 `
  --data-root "D:\dfdata\lora_bundle" `
  --out experiments\lora_codemix_eval.json

# baseline, same manifest
.venv\Scripts\python.exe -m src.training.evaluate `
  --checkpoint checkpoints\baseline\best.pt `
  --manifest data\manifests\codemix_eval.csv `
  --device cuda --batch-size 32 --num-workers 4 `
  --data-root "D:\dfdata\lora_bundle" `
  --out experiments\baseline_codemix_eval.json
```

`--data-root` is required and is *not* the same as `$DATA_ROOT`: portable manifests
store `${DATA_ROOT}/clips/<name>.wav`, and the bundle root is `D:\dfdata\lora_bundle`.
Without it every clip resolves to `D:\dfdata\clips\` and is missing.
