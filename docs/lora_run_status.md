# Stage-3 LoRA run status — college PC (RTX 4500 Ada)

**Last updated:** 2026-08-25, ~15:45 IST
**Machine:** college PC, Windows 11, NVIDIA RTX 4500 Ada (24 GB), driver 595.95
**Repo:** `C:\Users\Admin\codemix-deepfake-detection`
**Data root:** `D:\dfdata`
**Companion doc:** [college_pc_setup.md](college_pc_setup.md) — that file is the recipe; this file tracks *where we are* in it.

---

## TL;DR — the run is FINISHED

**Steps 1–8 are ALL DONE.** Training ran clean, both evaluations are in, and the
gap-closure result is written up in [results/gap_closure_v1.md](results/gap_closure_v1.md).

| Column | Stage-1 baseline | + LoRA adapter |
|---|---|---|
| Held-out eval pool (15 spk, 3,966 clips) | 53.71% EER | **1.34%** EER (0.9997 AUC) |
| Published gap column (25 spk, 4,434 clips) | 44.65% EER | **1.40%** EER (0.9992 AUC) |

Repeated over seeds 1234 / 2025 / 7: **1.45% ± 0.10pp** and **1.47% ± 0.26pp**.

**Read [results/channel_matched_v1.md](results/channel_matched_v1.md) before quoting
the 1.34%.** Under G.711 telephony the clean-trained adapter falls to 38.58% EER; a
channel-matched adapter reaches 3.89%. The gap closes over a phone line only if the
adapter is trained for one.

**Still open — W7-T2, blocked on data.** Whether the adapter costs English
performance is unmeasured: ASVspoof 2019 LA is not on this machine. See
[results/gap_closure_v1.md](results/gap_closure_v1.md#the-open-question-this-does-not-answer-what-did-english-cost).

Training took **2.2 minutes**, not the 20–40 min estimated below — see
"num_workers was the bottleneck". Nothing here needs re-running.

> **The "53.3% EER baseline" this file used to tell you to compare against was a
> misattribution.** [results/gap_matrix_v1.md](results/gap_matrix_v1.md) reports the
> code-mixed column as **44.65% EER with AUC 0.533**; the 0.533 is the AUC. That
> 44.65% is also the wrong "before" regardless, because it was measured on a
> different set (4,434 clips, 25 *train*-pool speakers). The 53.71% above was
> measured by running the unadapted checkpoint over *this* eval manifest, which is
> the only comparison where the adapter is the sole difference.

---

## Environment — already done, do not repeat

| Item | State |
|---|---|
| Git hooks | installed (`core.hooksPath=githooks`) |
| `.venv` | Python 3.11.7, all `requirements.txt` installed |
| **torch** | **2.8.0+cu128** — CUDA verified working |
| torchaudio | 2.8.0+cu128 |
| `DATA_ROOT` | `D:\dfdata` (set via `setx`) |
| `DFD_DEVICE` | `cuda` (set via `setx`) |
| Windows power plan | High performance |
| GPU power limit | 210 W = already the card's max, nothing to raise |
| Test suite | green (`python -m pytest tests/ -q`) |

### Gotcha that cost time — do not hit it again

`pip install -r requirements.txt` installs **`torch 2.8.0+cpu`** on Windows (the default PyPI wheel is CPU-only). `torch.cuda.is_available()` was `False` and the GPU would never have been touched. The fix, already applied:

```powershell
.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu128 `
  torch==2.8.0+cu128 torchaudio==2.8.0+cu128 torchvision==0.23.0+cu128
```

The `+cu128` suffix is **required**. Without it pip sees `2.8.0` already installed and silently does nothing — it exits 0, which looks like success.

Verify any time:

```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expect: 2.8.0+cu128 True
```

---

## Data staged (Step 3) — done

Copied from `E:\college_pc_transfer`:

| Source | Destination | Contents |
|---|---|---|
| `best.pt` | `checkpoints\baseline\best.pt` | 362 MB Stage-1 checkpoint |
| `colab_bundle\` | `D:\dfdata\colab_bundle\` | 4,434 clips |
| `raw\mucs2021\train\` | `D:\dfdata\raw\mucs2021\train\` | 25 reference wavs |
| `tts_cache\` | `%LOCALAPPDATA%\tts\...xtts_v2\` | 1.9 GB, `hash.md5` present |
| `manifests\` | `data\manifests\` | `clip_index.csv`, `speaker_pools.csv` |
| `.env.git` | repo root | git identity / PAT |

**`.env` was NOT on the USB** (only `.env.git`). It holds WANDB/HF tokens. Not needed — `configs/train_lora_codemix.yaml` sets `wandb_mode: offline`.

### clip_index.csv path rewrite — already applied

`clip_index.csv` shipped with hardcoded **`C:\dfdata\...`** paths from the laptop. `pool_jobs` crashed on the first wav with `soundfile.LibsndfileError: System error`. All 56,143 `wav_path` rows were rewritten `C:\dfdata` → `D:\dfdata`.

- Backup: `data\manifests\clip_index.csv.bak`
- The file is gitignored (`.gitignore:44`), so this is local-only and will **not** travel back in a commit.
- Verified after: all 10 adaptation + 15 eval reference wavs resolve, 0 missing. (25 train-pool wavs do not resolve — correct, this stage does not need them.)

**If you ever re-copy `clip_index.csv` from the USB, redo this rewrite.**

---

## Ethics gate — OPEN

```
ethics gate OPEN: signed note found (mentor_signoff_2026-08-12.pdf)
```

`docs\ethics\mentor_signoff_2026-08-12.pdf` is in place. It is gitignored (`docs/ethics/*.pdf`), so it lives only in this working tree — **do not delete it**, and it will not survive a fresh clone.

---

## Step 4 — spoof generation

Rate observed on this GPU: **~30–33 clips/min** (~1.8–2.0 s/clip).

### 4a. Adaptation pool — DONE (1,600 / 1,600)

```
total 1600, xtts_v2, hi, pool=adaptation, 10 speakers, 160/speaker exactly
-> D:\dfdata\generated\xtts_v2_adaptation\outputs
```

### 4b. Eval pool — DONE (2,400 / 2,400)

```
total 2400, xtts_v2, hi, pool=eval, 15 speakers, 160/speaker exactly
-> D:\dfdata\generated\xtts_v2_eval\outputs
```

Both pools are complete. Nothing to re-run here. Spot-check any time:

```powershell
(Get-ChildItem D:\dfdata\generated\xtts_v2_adaptation\outputs\*.wav).Count   # 1600
(Get-ChildItem D:\dfdata\generated\xtts_v2_eval\outputs\*.wav).Count         # 2400
```

Generation is resumable if it is ever interrupted — re-running the same command
counts what is on disk and makes up only the difference.

---

## Step 5 — LoRA manifests

### 5a. Adaptation manifests — DONE and verified

| Split | bonafide | spoof | rows | unique clips | speakers |
|---|---|---|---|---|---|
| train | 874 | 1,280 | 2,154 | 2,154 | 8 |
| dev | 186 | 320 | 506 | 506 | 2 |

Files: `data\manifests\codemix_adapt_train.csv`, `codemix_adapt_dev.csv` (plus the `*_clean.csv` intermediates). Audio bundle: `D:\dfdata\lora_bundle\clips\` (2,660 clips).

Verified: no train/dev speaker overlap; every clip resolves; class durations comparable (bonafide mean 5.87 s vs spoof 6.11 s — no trivial length cue).

### 5b. Eval manifest — DONE and verified

| Split | bonafide | spoof | rows | unique clips | speakers |
|---|---|---|---|---|---|
| eval | 1,566 | 2,400 | 3,966 | 3,966 | 15 |

File: `data\manifests\codemix_eval.csv`. Audio went into the same bundle,
`D:\dfdata\lora_bundle\clips\`, which now holds **6,626** clips total
(2,660 adaptation + 3,966 eval).

`rows == unique` for both classes and no `WARNING: N duplicate clip(s)` line —
that is the check that Bug 1 and Bug 2 below are still fixed. If a future
rebuild reports bonafide as ~15 rows, those fixes have been lost.

### Speaker firewall — verified clean

```
adaptation speakers: 10   eval speakers: 15
FIREWALL overlap (must be empty): set()
```

The adaptation and eval pools share no speakers, so the gap-closure number
comes from genuinely held-out voices. Re-check any time:

```powershell
.venv\Scripts\python.exe -c "import pandas as pd; tr=pd.read_csv('data/manifests/codemix_adapt_train.csv'); dv=pd.read_csv('data/manifests/codemix_adapt_dev.csv'); ev=pd.read_csv('data/manifests/codemix_eval.csv'); a=set(tr.speaker.astype(str))|set(dv.speaker.astype(str)); print('overlap:', a & set(ev.speaker.astype(str)))"
```

---

## Step 6 — LoRA training — DONE

```powershell
$env:DATA_ROOT = "D:\dfdata"
.venv\Scripts\python.exe -m pytest tests\test_splits.py tests\test_lora.py -q   # must be green

.venv\Scripts\python.exe -m src.training.train `
  --config configs\train_lora_codemix.yaml `
  --device cuda --data-root "D:\dfdata\lora_bundle"
```

### `--data-root` is REQUIRED and is not in the setup doc

Portable manifests store paths as `${DATA_ROOT}/clips/<name>.wav`. The bundle root **is** `$DATA_ROOT` at train time (same convention as `colab_bundle`). Our bundle is at `D:\dfdata\lora_bundle`, so `--data-root "D:\dfdata\lora_bundle"` must be passed to **both** `train` and `evaluate`. Without it the loader looks in `D:\dfdata\clips\` and every file is missing.

### Outcome

```
training: 2154 train / 506 dev clips, batch 8, 270 steps/epoch x 5 epochs
epoch 1: dev EER 0.0275   score std 0.4621   (0.7 min)
epoch 2: dev EER 0.0159   score std 0.4748   (1.1 min)
epoch 3: dev EER 0.0159   score std 0.4762   (1.5 min)
epoch 4: dev EER 0.0101   score std 0.4778   (1.8 min)
epoch 5: dev EER 0.0101   score std 0.4781   (2.2 min)
done: best dev EER = 0.0101
```

Score std stayed near 0.48 throughout, three orders of magnitude above the
`COLLAPSE_STD` tripwire — the model never stopped depending on its input.

### num_workers was the bottleneck, not the GPU

The 20–40 min estimate assumed `num_workers: 0`, which decodes every wav on the
main process while the card idles. Raised to **4** in `configs/train_lora_codemix.yaml`
and the run finished in **2.2 min at ~10.4 steps/s**.

This is numerically neutral, which is why it was safe to change and `batch_size`
was not: the train crop RNG is seeded per-index (`dataset._crop`) and the shuffle
sampler runs in the parent process, so the workers see exactly the same batches.

### Earlier smoke run

A `--smoke` run completed successfully end-to-end on the GPU:

```
device=cuda (NVIDIA RTX 4500 Ada Generation, 24.0 GiB)
initialised from checkpoints/baseline/best.pt
lora: r=8 alpha=16.0 on 48 layers ('q_proj','k_proj','v_proj','out_proj');
      trainable 1,082,371/95,454,083 (1.13%)
```

Its checkpoints were deleted before the real run, so it started clean and
`--resume` was correctly not passed.

### Config, unchanged on purpose

`configs\train_lora_codemix.yaml`: `lr 1e-4`, `batch_size 8`, `max_seconds 4.0`, `epochs 5`, `class_weighted: true`, `amp: true`, `wandb_mode: offline`.

`batch_size` was deliberately **left at 8** rather than raised to fill the 24 GB. Changing it without retuning `lr` would move the gap-closure number being compared against the 53.3% EER baseline. Raise it only as a conscious experiment, not for speed.

Progress prints every 25 steps; `last.pt` is written every epoch, so `--resume` picks up an interrupted run.

---

## Step 7 — evaluate — DONE

Both rows below are the same `data\manifests\codemix_eval.csv` — 1,566 bonafide +
2,400 spoof over the 15 eval-pool speakers. Only the adapter differs.

```powershell
# adapted -> experiments\lora_codemix_eval.json      EER 1.34%   AUC 0.9997
# baseline -> experimentsaseline_codemix_eval.json  EER 53.71%  AUC 0.459
.venv\Scripts\python.exe -m src.training.evaluate `
  --checkpoint checkpoints\lora_codemixest.pt `
  --manifest data\manifests\codemix_eval.csv `
  --device cuda --batch-size 32 --num-workers 4 `
  --data-root "D:\dfdata\lora_bundle" `
  --out experiments\lora_codemix_eval.json
```

The baseline row is an addition to the original plan. Without it the only "before"
available was the gap matrix's 44.65%, measured on different speakers — see the
warning in the TL;DR.

### Bug 3 — `evaluate.py` could not load an adapted checkpoint at all

This blocked Step 7 outright and is **not** anticipated anywhere in the setup doc.
`apply_lora` renames `...q_proj.weight` to `...q_proj.base.weight` and adds
`lora_A`/`lora_B`, so a Stage-3 `best.pt` does not fit the unadapted module tree
`score_manifest` builds — `load_state_dict` raises on every adapted key.

**Fix:** `src/training/evaluate.py` now rebuilds the LoRA wrapping before loading.
Targets are recovered from the checkpoint's own key names and the rank from
`lora_A`'s shape, so a non-default target set still reloads; `alpha` cannot be
recovered from shapes, so the saved config is authoritative. A Stage-1 checkpoint
is left untouched. Three tests in `tests/test_lora.py` cover it.

`--num-workers` was also added to the eval CLI (default 0, so nothing changes for
existing callers).

---

## Two source bugs found and fixed — NEEDS REVIEW

These are **uncommitted changes to tracked files**. Review before committing. Without them the training run is worthless.

```
 src/data/codemix_manifests.py   | 15 ++++++++++++---
 src/data/portable_bundle.py     |  7 +++++--
 tests/test_codemix_manifests.py | 13 ++++++++++---
```

### Bug 1 — bonafide set collapsed to 8 clips

`bonafide_rows()` did `drop_duplicates("filepath")`. MUCS packs every utterance of a speaker into one 8–10 min wav, so this collapsed **1,060 adaptation utterances down to 10 recordings**. The manifest came out as **8 bonafide vs 1,280 spoof**, with a dev EER computed from **2** real clips — statistically meaningless.

Evidence this is a bug, not intent:

- The module's own docstring names this exact failure: *"a manifest built from `wav_path` alone collapsed 2,217 rows to 25 files"*.
- `docs/results/gap_matrix_v1.md` Method requires **balanced 2,217 / 2,217**; `gap_codemix_clean.csv` on disk is exactly that.
- `portable_bundle.attach_spans` is explicitly built to assign N distinct spans to N rows — dead code if only one row per recording ever reaches it.
- History: `4691e0d` fixed this collapse in `portable_bundle`; `d65d608` then reintroduced it one layer earlier in `codemix_manifests`.

**Fix:** dedupe on `utt_id`, and carry `start_seconds`/`end_seconds` through so each row's true span travels with it.

**Test change — review this one closely.** `test_bonafide_rows_deduplicate_by_filepath` asserted the broken behaviour (`len(rows) == 5`). It was replaced with `test_bonafide_rows_keep_one_row_per_utterance` asserting `len(rows) == 15` plus unique `utt_id` and non-null spans.

### Bug 2 — every spoof row named `nan.wav`

Latent bug, exposed by the Bug 1 fix. In `portable_bundle.clip_name`:

```python
utt = str(row.get("utt_id") or "").strip()   # BROKEN
```

`float('nan')` is **truthy** in Python, so a missing `utt_id` stringified to `"nan"` and all 1,280 spoof rows wrote to the single file `nan.wav`.

**Fix:** explicit `pd.isna()` test instead of relying on truthiness.

After both fixes the full suite is green and the bundle reports `2,154 rows, 2,154 unique clips` with no duplicate warning.

---

## Untracked files produced this session

```
data/manifests/adaptation_generation_jobs.csv
data/manifests/eval_generation_jobs.csv
data/manifests/codemix_adapt_train.csv        codemix_adapt_train_clean.csv
data/manifests/codemix_adapt_dev.csv          codemix_adapt_dev_clean.csv
data/manifests/codemix_eval.csv               codemix_eval_clean.csv
docs/lora_run_status.md                       <- this file
docs/results/gap_closure_v1.md                <- the result; tracked, committed
logs/lora_train.log  logs/lora_eval.log  logs/baseline_eval.log
```

`experiments/*_scores.csv` hold the per-clip scores behind the tables and are
committed alongside the `.json` summaries, following `stage1_eval_scores.csv` and
`gap_codemix_v2_scores.csv`. `logs/` is machine-local and is now gitignored.

Per `GIT_RULES.md` §1, data and checkpoints never enter git. Step 8 of the setup doc commits only `experiments\lora_codemix_eval.json` and `docs\progress.md`.

---

## Known-good verification commands

```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
.venv\Scripts\python.exe -m src.data.ethics_gate
.venv\Scripts\python.exe -m pytest tests\ -q
(Get-ChildItem D:\dfdata\generated\xtts_v2_adaptation\outputs\*.wav).Count   # 1600
(Get-ChildItem D:\dfdata\generated\xtts_v2_eval\outputs\*.wav).Count         # target 2400
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader
```
