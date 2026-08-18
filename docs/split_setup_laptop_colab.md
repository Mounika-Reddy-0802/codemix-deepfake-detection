# Split setup: develop on the laptop, train on Colab

This laptop has no CUDA (`torch 2.10.0+cpu`, Python 3.13). Training runs on a
Colab GPU runtime — or on a teammate's GPU laptop — from the *same* checkout. This
document is the operating procedure for that split.

The rule the code enforces: **nothing branches on which machine it is.** Two
environment variables carry every machine-specific fact.

| Variable | This laptop | Colab / GPU box | Meaning |
|---|---|---|---|
| `DATA_ROOT` | `C:/dfdata` | `/content/data` | where the `raw/ processed/ generated/` tree lives |
| `DFD_DEVICE` | unset (→ `cpu`) | `cuda` | which device to run on; `cuda` fails loudly if no GPU is attached |

---

## 1. What runs where

Everything is in one repo — splitting it would mean two places to fix a bug. What
differs is which entry point you *invoke*.

**Runs on the GPU (Colab / GPU laptop)**

| Path | Why it needs a GPU |
|---|---|
| `src/training/train.py` | S1/S3 fine-tuning of wav2vec2/WavLM |
| `src/training/evaluate.py` | gap-matrix scoring — thousands of forward passes |
| `src/models/` | encoder + pooling + head, loaded onto the device |
| `src/data/spoof_generation.py`, `heldout_tts.py` | XTTS-v2 / Tortoise synthesis |
| `notebooks/pilot_xtts_colab.ipynb` | the W3-T5 pilot driver |
| `configs/train_baseline.yaml`, `train_lora_codemix.yaml` | the run definitions |

**Stays on this laptop (CPU is enough)**

| Path | Why it is fine on CPU |
|---|---|
| `src/data/corpora.py`, `speaker_selection.py`, `speaker_pools.py`, `pilot_jobs.py` | pandas over metadata; no audio decoded |
| `src/data/preprocess.py`, `channel_sim.py`, `channel_qa.py` | numpy DSP; slow but not GPU work |
| `src/data/build_manifests.py`, `checksums.py`, `quarantine.py`, `ethics_gate.py` | pure logic and hashing |
| `src/inference/predict.py` | one clip at a time — CPU latency is the number the paper reports |
| `live_call/`, `demo/` | FastAPI + WebRTC + Gradio; CPU inference by design |
| `tests/`, `docs/`, `paper/`, `report/` | everything else |

**Never syncs, in either direction:** `data/**` (audio), `checkpoints/**`,
`.env`, `.env.git`, `docs/ethics/*.pdf`, `wandb/`. All are already in
[.gitignore](.gitignore).

---

## 2. Code sync — GitHub, not a zip

The repo is already set up for exactly this: `.gitignore` line 22 excludes
`data/**` and line 38 excludes `checkpoints/**`, while **manifests stay tracked**
(`!data/manifests/*.csv`) because a frozen split has to be reviewable. So `git
pull` on Colab brings the code *and* the frozen speaker pools, and brings no audio.

Push from here (owner identity per `CLAUDE.md` §1):

```bash
set -a; . ./.env.git; set +a
git -c user.name="$M_GIT_NAME" -c user.email="$M_GIT_EMAIL" commit -m "..."
git push "https://$M_GH_USER:$M_GH_PAT@$REPO_URL" dev
```

Pull on Colab — see §3. Pull on a teammate's GPU laptop:

```bash
git clone https://github.com/Mounika-Reddy-0802/codemix-deepfake-detection.git
cd codemix-deepfake-detection
sh scripts/install_hooks.sh
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # pin torch to that box's CUDA build
export DATA_ROOT=/path/to/their/dfdata
export DFD_DEVICE=cuda
python -m src.utils.device               # must print a GPU name
```

Results come back the same way: `experiments/`, `docs/`, `configs/` are tracked,
so committing them on the GPU side and pulling here is the whole round trip.
Weights are the exception — see §5.

---

## 3. Running training on Colab

`notebooks/pilot_xtts_colab.ipynb` already models the pattern (mount Drive → clone
the repo → open the ethics gate → call into `src/`). A training notebook is the
same five steps.

**Upload to Drive once** (not to git):

```
MyDrive/capstone/data/raw/asvspoof2019_LA/...   <- the S1 training corpus
MyDrive/capstone/data/raw/mucs2021/...          <- only if training S3
MyDrive/capstone/ethics/mentor_signoff_*.pdf    <- gitignored; generation needs it
```

**In the Colab cell:**

```python
# 1. GPU runtime: Runtime -> Change runtime type -> T4 GPU
from google.colab import drive
drive.mount("/content/drive")

# 2. Code from git (no data comes with it)
!git clone --branch dev https://<user>:<PAT>@github.com/Mounika-Reddy-0802/codemix-deepfake-detection.git /content/repo
%cd /content/repo
!pip install -q -r requirements.txt

# 3. The two variables that make this machine "the GPU machine"
import os
os.environ["DATA_ROOT"]  = "/content/drive/MyDrive/capstone/data"
os.environ["DFD_DEVICE"] = "cuda"      # raises if no GPU is attached
!python -m src.utils.device            # confirm before spending an hour

# 4. Smoke first, always. 1% subset, 20 steps, W&B offline.
!python -m src.training.train --config configs/train_baseline.yaml --smoke

# 5. The real run, checkpointing into Drive so a disconnect is not fatal
!python -m src.training.train --config configs/train_baseline.yaml
```

**No path edits are needed.** `src/utils/paths.py` rebases every manifest clip
path onto `$DATA_ROOT` by its first layout anchor (`raw`/`processed`/`generated`/
`manifests`), so a manifest written here as
`C:/dfdata/raw/mucs2021/train/x.wav` loads on Colab as
`/content/drive/MyDrive/capstone/data/raw/mucs2021/train/x.wav`. New manifests
should store `${DATA_ROOT}/raw/...` directly (`paths.portable`).

**Two Colab-specific settings worth overriding in the config:**

- `out_dir:` point it at Drive (`/content/drive/MyDrive/capstone/checkpoints/baseline`)
  — a session can be reclaimed at any moment, and `/content` is not persistent.
- `num_workers:` Colab gives 2 vCPUs; 2 is right there, 0 is right on this laptop.

Copying `data/raw/asvspoof2019_LA` from Drive to local `/content` disk before
training is usually worth it — Drive I/O is the bottleneck on small-clip datasets.

---

## 4. Device handling in the code

Already done, in `src/utils/device.py`. Every entry point calls `resolve_device()`
rather than testing `torch.cuda.is_available()` itself:

```python
from src.utils.device import resolve_device, amp_enabled, make_grad_scaler

device = resolve_device(cfg.device)      # arg -> $DFD_DEVICE -> auto (cuda/mps/cpu)
use_amp = amp_enabled(device, cfg.amp)   # AMP is a CUDA feature; off on CPU
scaler  = make_grad_scaler(device, use_amp)
```

Design points that matter in practice:

- **Auto-detect falls back quietly; an explicit request fails loudly.** With
  `DFD_DEVICE=cuda` set on Colab, a runtime that lost its GPU raises
  `DeviceUnavailableError` in the first second instead of silently training on
  CPU for the rest of the session.
- **`amp: true` in the config is honoured only on CUDA.** On CPU, AMP costs
  accuracy and buys no speed, so it disables itself; the config need not change
  between machines.
- **`torch.cuda.amp.GradScaler` (deprecated from torch 2.4) is gone**, replaced by
  `torch.amp.GradScaler(family, ...)` with a fallback for older torch.
- **`pin_memory` follows the device**; `persistent_workers` is only set when there
  are workers, which keeps Windows' spawn cost out of a laptop smoke run.

Force either mode anywhere:

```bash
python -m src.training.train --config configs/train_baseline.yaml --device cpu
DFD_DEVICE=cpu python -m src.training.train --config configs/train_baseline.yaml
```

---

## 5. Bringing the trained checkpoint back

`checkpoints/**` is gitignored — a wav2vec2-base checkpoint is ~360 MB and must
never enter git history. Move it out of band and record its hash.

**On Colab, after training:**

```python
import hashlib, shutil
src = "/content/drive/MyDrive/capstone/checkpoints/baseline/best.pt"
print(hashlib.sha256(open(src, "rb").read()).hexdigest())
```

Record that hash in `docs/progress.md` (project rule §5 — data and checkpoints are
referenced by link + hash, never committed).

**On this laptop:**

```powershell
# Download best.pt from Drive, then:
mkdir -Force checkpoints\baseline
Move-Item ~\Downloads\best.pt checkpoints\baseline\best.pt
```

**What to change to use it — the short answer is nothing, if you use the default
path.** The checkpoint written by `src/training/train.py` carries its own config:

```python
torch.save({"model": ..., "config": cfg.__dict__, "metrics": ...}, out_dir / "best.pt")
```

so the inference side reads the encoder name and the EER threshold back out of the
file rather than being told. The three values you may need to set:

| Where | Value | Default |
|---|---|---|
| `checkpoints/baseline/best.pt` | where you put the file | `TrainConfig.out_dir` + `best.pt` |
| `DETECTOR_CHECKPOINT` env var | what the FastAPI server serves | unset → `/predict` returns 503 |
| `--threshold` / `configs/threshold.yaml` | operating point | the checkpoint's stored EER threshold |

> **Blocker, be aware:** `src/inference/predict.py` is still a placeholder
> docstring on `dev`. A complete implementation — `load_detector`,
> `predict_file`, `Prediction`, `threshold_from_checkpoint`, plus
> `live_call/server.py` with `/health` + `/predict` and `src/training/evaluate.py`
> — exists on the local branch `parked/v1-week4-superseded` and is the W5-T1/W5-T5
> deliverable. Restore it when that week opens:
>
> ```bash
> git checkout dev
> git checkout parked/v1-week4-superseded -- \
>   src/inference/predict.py src/training/evaluate.py src/training/schedules.py \
>   live_call/server.py tests/test_predict.py tests/test_evaluate.py \
>   tests/test_schedules.py tests/test_server.py
> ```
>
> Until then there is no CPU inference path to plug a checkpoint into.

---

## 6. Compute-plan note (carried from `Works updates.md` §8)

The v2 plan sizes Weeks 5–9 around *3 Kaggle accounts × 30 GPU-hours/week*, with
S1 and S3 training in parallel on separate accounts (W5-T3). Colab has no
comparable weekly quota and reclaims idle sessions, so that parallelism does not
hold. Decide before launching S1/S3: buy Colab Pro, keep Kaggle for the long runs,
or serialise S1 then S3.
