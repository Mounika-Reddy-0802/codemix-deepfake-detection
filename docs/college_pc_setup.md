# Running Stage-3 LoRA adaptation on the college PC

The college PC has an RTX 4500 Ada (24 GB VRAM) sitting idle -- a real upgrade
over Colab's T4 (16 GB, no bf16). This doc is the literal, ordered command list
to get from a bare Windows machine to a trained + evaluated LoRA adapter.

**Do not zip and copy the whole repo folder.** A zip drags `.git` (190+ commits
of history, any cached credential helper state) and every gitignored local
artefact (`__pycache__`, `.venv`, logs) across for no reason. Clone fresh
instead -- it is smaller, and it guarantees the college PC starts from exactly
what is on GitHub, not from whatever this laptop's working tree happens to hold
right now.

---

## What you need before you start

Stage-3 needs three things that only exist on this laptop:

1. **The Stage-1 checkpoint** (`checkpoints/baseline/best.pt`, 362 MB) --
   LoRA adapts this, it does not train from scratch.
2. **Code-mixed spoof clips for the eval column** that already exist
   (`C:\dfdata\colab_bundle\`, 0.82 GB, 4,434 clips).
3. **25 MUCS source recordings** (~500 MB) for the 10 adaptation-pool + 15
   eval-pool speakers -- references XTTS needs to clone those voices, since
   every clip generated so far used the other 25 (train-pool) speakers only.

None of this is in git (data and checkpoints are gitignored by design --
`GIT_RULES.md` §1). It travels by USB.

---

## Step 1 -- on THIS laptop: stage the transfer folder

```powershell
cd C:\Users\kuchu\codemix-deepfake-detection
.\scripts\stage_college_pc_transfer.ps1 -OutDir E:\college_pc_transfer
```

(Replace `E:\` with your USB drive's letter.) This script copies exactly the
four items above into one folder -- nothing else -- and prints the total size
(should land around 3 GB). Read its output: it warns loudly if anything is
missing rather than silently shipping a partial transfer.

**Verify before unplugging the drive:**

```powershell
Get-ChildItem E:\college_pc_transfer -Recurse -File | Measure-Object Length -Sum
```

Expect roughly: `best.pt` (362 MB) + `colab_bundle\` (0.82 GB) +
`raw\mucs2021\train\` (~500 MB) + `tts_cache\` (~1.9 GB) + `manifests\` (small)
+ `.env` / `.env.git` (tiny) ≈ **3 GB**.

---

## Step 2 -- on the college PC: clone and set up Python

```powershell
git clone https://github.com/Mounika-Reddy-0802/codemix-deepfake-detection.git
cd codemix-deepfake-detection
sh scripts/install_hooks.sh          # or: git config core.hooksPath githooks

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`ffmpeg` must be on `PATH` for AMR-NB channel simulation later; not required
for the LoRA run itself.

---

## Step 3 -- on the college PC: copy the staged data into place

Plug in the USB drive (say it mounts as `E:\college_pc_transfer`). Pick a data
drive with room -- **not C:**, this laptop ran out of space doing exactly this.

```powershell
$DataRoot = "D:\dfdata"
$Staged   = "E:\college_pc_transfer"

New-Item -ItemType Directory -Force -Path $DataRoot | Out-Null
Copy-Item "$Staged\.env"     .\  -Force
Copy-Item "$Staged\.env.git" .\  -Force

New-Item -ItemType Directory -Force -Path "checkpoints\baseline" | Out-Null
Copy-Item "$Staged\best.pt" "checkpoints\baseline\best.pt" -Force

robocopy "$Staged\colab_bundle"        "$DataRoot\colab_bundle"        /E
robocopy "$Staged\raw\mucs2021\train"  "$DataRoot\raw\mucs2021\train"  /E
Copy-Item "$Staged\manifests\clip_index.csv"    "data\manifests\" -Force
Copy-Item "$Staged\manifests\speaker_pools.csv" "data\manifests\" -Force

New-Item -ItemType Directory -Force -Path "$env:LOCALAPPDATA\tts" | Out-Null
robocopy "$Staged\tts_cache\tts_models--multilingual--multi-dataset--xtts_v2" `
         "$env:LOCALAPPDATA\tts\tts_models--multilingual--multi-dataset--xtts_v2" /E

setx DATA_ROOT $DataRoot
setx DFD_DEVICE cuda
```

**Close and reopen PowerShell** after `setx` (it does not affect the current
session). Then verify:

```powershell
echo $env:DATA_ROOT                 # should print D:\dfdata (new window only)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
Test-Path "checkpoints\baseline\best.pt"
Test-Path "$env:LOCALAPPDATA\tts\tts_models--multilingual--multi-dataset--xtts_v2\hash.md5"
```

If `hash.md5` is missing, Coqui will silently re-download 1.9 GB the first
time you generate -- not fatal, just slower.

---

## Step 4 -- generate the adaptation-pool spoof clips

Builds the job table (10 adaptation speakers, one reference clip each, ~1,600
clips at 160/speaker) and cuts the reference wavs:

```powershell
python -m src.data.pool_jobs --pool adaptation --data-root $env:DATA_ROOT
```

Then actually synthesise (this is the slow step -- the RTX 4500 will do this
far faster than a CPU laptop; expect roughly 1-3 seconds/clip on that GPU):

```powershell
python -m src.data.spoof_generation `
  --jobs data\manifests\adaptation_generation_jobs.csv `
  --pack-dir "$env:DATA_ROOT\generated\xtts_v2_adaptation" `
  --device cuda
```

Repeat for the eval pool (15 speakers, needed so LoRA's gap-closure number
comes from held-out voices, not the ones it trained on):

```powershell
python -m src.data.pool_jobs --pool eval --data-root $env:DATA_ROOT

python -m src.data.spoof_generation `
  --jobs data\manifests\eval_generation_jobs.csv `
  --pack-dir "$env:DATA_ROOT\generated\xtts_v2_eval" `
  --device cuda
```

**The ethics gate applies here too** -- generation refuses to run without the
signed mentor note. It is already in `docs/ethics/` in this repo (cloned in
Step 2), so it should pass without anything extra.

---

## Step 5 -- build the LoRA train/dev/eval manifests

```powershell
python -m src.data.codemix_manifests --pool adaptation `
  --jobs "$env:DATA_ROOT\generated\xtts_v2_adaptation\generation_jobs.csv" `
  --pack-dir "$env:DATA_ROOT\generated\xtts_v2_adaptation"

python -m src.data.codemix_manifests --pool eval `
  --jobs "$env:DATA_ROOT\generated\xtts_v2_eval\generation_jobs.csv" `
  --pack-dir "$env:DATA_ROOT\generated\xtts_v2_eval"
```

This writes three `*_clean.csv` files -- bonafide rows still point at whole
MUCS recordings, uncropped. Materialise them (crops each row to its actual
utterance; this is the exact step that fixed the "25 repeated clips" gap-matrix
bug, so do not skip it):

```powershell
python -m src.data.portable_bundle `
  --manifest data\manifests\codemix_adapt_train_clean.csv `
  --out-dir "$env:DATA_ROOT\lora_bundle" `
  --manifest-out data\manifests\codemix_adapt_train.csv

python -m src.data.portable_bundle `
  --manifest data\manifests\codemix_adapt_dev_clean.csv `
  --out-dir "$env:DATA_ROOT\lora_bundle" `
  --manifest-out data\manifests\codemix_adapt_dev.csv

python -m src.data.portable_bundle `
  --manifest data\manifests\codemix_eval_clean.csv `
  --out-dir "$env:DATA_ROOT\lora_bundle" `
  --manifest-out data\manifests\codemix_eval.csv
```

---

## Step 6 -- run LoRA adaptation

```powershell
python -m pytest tests\test_splits.py tests\test_lora.py -q     # gate, must be green
python -m src.training.train --config configs\train_lora_codemix.yaml --device cuda
```

`configs/train_lora_codemix.yaml` already points `init_from` at
`checkpoints/baseline/best.pt` and `train/dev_manifest` at the two files Step 5
just produced. Progress prints every 25 steps (`log_every`); it writes
`checkpoints/lora_codemix/last.pt` every epoch, so `--resume` picks back up if
the session is interrupted.

---

## Step 7 -- evaluate the gap-closure number

```powershell
python -m src.training.evaluate `
  --checkpoint checkpoints\lora_codemix\best.pt `
  --manifest data\manifests\codemix_eval.csv `
  --device cuda --batch-size 32 `
  --out experiments\lora_codemix_eval.json
```

Compare `experiments/lora_codemix_eval.json`'s EER against the existing
`docs/results/gap_matrix_v1.md` code-mixed number (53.3% EER, S1 unmodified) --
that delta is the paper's gap-closure result.

---

## Step 8 -- bring the results back

Results are small text files; they go in git. Checkpoints do not (`GIT_RULES.md`
§1: data and checkpoints never enter git).

```powershell
set -a; . ./.env.git; set +a   # if using bash; in PowerShell, load .env.git values manually
git add experiments\lora_codemix_eval.json docs\progress.md
git -c user.name="$M_GIT_NAME" -c user.email="$M_GIT_EMAIL" commit -m "add lora codemix eval results"
git push "https://$M_GH_USER:$M_GH_PAT@$REPO_URL" week3-mounika-affectdf-taxonomy
```

Copy `checkpoints/lora_codemix/best.pt` back to this laptop or Drive by USB --
same reasoning as `best.pt` in Step 1, just in reverse.

---

## If something breaks

- **`CUDA out of memory`** -- lower `batch_size` in `configs/train_lora_codemix.yaml`
  (24 GB should comfortably hold the default `batch_size: 8`, `max_seconds: 4.0`;
  this is a config problem, not a hardware one, if it happens).
- **`DATA_ROOT` not resolving** -- you opened the terminal before `setx` ran, or
  before this session's PowerShell was reopened. Check `echo $env:DATA_ROOT`.
- **Ethics gate refuses to run** -- confirm `docs/ethics/mentor_signoff_*.pdf`
  exists (it should have come through the clone; it is a tracked file, unlike
  most of `docs/ethics/`).
- **XTTS re-downloads 1.9 GB anyway** -- `hash.md5` did not make it across;
  Coqui compares that file's contents against its manifest, not the model
  weights themselves, so a missing `hash.md5` looks unverified even with
  `model.pth` present.
