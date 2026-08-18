# Moving generation + training to the GPU laptop

Everything heavy — Week-4 spoof generation, S1/S2/S3 training, the gap matrix —
runs on the GPU laptop from here on. This laptop keeps the CPU-side work
(indexing, preprocessing, channel sim, inference, the demos) and the writing.

Nothing in the code needs editing to move. Two environment variables carry every
machine-specific fact, and the manifests are path-portable
(`src/utils/paths.py` rebases clip paths onto whatever `DATA_ROOT` says).

| Variable | This laptop | GPU laptop |
|---|---|---|
| `DATA_ROOT` | `C:/dfdata` | wherever you put the data, e.g. `D:/dfdata` |
| `DFD_DEVICE` | unset (→ cpu) | `cuda` |

---

## 1. How much audio are we actually generating?

**The 20-clip pilot is not the corpus.** It is a 4-cell experiment (Devanagari vs
romanised vs mixed script × `hi` vs `en` tag, 5 clips each) whose only purpose is
to decide *how* to generate before spending GPU hours generating wrong. It is
done; it now needs ears, not compute.

The corpus is Week 4. Measured from the frozen speaker pools and the real MUCS
transcripts (`clip_index.csv`, after dropping transcripts that are too short, over
XTTS's 150-char Hindi limit, or contain digits XTTS cannot say in Hindi):

| Pool | Speakers | Usable transcripts | What it generates |
|---|---|---|---|
| **train** | 25 | **1,454** | XTTS-v2 + RVC — the *seen* attack, may enter training |
| **adaptation** | 10 | **748** | Stage-3 / LoRA code-mixed training slice |
| **eval** | 15 | **1,014** | Tortoise — the *unseen* held-out attack, never trained on |

Plan targets (`PROJECT_PLAN_V2_AFFECTDF.md` W4-T1…T3):

| Track | Tool | Pool | Target | Notes |
|---|---|---|---|---|
| Seen attack | XTTS-v2 | train | **~4,000** | 1,454 unique transcripts × 25 speaker voices = ~36k possible pairings, so 4,000 is a *choice*, not a ceiling |
| Second attack family | RVC | train | 10–15 speaker models | per-speaker training, overnight batches |
| **Unseen attack** | Tortoise | **eval only** | **400–600** | firewalled — `tests/test_splits.py` fails if it ever appears in a training manifest |

So roughly **4,500–5,000 generated clips**, not 20.

### How long that takes

Measured on this CPU laptop: **~3–4 minutes per clip.** A GPU runs XTTS-v2 roughly
20–40× faster.

| Job | CPU (here) | GPU laptop |
|---|---|---|
| 20-clip pilot | ~75 min | ~5 min |
| 4,000 XTTS clips | **~11 days** | **~11–17 hours** |
| 500 Tortoise clips | weeks | ~8–17 h (Tortoise is slow even on GPU) |

That is why generation moves. The run is **resumable** — `pending_jobs` skips any
clip whose file already exists — so you can stop and restart across sessions
without regenerating or duplicating metadata.

---

## 2. Get the code onto the GPU laptop

Merge the open PRs into `dev` first, so the new machine starts from a complete
tree.

```powershell
git clone https://github.com/Mounika-Reddy-0802/codemix-deepfake-detection.git
cd codemix-deepfake-detection
git checkout dev
git config core.hooksPath githooks      # attribution guard; replaces install_hooks.sh
```

Python environment (3.10–3.13):

```powershell
python -m venv .venv
.venv\Scripts\activate

# CUDA build FIRST, so the CPU wheels in requirements.txt do not win.
# cu121 works for most consumer GPUs; check https://pytorch.org for your driver.
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

> **Do not "upgrade" torch past 2.8.** From 2.9 torchaudio routes audio IO through
> `torchcodec`, which fails to load against FFmpeg 8/9 (`WinError 127`). Keep
> torch / torchaudio / torchvision on the same minor or transformers dies with
> *"operator torchvision::nms does not exist"*. This is pinned in
> `requirements.txt` with the reasoning.

ffmpeg (needed for the AMR-NB channel condition):

```powershell
winget install Gyan.FFmpeg
```

Verify before doing anything expensive:

```powershell
python -m src.utils.device        # MUST print a GPU name, not "cpu"
python -m pytest -q               # 337 tests
```

### Pre-fetch the XTTS-v2 model (saves an hour of grief)

The first generation run downloads ~1.9 GB of model weights through Coqui's own
downloader, which has **no resume**. On a domestic connection that is 20+ minutes,
and if the run is interrupted the next one starts again from zero.

Pull the same files from Hugging Face instead — it resumes, and it is faster:

```powershell
python -c "from huggingface_hub import hf_hub_download as d; [d('coqui/XTTS-v2', f, local_dir=r'$env:LOCALAPPDATA\tts\tts_models--multilingual--multi-dataset--xtts_v2') for f in ('config.json','vocab.json','speakers_xtts.pth','hash.md5','model.pth')]"
```

> **`hash.md5` is not optional.** Coqui decides a model is already present by
> comparing that file's contents to the checksum in its `models.json`; if the file
> is missing it logs *"has been updated, clearing model cache..."* and re-downloads
> everything — **deleting the model.pth you just fetched**. Downloading the four
> obvious files and skipping the 32-byte one costs the whole 1.9 GB again. It is a
> version marker, not a checksum of `model.pth`, so its contents will not match
> that file's actual md5.

---

## 3. Get the three things git will NOT bring

`.gitignore` deliberately excludes these. Copy them by hand (USB, or any private
channel — **never** commit them):

| File | Why it matters |
|---|---|
| `docs/ethics/mentor_signoff_2026-08-12.pdf` | **Generation refuses to run without it.** Gitignored because it carries real signatures. There is no override flag. |
| `.env` | `HF_TOKEN`, `WANDB_API_KEY` |
| `.env.git` | git identities for per-owner commits |

Confirm the gate opens on the new machine:

```powershell
python -m src.data.ethics_gate     # must print "ethics gate OPEN" and exit 0
```

---

## 4a. If the GPU laptop already has the archives

This is the common case: the zips/tarballs were copied or downloaded there, but
nothing is extracted, quarantined or indexed. **Do not extract by hand** — hand
extraction is exactly how the HiACC child quarantine gets skipped, and that is the
one mistake in this project that is an ethics violation rather than a bug.

```bash
# Git Bash. DATA_ROOT is where the dfdata tree should end up.
cd codemix-deepfake-detection

# Archives sitting in DATA_ROOT/raw/<corpus>/ already:
DATA_ROOT=/d/dfdata bash scripts/01_download_data.sh --extract-only

# Or archives sitting somewhere else (a USB drive, Downloads):
DATA_ROOT=/d/dfdata bash scripts/01_download_data.sh --extract-only --archives /e/archives

# One corpus at a time if you prefer:
DATA_ROOT=/d/dfdata bash scripts/01_download_data.sh --extract-only --only hiacc
```

`--extract-only` runs the **same** extraction and the **same** child-quarantine
sweep as the download path, with no network. It refuses to finish if a
child-looking folder survives the sweep. Missing archives are reported and
skipped rather than silently producing an empty corpus.

Then reach the same state as the first laptop:

```bash
python -m src.data.quarantine --root /d/dfdata/raw/hiacc   # audit + evidence report
python -m src.data.corpora --data-root /d/dfdata           # rebuild clip_index.csv (~1 min)
```

Expected result — compare against the first laptop:

```
raw/mucs2021/train/transcripts/{segments,utt2spk,wav.scp,text}
raw/mucs2021/test/transcripts/...
raw/hiacc/Corpus/adult/{audio/{train_split,val_split,test_split},metadata,annotations,transcription}
raw/hiacc/_EXCLUDED_children/...          <- child audio, quarantined
raw/asvspoof2019_LA/LA/...
```

Sanity numbers that must match: **MUCS 52,825 utterances / 520 speakers**,
**HiACC 3,318 adult clips / 24 speakers**, **1,858 child files quarantined**.
`python -m src.data.corpora` prints all of these. If they differ, stop — an
archive is incomplete or a folder is in the wrong place.

`speaker_pools.csv` is **frozen and committed** (SHA-256 `f57e0d85…`); it comes
from git and must never be regenerated, or the two machines are no longer
measuring the same split.

`processed/` does not need copying — it is regenerable, and nothing downstream of
`corpora.py` currently reads it.

## 4b. Get the dataset across (if the GPU laptop does NOT have it)

Current sizes under `C:\dfdata`:

| Directory | Size | Move it? |
|---|---|---|
| `raw/mucs2021` | **17.45 GB** | **Yes** — transcripts + reference audio; everything depends on it |
| `raw/hiacc` | **1.06 GB** | **Yes** — code-mixed eval column (adult only; child audio already quarantined) |
| `raw/asvspoof2019_LA` | 1.75 GB | **No — re-download.** It is an incomplete 25% of 7.11 GB; the Edinburgh server resets on sustained transfers, so pull it fresh there |
| `processed/` | 6.69 GB | **No** — fully regenerable from `raw/` in minutes |
| `generated/pilot` | 0.01 GB | Optional — tiny, and regenerable |

**≈ 18.5 GB to move.**

### Recommended: portable SSD / USB drive

Fastest and least error-prone for 18 GB. Copy `raw\mucs2021` and `raw\hiacc` only.

```powershell
# On THIS laptop -- E: is the USB drive
robocopy C:\dfdata\raw\mucs2021 E:\dfdata\raw\mucs2021 /E /Z /R:3 /W:5
robocopy C:\dfdata\raw\hiacc    E:\dfdata\raw\hiacc    /E /Z /R:3 /W:5

# On the GPU laptop -- D:\dfdata is the new DATA_ROOT
robocopy E:\dfdata\raw D:\dfdata\raw /E /Z /R:3 /W:5
```

`/Z` makes the copy restartable, so a disconnect mid-transfer resumes rather than
starting over.

### Alternative: both laptops on the same WiFi

Share `C:\dfdata\raw` on this machine, then from the GPU laptop:

```powershell
robocopy \\THIS-LAPTOP-NAME\raw D:\dfdata\raw /E /Z /R:3 /W:5
```

Slower than USB but needs no drive.

### Do NOT use Google Drive for this

18 GB of many small files through Drive sync is slow and silently skips files.
Use Drive only for the small artefacts (the signed PDF, checkpoints later).

### Then, on the GPU laptop

```powershell
setx DATA_ROOT "D:\dfdata"
setx DFD_DEVICE "cuda"
# reopen the terminal so these take effect

# Rebuild the clip index for the new paths (~1 min; the file is gitignored
# precisely because it is full of machine-local absolute paths)
python -m src.data.corpora --data-root D:\dfdata

# Re-download ASVspoof there (Stage-1's only training corpus)
bash scripts/01_download_data.sh --run
```

`speaker_pools.csv` is committed and frozen (SHA-256
`f57e0d85dbd3c8f5b96ad6af59edae7218104b0167807eae6433314947063e7b`), so the
train/adaptation/eval split travels with git and does **not** get recomputed. That
is what keeps the gap matrix valid across machines.

---

## 5. Run the work there

```powershell
# 1. Rebuild the pilot pack against the new DATA_ROOT
python -m src.data.pilot_jobs --data-root D:\dfdata --pack-dir D:\dfdata\generated\pilot

# 2. Reproduce the pilot on GPU (~5 min) to confirm the stack end to end
$env:COQUI_TOS_AGREED="1"
python -m src.data.spoof_generation `
  --jobs D:\dfdata\generated\pilot\generation_jobs.csv `
  --pack-dir D:\dfdata\generated\pilot `
  --out-dir D:\dfdata\generated\pilot\outputs

# 3. Week 4 at scale -- only after the team has rated the pilot and recorded
#    the script / language-tag / quality-bar decisions.

# 4. Training (Stage 1), once manifests exist
python -m src.training.train --config configs/train_baseline.yaml --smoke   # always smoke first
python -m src.training.train --config configs/train_baseline.yaml
```

---

## 6. Keeping track — what lives where

**In git (travels automatically):** all code, configs, `data/manifests/*.csv`
(including the frozen pools), docs, tests.

**Never in git, moved by hand once:** the signed ethics PDF, `.env`, `.env.git`.

**Never in git, moved by hand or regenerated:** raw audio, generated audio,
checkpoints. Reference these by **link + SHA-256 recorded in `docs/progress.md`**
(project rule §5) rather than committing them — that is how you keep track
without bloating the repo.

**Regenerated on each machine, never copied:** `data/manifests/clip_index.csv`
(gitignored — machine-local absolute paths), `processed/`.

Work stays synced the normal way: branch per person per week on the GPU laptop
too, push, open a PR into `dev`, merge after review. Results (`experiments/`,
`docs/`, `configs/`) are tracked, so pulling here brings the numbers back.
Checkpoints come back out of band — see `docs/split_setup_laptop_colab.md` §5.
