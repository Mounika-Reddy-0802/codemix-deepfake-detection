<#
.SYNOPSIS
  One-shot setup of the GPU laptop: repo, Python env, corpora, verification.

.DESCRIPTION
  Brings a second Windows machine to the exact state of the dev laptop so that
  spoof generation and training can run there. Safe to re-run: every step checks
  whether it is already done.

  It does NOT download the corpora. It expects the archives to already be on the
  machine (LA.zip, Hindi-English_train.tar.gz, Hindi-English_test.tar.gz, and the
  HiACC zip) and extracts them in place, applying the same child quarantine the
  download path applies.

.EXAMPLE
  .\setup_gpu_laptop.ps1 -DataRoot D:\dfdata -Archives D:\downloads -Bundle .\gpu-laptop-bundle.zip
#>
param(
  # Where the dfdata tree should live on THIS machine.
  [string]$DataRoot = "D:\dfdata",
  # Folder holding the corpus archives, if they are not already under $DataRoot\raw.
  [string]$Archives = "",
  # Where to clone the repo.
  [string]$RepoDir = "$HOME\codemix-deepfake-detection",
  # gpu-laptop-bundle.zip (ethics note + pilot pack). Defaults to one next to this script.
  [string]$Bundle = "",
  # Skip the pip install step (useful when re-running after it succeeded once).
  [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$RepoUrl = "https://github.com/Mounika-Reddy-0802/codemix-deepfake-detection.git"
$WeekBranches = @(
  "week3-lahari-preprocess-mucs-hiacc",
  "week3-krishna-xtts-rvc-pilot",
  "week3-mounika-affectdf-taxonomy"
)

function Step($n, $msg) { Write-Host "`n[$n] $msg" -ForegroundColor Cyan }
function Ok($msg)       { Write-Host "    OK  $msg" -ForegroundColor Green }
function Warn2($msg)    { Write-Host "    !!  $msg" -ForegroundColor Yellow }
function Die($msg)      { Write-Host "`nFAILED: $msg" -ForegroundColor Red; exit 1 }

function Have($name) {
  $c = Get-Command $name -ErrorAction SilentlyContinue
  return ($null -ne $c)
}

Write-Host "=================================================================="
Write-Host " GPU laptop setup  --  codemix deepfake detection"
Write-Host " DataRoot : $DataRoot"
Write-Host " RepoDir  : $RepoDir"
Write-Host "=================================================================="

# ---------------------------------------------------------------- 1. preflight
Step 1 "Checking prerequisites"

if (-not (Have git))    { Die "git not found. Install Git for Windows: https://git-scm.com/download/win" }
Ok "git"

$py = $null
foreach ($cand in @("python", "py")) {
  if (Have $cand) {
    $v = & $cand -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
    if ($LASTEXITCODE -eq 0 -and $v) { $py = $cand; Ok "python $v ($cand)"; break }
  }
}
if (-not $py) { Die "no working Python found. Install Python 3.10-3.13 and tick 'Add to PATH'." }

if (Have nvidia-smi) {
  $gpu = (& nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1)
  if ($gpu) { Ok "GPU: $gpu" } else { Warn2 "nvidia-smi ran but reported no GPU" }
} else {
  Warn2 "nvidia-smi not found. If this machine has no NVIDIA GPU, training will be very slow."
}

# Git Bash is needed for the corpus extraction script.
$bash = $null
foreach ($p in @("C:\Program Files\Git\bin\bash.exe", "C:\Program Files (x86)\Git\bin\bash.exe")) {
  if (Test-Path $p) { $bash = $p; break }
}
if (-not $bash -and (Have bash)) { $bash = (Get-Command bash).Source }
if ($bash) { Ok "bash: $bash" } else { Warn2 "Git Bash not found - corpus extraction will be skipped" }

# ------------------------------------------------------------------- 2. repo
Step 2 "Getting the repository"

if (Test-Path (Join-Path $RepoDir ".git")) {
  Ok "repo already at $RepoDir"
  Push-Location $RepoDir
  git fetch --all --prune
} else {
  git clone $RepoUrl $RepoDir
  if ($LASTEXITCODE -ne 0) { Die "git clone failed" }
  Push-Location $RepoDir
}

git config core.hooksPath githooks
git checkout dev
git pull --ff-only origin dev

# The week-3 work may still be sitting in open PRs. Merge whatever is not yet in
# dev onto a local branch so this machine has the generation CLI and the fixed
# dependency pins either way. If the PRs are already merged these are no-ops.
git checkout -B gpu-setup dev | Out-Null
foreach ($b in $WeekBranches) {
  git rev-parse --verify "origin/$b" *>$null
  if ($LASTEXITCODE -ne 0) { Warn2 "branch $b not on the remote (already merged and deleted?)"; continue }
  $ahead = (git rev-list --count "dev..origin/$b" 2>$null)
  if ($ahead -eq "0") { Ok "$b already in dev" ; continue }
  git merge --no-edit "origin/$b" *>$null
  if ($LASTEXITCODE -ne 0) { git merge --abort; Die "merge conflict on $b - resolve by hand" }
  Ok "merged $b ($ahead commit(s))"
}
Ok "on branch gpu-setup"

# ----------------------------------------------------------- 3. python env
Step 3 "Python environment"

if (-not (Test-Path ".venv")) { & $py -m venv .venv; Ok "created .venv" } else { Ok ".venv exists" }
$vpy = Join-Path $PWD ".venv\Scripts\python.exe"
if (-not (Test-Path $vpy)) { Die "venv python missing at $vpy" }

if ($SkipInstall) {
  Warn2 "skipping installs (-SkipInstall)"
} else {
  & $vpy -m pip install --quiet --upgrade pip

  # CUDA build FIRST so the generic wheels in requirements.txt do not win.
  # Pinned to 2.8: from 2.9 torchaudio routes audio IO through torchcodec, which
  # fails to load against FFmpeg 8/9 and breaks XTTS.
  $cuda = & $vpy -c "import torch,sys; sys.stdout.write('1' if torch.cuda.is_available() else '0')" 2>$null
  if ($cuda -ne "1") {
    Write-Host "    installing CUDA torch (large download, please wait)..."
    & $vpy -m pip install --quiet torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu121
    if ($LASTEXITCODE -ne 0) { Warn2 "CUDA torch install failed - check https://pytorch.org for your driver" }
  }
  Write-Host "    installing project requirements..."
  & $vpy -m pip install --quiet -r requirements.txt
  if ($LASTEXITCODE -ne 0) { Die "pip install -r requirements.txt failed" }
  Ok "dependencies installed"
}

# ffmpeg: the CLI covers AMR-NB; the shared build is what torchcodec loads.
if (-not (Have ffmpeg)) {
  if (Have winget) {
    Write-Host "    installing ffmpeg..."
    winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements --disable-interactivity | Out-Null
    winget install --id Gyan.FFmpeg.Shared -e --accept-source-agreements --accept-package-agreements --disable-interactivity | Out-Null
    Ok "ffmpeg installed (reopen the terminal if it is still not on PATH)"
  } else {
    Warn2 "ffmpeg missing and winget unavailable - the AMR-NB channel condition will fall back to G.711"
  }
} else { Ok "ffmpeg" }

# ------------------------------------------------------------- 4. the bundle
Step 4 "Unpacking the transfer bundle (ethics note + pilot pack)"

if (-not $Bundle) {
  $guess = Join-Path $PSScriptRoot "gpu-laptop-bundle.zip"
  if (Test-Path $guess) { $Bundle = $guess }
}
if ($Bundle -and (Test-Path $Bundle)) {
  $tmp = Join-Path $env:TEMP "dfbundle"
  if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
  Expand-Archive -Path $Bundle -DestinationPath $tmp -Force

  $pdf = Get-ChildItem $tmp -Recurse -Filter "mentor_signoff*.pdf" -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($pdf) {
    New-Item -ItemType Directory -Force -Path "docs\ethics" | Out-Null
    Copy-Item $pdf.FullName "docs\ethics\" -Force
    Ok "ethics note -> docs\ethics\$($pdf.Name)"
  } else { Warn2 "no signed ethics PDF in the bundle - generation will refuse to run" }

  $pilot = Join-Path $tmp "pilot"
  if (Test-Path $pilot) {
    $dest = Join-Path $DataRoot "generated\pilot"
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    Copy-Item "$pilot\*" $dest -Recurse -Force
    Ok "pilot pack -> $dest"
  }
} else {
  Warn2 "no bundle given. Copy docs\ethics\mentor_signoff_*.pdf by hand or generation will refuse."
}

# --------------------------------------------------------- 5. extract corpora
Step 5 "Extracting the corpora (with child quarantine)"

$rawDir = Join-Path $DataRoot "raw"
New-Item -ItemType Directory -Force -Path $rawDir | Out-Null

if ($bash) {
  $dr = "/" + $DataRoot.Substring(0,1).ToLower() + ($DataRoot -replace "\\","/").Substring(2)
  $cmd = "DATA_ROOT='$dr' bash scripts/01_download_data.sh --extract-only"
  if ($Archives) {
    $ar = "/" + $Archives.Substring(0,1).ToLower() + ($Archives -replace "\\","/").Substring(2)
    $cmd += " --archives '$ar'"
  }
  Write-Host "    $cmd"
  & $bash -lc "cd '$((Get-Location).Path -replace '\\','/')' && $cmd"
  if ($LASTEXITCODE -ne 0) { Warn2 "extraction reported problems - check the output above" } else { Ok "corpora extracted" }
} else {
  Warn2 "no bash: run this yourself in Git Bash"
  Warn2 "  DATA_ROOT=/d/dfdata bash scripts/01_download_data.sh --extract-only"
}

# ------------------------------------------------- 6. index + persist env vars
Step 6 "Indexing corpora and setting environment variables"

[Environment]::SetEnvironmentVariable("DATA_ROOT", $DataRoot, "User")
[Environment]::SetEnvironmentVariable("DFD_DEVICE", "cuda", "User")
$env:DATA_ROOT = $DataRoot
$env:DFD_DEVICE = "cuda"
Ok "DATA_ROOT and DFD_DEVICE set for your user (permanent)"

$hiacc = Join-Path $rawDir "hiacc"
if (Test-Path $hiacc) {
  & $vpy -m src.data.quarantine --root $hiacc
  Ok "child-quarantine audit written"
}

& $vpy -m src.data.corpora --data-root $DataRoot
if ($LASTEXITCODE -ne 0) { Warn2 "indexing failed - are the corpora extracted?" } else { Ok "clip_index.csv rebuilt" }

# ------------------------------------------------------------- 7. verify
Step 7 "Verification"

& $vpy -m src.utils.device
& $vpy -m src.data.ethics_gate
if ($LASTEXITCODE -ne 0) { Warn2 "ETHICS GATE CLOSED - generation will refuse until the signed PDF is in docs\ethics\" }

& $vpy -m pytest -q 2>&1 | Select-Object -Last 3

Pop-Location

Write-Host "`n=================================================================="
Write-Host " Setup finished." -ForegroundColor Green
Write-Host "=================================================================="
Write-Host @"

Check these numbers match the dev laptop (printed in step 6):
    MUCS   52,825 utterances / 520 speakers
    HiACC   3,318 adult clips / 24 speakers  (1,858 child files quarantined)
If they differ, an archive is incomplete - stop and fix before generating.

Open a NEW terminal (so DATA_ROOT / DFD_DEVICE apply), then:

  cd $RepoDir
  .venv\Scripts\activate

  # Reproduce the 20-clip pilot on GPU (~5 min) to prove the stack end to end
  python -m src.data.pilot_jobs --data-root $DataRoot --pack-dir $DataRoot\generated\pilot
  `$env:COQUI_TOS_AGREED="1"
  python -m src.data.spoof_generation ``
    --jobs $DataRoot\generated\pilot\generation_jobs.csv ``
    --pack-dir $DataRoot\generated\pilot ``
    --out-dir $DataRoot\generated\pilot\outputs_gpu

  # Training, once manifests exist (always smoke-test first)
  python -m src.training.train --config configs/train_baseline.yaml --smoke
  python -m src.training.train --config configs/train_baseline.yaml

NOT included in the bundle, set up by hand:
  .env       - HF_TOKEN and WANDB_API_KEY (copy .env.example and fill in)
  .env.git   - git identities + tokens. Deliberately excluded: it holds real
               GitHub tokens and must never travel in a shared zip.

"@
