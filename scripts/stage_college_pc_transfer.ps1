<#
.SYNOPSIS
  Build a USB-ready folder holding everything Stage-3 LoRA needs on a third
  machine, and nothing else.

.DESCRIPTION
  Run on THIS machine before copying to a USB drive. Collects:
    - the Stage-1 checkpoint (362 MB)
    - the existing portable code-mixed bundle, colab_bundle/ (0.82 GB)
    - ONLY the MUCS source recordings the adaptation + eval pool speakers need
      (25 files, ~500 MB -- not the full 11 GB corpus)
    - the XTTS-v2 model cache, so the college PC does not re-download 1.9 GB
    - .env / .env.git (never touch the repo's git history for these -- they
      are gitignored on purpose)

  Does NOT copy the repo itself. Clone that fresh on the college PC instead
  (see docs/college_pc_setup.md) -- copying .git as a folder risks dragging a
  stale working tree or, worse, credentials cached in a helper, over with it.

.EXAMPLE
  .\stage_college_pc_transfer.ps1 -OutDir E:\college_pc_transfer
#>
param(
  [Parameter(Mandatory = $true)][string]$OutDir,
  [string]$DataRoot = "C:\dfdata",
  [string]$RepoDir = (Get-Location)
)

$ErrorActionPreference = "Stop"
function Step($n, $msg) { Write-Host "`n[$n] $msg" -ForegroundColor Cyan }
function Ok($msg)       { Write-Host "    OK  $msg" -ForegroundColor Green }
function Warn($msg)     { Write-Host "    !!  $msg" -ForegroundColor Yellow }

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Step 1 "Stage-1 checkpoint"
$ckpt = Join-Path $RepoDir "checkpoints\baseline\best.pt"
if (Test-Path $ckpt) {
  Copy-Item $ckpt (Join-Path $OutDir "best.pt") -Force
  Ok "best.pt copied ($((Get-Item $ckpt).Length / 1MB) MB)"
} else {
  Warn "best.pt not found at $ckpt -- Stage-3 needs this. Fix before copying to USB."
}

Step 2 "Portable code-mixed bundle (existing gap-matrix clips)"
$bundle = Join-Path $DataRoot "colab_bundle"
if (Test-Path $bundle) {
  robocopy $bundle (Join-Path $OutDir "colab_bundle") /E /NFL /NDL /NJH /NJS | Out-Null
  Ok "colab_bundle copied"
} else {
  Warn "$bundle not found -- skipping (gap-matrix eval column will be unavailable)"
}

Step 3 "MUCS source recordings for the adaptation + eval pool speakers only"
Push-Location $RepoDir
$py = @"
import pandas as pd
idx = pd.read_csv('data/manifests/clip_index.csv')
pools = pd.read_csv('data/manifests/speaker_pools.csv')
pools['speaker'] = pools['speaker'].astype(str); idx['speaker'] = idx['speaker'].astype(str)
need = set(pools.loc[pools['pool'].isin(['adaptation', 'eval']), 'speaker'])
sub = idx[idx['speaker'].isin(need)]
for p in sorted(sub['wav_path'].dropna().unique()):
    print(p)
"@
$venvPy = Join-Path $RepoDir ".venv\Scripts\python.exe"
$pythonExe = if (Test-Path $venvPy) { $venvPy } else { "python" }
$files = $py | & $pythonExe -
Pop-Location
$mucsOut = Join-Path $OutDir "raw\mucs2021\train"
New-Item -ItemType Directory -Force -Path $mucsOut | Out-Null
$copied = 0
foreach ($f in $files) {
  if (Test-Path $f) {
    Copy-Item $f (Join-Path $mucsOut (Split-Path $f -Leaf)) -Force
    $copied++
  } else {
    Warn "source file missing on disk: $f"
  }
}
Ok "$copied MUCS recording(s) copied to raw\mucs2021\train\"

Step 4 "Speaker pool + clip index manifests (small, needed for job assembly)"
$manifestOut = Join-Path $OutDir "manifests"
New-Item -ItemType Directory -Force -Path $manifestOut | Out-Null
Copy-Item (Join-Path $RepoDir "data\manifests\clip_index.csv") $manifestOut -Force
Copy-Item (Join-Path $RepoDir "data\manifests\speaker_pools.csv") $manifestOut -Force
Ok "clip_index.csv + speaker_pools.csv copied"

Step 5 "XTTS-v2 model cache (avoids a 1.9 GB re-download)"
$xtts = Join-Path $env:LOCALAPPDATA "tts\tts_models--multilingual--multi-dataset--xtts_v2"
if (Test-Path $xtts) {
  $xttsOut = Join-Path $OutDir "tts_cache\tts_models--multilingual--multi-dataset--xtts_v2"
  robocopy $xtts $xttsOut /E /NFL /NDL /NJH /NJS | Out-Null
  if (Test-Path (Join-Path $xttsOut "hash.md5")) {
    Ok "XTTS model cache copied (hash.md5 present)"
  } else {
    Warn "hash.md5 missing -- Coqui will re-verify/re-download on the college PC"
  }
} else {
  Warn "$xtts not found -- the college PC will download ~1.9 GB on first run"
}

Step 6 "Secrets (.env, .env.git -- gitignored, copy by hand, not via git)"
foreach ($f in @(".env", ".env.git")) {
  $src = Join-Path $RepoDir $f
  if (Test-Path $src) {
    Copy-Item $src $OutDir -Force
    Ok "$f copied"
  } else {
    Warn "$f not found -- copy it manually before running anything on the college PC"
  }
}

$size = "{0:N2}" -f ((Get-ChildItem $OutDir -Recurse -File | Measure-Object Length -Sum).Sum / 1GB)
Write-Host "`nStaged $size GB at $OutDir -- copy this whole folder to the USB drive." -ForegroundColor Green
Write-Host "Next: docs/college_pc_setup.md on the college PC." -ForegroundColor Green
