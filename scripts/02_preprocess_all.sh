#!/usr/bin/env bash
# =============================================================================
# 02_preprocess_all.sh  --  preprocess every corpus to 16 kHz segments (Week 2, L)
#
# Runs: resample -> mono -> silero VAD trim -> loudness norm -> 2-10 s segmentation
# over each downloaded corpus, writing clean segments to data/processed/clean/<corpus>.
# Channel-simulated copies are produced later (Week 4, at manifest/eval build time)
# so the clean copies are retained separately.
#
# Requires the ML deps installed (torch/torchaudio/librosa/silero-vad/soundfile)
# and the corpora downloaded (scripts/01_download_data.sh --run).
#
#   bash scripts/02_preprocess_all.sh
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RAW="data/raw"
OUT="data/processed/clean"

# corpus raw-subdir -> output name
CORPORA=(
  "asvspoof2019_LA:asvspoof2019_LA"
  "mucs2021:mucs2021"
  "hiacc:hiacc"          # adult only; _EXCLUDED_children/ is skipped below
)

for entry in "${CORPORA[@]}"; do
  src="${RAW}/${entry%%:*}"
  name="${entry##*:}"
  if [ ! -d "$src" ]; then
    echo "[skip] $src not found (download it first with scripts/01_download_data.sh --run)"
    continue
  fi
  echo "=== preprocessing $name ==="
  # Guard: never touch the quarantined child folder.
  if [ "$name" = "hiacc" ] && [ -d "${src}/_EXCLUDED_children" ]; then
    echo "  (HiACC child folder is quarantined and will not be processed)"
  fi
  python -m src.data.preprocess --in-dir "$src" --out-dir "${OUT}/${name}"
done

echo "preprocessing complete -> ${OUT}"
