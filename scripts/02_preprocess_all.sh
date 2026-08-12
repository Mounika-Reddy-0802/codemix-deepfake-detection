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

# -----------------------------------------------------------------------------
# HiACC ethics gate. The old version of this script only PRINTED a reassuring
# line while src.data.preprocess walked the corpus root with rglob -- which would
# have pulled the quarantined child audio straight back in. The exclusion now
# lives in preprocess.audio_files(), and the audit below refuses to let
# preprocessing start while anything child-looking is still outside quarantine.
# -----------------------------------------------------------------------------
hiacc_gate() {
  local src="$1"
  echo "--- HiACC child-quarantine audit ---"
  python -m src.data.quarantine --root "$src" --out docs/qa/child_quarantine_check.md \
    | tee /tmp/hiacc_audit.log
  if grep -q "WARNING" /tmp/hiacc_audit.log; then
    echo "REFUSING to preprocess HiACC: the audit raised a warning." >&2
    echo "Read docs/qa/child_quarantine_check.md and fix the quarantine first." >&2
    return 1
  fi
  if [ ! -f docs/qa/child_quarantine_check.md ]; then
    echo "REFUSING to preprocess HiACC: no audit report was written." >&2
    return 1
  fi
  echo "audit clean; the signed report is still required before generation."
}

for entry in "${CORPORA[@]}"; do
  src="${RAW}/${entry%%:*}"
  name="${entry##*:}"
  if [ ! -d "$src" ]; then
    echo "[skip] $src not found (download it first with scripts/01_download_data.sh --run)"
    continue
  fi
  if [ "$name" = "hiacc" ]; then
    hiacc_gate "$src" || { echo "[skip] hiacc blocked by the quarantine audit"; continue; }
  fi
  echo "=== preprocessing $name ==="
  python -m src.data.preprocess --in-dir "$src" --out-dir "${OUT}/${name}"
done

echo "preprocessing complete -> ${OUT}"
