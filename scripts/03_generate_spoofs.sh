#!/usr/bin/env bash
# =============================================================================
# 03_generate_spoofs.sh  --  XTTS-v2 (+ Tortoise) clone generation (Week 2-3)
#
# Week 2 (SK): --pilot generates ~20 clones for the team to listen to and rate.
# Week 3 (L):  full run generates the 1,500-2,500 XTTS-v2 clones (seen attack);
#              (SK) the held-out Tortoise set (--tool tortoise) is EVAL-ONLY.
#
# Speaker pools (eval vs Stage-3 adaptation) are carved up front by
# build_manifests.carve_pools, so speaker-disjointness precedes any generation.
# HiACC child audio is never a cloning reference (enforced in spoof_generation.py).
#
# Requires XTTS-v2 deps (TTS/torch), a GPU (Colab/Kaggle), and preprocessed
# reference audio (scripts/02_preprocess_all.sh). Heavy job -> run in background:
#
#   nohup bash scripts/03_generate_spoofs.sh --pilot > data/raw/logs/gen_pilot.log 2>&1 &
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODE="pilot"        # pilot | full
TOOL="xtts_v2"      # xtts_v2 | tortoise
OUT="data/generated"

while [ $# -gt 0 ]; do
  case "$1" in
    --pilot) MODE="pilot" ;;
    --full)  MODE="full" ;;
    --tool)  shift; TOOL="${1:-xtts_v2}" ;;
    -h|--help) echo "usage: $0 [--pilot|--full] [--tool xtts_v2|tortoise]"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
  shift
done

case "$TOOL" in
  xtts_v2)  OUT_DIR="${OUT}/xtts_v2" ;;
  tortoise) OUT_DIR="${OUT}/heldout_tool" ;;
  *) echo "unknown tool: $TOOL" >&2; exit 1 ;;
esac

echo "generation mode=$MODE tool=$TOOL -> $OUT_DIR"
echo "(driver: src/data/spoof_generation.py; provide a clone-jobs manifest built"
echo " from select_reference_speakers + carved speaker pools + extracted transcripts)"

# The actual generation is driven from Python so all logic stays in src/ behind
# a config. Example entry point (implemented as jobs are prepared):
#   python -m src.data.spoof_generation --mode "$MODE" --tool "$TOOL" --out "$OUT_DIR"
echo "[03_generate_spoofs] prepare the clone-jobs manifest, then invoke the python driver."
