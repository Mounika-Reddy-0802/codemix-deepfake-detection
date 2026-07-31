#!/usr/bin/env bash
# =============================================================================
# 01_download_data.sh  --  corpus download + verification (Week 1, owner L)
#
# Downloads and verifies the corpora the project actually keeps on disk:
#   1. ASVspoof 2019 LA   (Edinburgh DataShare)  -- the ONLY Stage-1 train corpus
#   2. MUCS 2021 Hindi-English  (OpenSLR 104)     -- train + test (Bengali SKIPPED)
#   3. HiACC              (Zenodo 15551669)       -- ADULT subset kept; CHILD folders
#                                                    quarantined immediately on extract
#
# The three eval-only Indic corpora (IndicSynth, IndicTTS-Deepfake, IndicVoices)
# are Hugging Face STREAMING only -- never bulk-downloaded here. Their setup lives
# in src/data/download.py. ASVspoof 2021 DF is intentionally SKIPPED (34.5 GB).
#
# SAFETY: this script does NOTHING by default. Multi-GB pulls start only when you
# pass --run (i.e. when the team says "run downloads now"). Recommended launch:
#
#   nohup bash scripts/01_download_data.sh --run > data/raw/logs/download_main.log 2>&1 &
#
# Each corpus streams to its own resumable (wget -c) log in data/raw/logs/.
# =============================================================================
set -euo pipefail

# --- paths (relative to repo root; script is run from repo root) --------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${REPO_ROOT}/data/raw"
LOG_DIR="${RAW_DIR}/logs"
ASV_DIR="${RAW_DIR}/asvspoof2019_LA"
MUCS_DIR="${RAW_DIR}/mucs2021"
HIACC_DIR="${RAW_DIR}/hiacc"
HIACC_EXCLUDED="${HIACC_DIR}/_EXCLUDED_children"

# --- verified source URLs (dataset matrix, checked 21 Jul 2026) ---------------
# ASVspoof 2019 LA partition (handle 10283/3336). The LA.zip bitstream:
ASV_LA_URL="https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip?sequence=3&isAllowed=y"
ASV_LA_ZIP="${ASV_DIR}/LA.zip"
ASV_LA_APPROX_BYTES=$((23 * 1024 * 1024 * 1024))   # ~23-24 GB, sanity check only

# MUCS 2021 subtask-2 code-switching, Hindi-English only (skip Bengali-English):
MUCS_TRAIN_URL="https://www.openslr.org/resources/104/Hindi-English_train.tar.gz"
MUCS_TEST_URL="https://www.openslr.org/resources/104/Hindi-English_test.tar.gz"
MUCS_TRAIN_TGZ="${MUCS_DIR}/Hindi-English_train.tar.gz"
MUCS_TEST_TGZ="${MUCS_DIR}/Hindi-English_test.tar.gz"

# HiACC lives on Zenodo record 15551669. The exact asset filename + md5 are read
# from the Zenodo REST API at runtime so we never hard-code a guessed filename.
HIACC_ZENODO_API="https://zenodo.org/api/records/15551669"

# Pattern used to identify CHILD speaker folders after extraction. HARD ETHICS
# RULE: HiACC child audio is never used for anything. VERIFY this pattern against
# the HiACC documentation before trusting the auto-quarantine (see warning below).
HIACC_CHILD_PATTERN='(?i)child|kid|minor'

# -----------------------------------------------------------------------------
log()  { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
warn() { printf '[%s] WARNING: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; }
die()  { printf '[%s] ERROR: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; exit 1; }

need_cmd() { command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"; }

# md5 helper that works on Linux (md5sum) and macOS (md5 -q)
md5_of() {
  if command -v md5sum >/dev/null 2>&1; then md5sum "$1" | awk '{print $1}';
  elif command -v md5 >/dev/null 2>&1;    then md5 -q "$1";
  else warn "no md5 tool available; skipping checksum for $1"; echo ""; fi
}

# Resumable download to a target path, logging to its own file in LOG_DIR.
# Usage: fetch <url> <outfile> <logname>
fetch() {
  local url="$1" out="$2" logname="$3"
  local logfile="${LOG_DIR}/${logname}.log"
  mkdir -p "$(dirname "$out")"
  log "downloading -> ${out}  (log: ${logfile})"
  # -c resume, --tries keeps retrying flaky college wifi, timestamping off.
  wget -c --tries=20 --timeout=60 --waitretry=10 -O "$out" "$url" >>"$logfile" 2>&1
}

size_of() { stat -c '%s' "$1" 2>/dev/null || stat -f '%z' "$1" 2>/dev/null || echo 0; }

check_min_size() {
  local f="$1" approx="$2"
  local got; got="$(size_of "$f")"
  if [ "$got" -lt $((approx / 2)) ]; then
    warn "$(basename "$f") is ${got} bytes, far below the expected ~${approx}. Download may be incomplete -- re-run with --run to resume."
  else
    log "$(basename "$f") size ${got} bytes (expected ~${approx}) OK"
  fi
}

# =============================================================================
# 1. ASVspoof 2019 LA
# =============================================================================
dl_asvspoof2019_la() {
  log "=== ASVspoof 2019 LA (Stage-1 TRAIN corpus + English eval split) ==="
  fetch "$ASV_LA_URL" "$ASV_LA_ZIP" "asvspoof2019_LA"
  check_min_size "$ASV_LA_ZIP" "$ASV_LA_APPROX_BYTES"
  log "extracting LA.zip ..."
  ( cd "$ASV_DIR" && unzip -n -q LA.zip ) || warn "unzip reported issues; inspect $ASV_DIR"
  log "ASVspoof 2019 LA ready under ${ASV_DIR}/LA"
  log "NOTE: ASVspoof 2021 DF is intentionally SKIPPED (34.5 GB, busts free storage)."
}

# =============================================================================
# 2. MUCS 2021 Hindi-English (OpenSLR 104) -- Bengali-English deliberately skipped
# =============================================================================
dl_mucs() {
  log "=== MUCS 2021 Hindi-English (OpenSLR 104) -- skipping Bengali-English ==="
  fetch "$MUCS_TRAIN_URL" "$MUCS_TRAIN_TGZ" "mucs_train"
  fetch "$MUCS_TEST_URL"  "$MUCS_TEST_TGZ"  "mucs_test"
  check_min_size "$MUCS_TRAIN_TGZ" $((7 * 1024 * 1024 * 1024))   # ~7.3 GB
  check_min_size "$MUCS_TEST_TGZ"  $((400 * 1024 * 1024))        # ~443 MB
  log "extracting MUCS archives ..."
  ( cd "$MUCS_DIR" && tar -xzf Hindi-English_train.tar.gz && tar -xzf Hindi-English_test.tar.gz )
  log "MUCS Hindi-English ready under ${MUCS_DIR}"
}

# =============================================================================
# 3. HiACC (Zenodo 15551669) -- adult subset kept, CHILD folders quarantined
# =============================================================================
dl_hiacc() {
  log "=== HiACC (Zenodo 15551669) -- ADULT subset only ==="
  need_cmd curl
  need_cmd python3 || need_cmd python
  local py; py="$(command -v python3 || command -v python)"

  log "querying Zenodo API for the exact file URL + md5 ..."
  local meta; meta="$(curl -fsSL "$HIACC_ZENODO_API")" || die "could not reach Zenodo API"
  # Parse the first file entry's download link, name and md5 checksum.
  local url name md5
  url="$(printf '%s' "$meta"  | "$py" -c "import sys,json;f=json.load(sys.stdin)['files'][0];print(f['links']['self'])")"
  name="$(printf '%s' "$meta" | "$py" -c "import sys,json;f=json.load(sys.stdin)['files'][0];print(f['key'])")"
  md5="$(printf '%s' "$meta"  | "$py" -c "import sys,json;f=json.load(sys.stdin)['files'][0];print(f.get('checksum','').replace('md5:',''))")"

  local out="${HIACC_DIR}/${name}"
  fetch "$url" "$out" "hiacc"

  if [ -n "$md5" ]; then
    local got; got="$(md5_of "$out")"
    if [ -n "$got" ] && [ "$got" != "$md5" ]; then
      die "HiACC md5 mismatch (expected ${md5}, got ${got}) -- delete ${out} and re-run"
    fi
    log "HiACC md5 verified (${md5})"
  fi

  log "extracting HiACC ..."
  case "$name" in
    *.zip)      ( cd "$HIACC_DIR" && unzip -n -q "$name" ) ;;
    *.tar.gz)   ( cd "$HIACC_DIR" && tar -xzf "$name" ) ;;
    *)          warn "unknown HiACC archive type: ${name}; extract it manually" ;;
  esac

  quarantine_hiacc_children
}

# Move any CHILD speaker folder out of the pipeline BEFORE anything can use it.
quarantine_hiacc_children() {
  mkdir -p "$HIACC_EXCLUDED"
  log "scanning HiACC for child-speaker folders (pattern: ${HIACC_CHILD_PATTERN}) ..."
  local moved=0
  # Find directories matching the child pattern, excluding the quarantine dir itself.
  while IFS= read -r d; do
    [ -z "$d" ] && continue
    case "$d" in "$HIACC_EXCLUDED"*) continue ;; esac
    log "  quarantining: ${d}"
    mv "$d" "$HIACC_EXCLUDED"/ && moved=$((moved + 1))
  done < <(find "$HIACC_DIR" -mindepth 1 -type d | grep -P "$HIACC_CHILD_PATTERN" || true)

  write_exclusion_readme "$moved"

  if [ "$moved" -eq 0 ]; then
    warn "No child folders matched the pattern. HiACC's adult/child layout MUST be"
    warn "verified by hand against the dataset documentation -- do NOT assume the"
    warn "whole extract is adult. If the split is by manifest/filename rather than"
    warn "folder, update HIACC_CHILD_PATTERN or quarantine manually before any use."
  else
    log "quarantined ${moved} child folder(s) into ${HIACC_EXCLUDED}"
  fi
}

write_exclusion_readme() {
  local moved="$1"
  cat > "${HIACC_EXCLUDED}/README.md" <<'EOF'
# HiACC child audio -- QUARANTINED, NEVER USED

The utterances in this folder are HiACC **child** speaker recordings.

**Hard project rule (ethics-critical):** HiACC child audio is excluded from the
pipeline entirely. It is:

- never used as bonafide in any evaluation column,
- never used as a voice-cloning reference for spoof generation,
- never referenced by any manifest (train, dev, or eval).

Only the HiACC **adult** subset (3,318 utterances) enters the project, as an
eval-only code-mixed bonafide column. This exclusion is stated explicitly in the
ethics/licence note (docs/ethics/). Do not move anything out of this folder.
EOF
  log "wrote ${HIACC_EXCLUDED}/README.md (moved ${moved} folder(s))"
}

# =============================================================================
# dry-run plan (default) vs real run (--run)
# =============================================================================
print_plan() {
  cat <<EOF
DRY RUN -- nothing downloaded. This script would fetch, into data/raw/:

  ASVspoof 2019 LA   ~23-24 GB   ${ASV_LA_URL%%\?*}
  MUCS train         ~7.3 GB     ${MUCS_TRAIN_URL}
  MUCS test          ~443 MB     ${MUCS_TEST_URL}
  HiACC              ~532 MB     (URL + md5 resolved from ${HIACC_ZENODO_API})

  SKIPPED on purpose: ASVspoof 2021 DF (34.5 GB); MUCS Bengali-English.
  Eval-only Indic corpora are HF streaming (src/data/download.py), not here.

HiACC child folders are quarantined into data/raw/hiacc/_EXCLUDED_children/
immediately after extraction.

To actually download (only when the team says "run downloads now"):

  mkdir -p data/raw/logs
  nohup bash scripts/01_download_data.sh --run > data/raw/logs/download_main.log 2>&1 &

Or one corpus at a time:  bash scripts/01_download_data.sh --run --only hiacc
EOF
}

main() {
  local do_run=0 only="all"
  while [ $# -gt 0 ]; do
    case "$1" in
      --run) do_run=1 ;;
      --only) shift; only="${1:-all}" ;;
      -h|--help) print_plan; exit 0 ;;
      *) die "unknown argument: $1 (use --run, --only <name>, or --help)" ;;
    esac
    shift
  done

  if [ "$do_run" -eq 0 ]; then
    print_plan
    exit 0
  fi

  need_cmd wget
  mkdir -p "$LOG_DIR" "$ASV_DIR" "$MUCS_DIR" "$HIACC_DIR"
  log "RUN mode: downloads starting. Logs in ${LOG_DIR}"

  case "$only" in
    all)      dl_asvspoof2019_la; dl_mucs; dl_hiacc ;;
    asvspoof) dl_asvspoof2019_la ;;
    mucs)     dl_mucs ;;
    hiacc)    dl_hiacc ;;
    *)        die "unknown --only target: ${only} (asvspoof|mucs|hiacc|all)" ;;
  esac

  log "all requested downloads complete."
}

main "$@"
