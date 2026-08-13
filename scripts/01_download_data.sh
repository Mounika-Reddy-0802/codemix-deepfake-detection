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

# DATA_ROOT lets the corpora live OUTSIDE the repo. This matters when the repo
# sits inside a cloud-synced folder (OneDrive / Dropbox / Google Drive): those
# clients ignore .gitignore and will happily try to upload 60+ GB of audio.
#   DATA_ROOT=/c/dfdata bash scripts/01_download_data.sh --run
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
RAW_DIR="${DATA_ROOT}/raw"
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
# POSIX ERE, used with `grep -Ei`. The `-i` flag supplies case-insensitivity --
# an inline `(?i)` is a PCRE construct and matches NOTHING under -E, which is the
# second half of why the 12 Aug quarantine sweep found no child folders.
HIACC_CHILD_PATTERN='child|kid|minor'

# -----------------------------------------------------------------------------
log()  { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
warn() { printf '[%s] WARNING: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; }
die()  { printf '[%s] ERROR: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; exit 1; }

need_cmd() { command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"; }

# Pick a Python that actually RUNS.
# On Windows, `python3` is usually the Microsoft Store alias stub: it resolves on
# PATH and prints "Python was not found; run without arguments to install from the
# Microsoft Store" instead of executing anything. `command -v python3` therefore
# selects a broken interpreter, which is exactly how the HiACC step failed on
# 12 Aug 2026. Checking the interpreter's OUTPUT is the only reliable test -- the
# stub cannot print 42.
pick_python() {
  local cand
  for cand in python python3 py; do
    command -v "$cand" >/dev/null 2>&1 || continue
    if [ "$("$cand" -c 'print(42)' 2>/dev/null)" = "42" ]; then
      printf '%s' "$cand"
      return 0
    fi
  done
  return 1
}

# md5 helper that works on Linux (md5sum) and macOS (md5 -q)
md5_of() {
  if command -v md5sum >/dev/null 2>&1; then md5sum "$1" | awk '{print $1}';
  elif command -v md5 >/dev/null 2>&1;    then md5 -q "$1";
  else warn "no md5 tool available; skipping checksum for $1"; echo ""; fi
}

# Resumable download to a target path, logging to its own file in LOG_DIR.
# Uses wget when present, else curl. Git Bash on Windows ships curl but NOT wget,
# so requiring wget would block the whole pipeline on the team's own machines.
# Usage: fetch <url> <outfile> <logname>
fetch() {
  local url="$1" out="$2" logname="$3"
  local logfile="${LOG_DIR}/${logname}.log"
  mkdir -p "$(dirname "$out")"
  log "downloading -> ${out}  (log: ${logfile})"
  if command -v wget >/dev/null 2>&1; then
    # -c resume, --tries keeps retrying flaky college wifi, timestamping off.
    wget -c --tries=20 --timeout=60 --waitretry=10 -O "$out" "$url" >>"$logfile" 2>&1
  else
    # -C - resumes a partial file, --retry rides out drops, -L follows the
    # redirects both Edinburgh DataShare and Zenodo use.
    curl -L -C - --retry 20 --retry-delay 10 --connect-timeout 60 \
      -o "$out" "$url" >>"$logfile" 2>&1
  fi
}

# Refuse to start a multi-GB pull that cannot possibly fit.
check_disk_space() {
  local need_gb="$1"
  local avail_gb
  avail_gb="$(df -BG "$DATA_ROOT" 2>/dev/null | awk 'NR==2 {gsub(/[A-Za-z]/,"",$4); print $4}')"
  [ -z "$avail_gb" ] && { warn "could not determine free space; continuing"; return 0; }
  log "free space at ${DATA_ROOT}: ${avail_gb} GB (need ~${need_gb} GB)"
  if [ "$avail_gb" -lt "$need_gb" ]; then
    die "only ${avail_gb} GB free but ~${need_gb} GB needed. Free space, or set DATA_ROOT to another drive."
  fi
}

# Cloud-sync clients do not honour .gitignore. Downloading tens of GB into a
# synced folder uploads all of it and usually blows the account quota.
warn_if_cloud_synced() {
  case "$DATA_ROOT" in
    *OneDrive*|*"One Drive"*|*Dropbox*|*"Google Drive"*|*GoogleDrive*)
      warn "DATA_ROOT is inside a cloud-synced folder:"
      warn "    ${DATA_ROOT}"
      warn "The sync client will try to upload every GB downloaded here."
      warn "Either pause syncing, exclude this folder, or re-run with e.g.:"
      warn "    DATA_ROOT=/c/dfdata bash scripts/01_download_data.sh --run"
      ;;
  esac
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
  local py
  py="$(pick_python)" || die "no working Python found (tried python, python3, py)"
  log "using interpreter: $(command -v "$py")"

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
#
# HISTORY (12 Aug 2026): this used `grep -P`, which on Git Bash dies with
# "grep: -P supports only unibyte and UTF-8 locales". grep exited non-zero, the
# `|| true` swallowed it, and the function cheerfully reported "moved 0 folder(s)"
# on a corpus whose `Corpus/children/` directory was sitting right there. The
# quarantine had silently done nothing. Now it uses the portable `grep -Ei` and,
# far more importantly, VERIFIES the result instead of trusting it: if any
# child-looking directory is still outside quarantine afterwards, the script dies
# rather than letting the pipeline continue.
quarantine_hiacc_children() {
  mkdir -p "$HIACC_EXCLUDED"
  log "scanning HiACC for child-speaker folders (pattern: ${HIACC_CHILD_PATTERN}) ..."
  local moved=0
  # Find directories matching the child pattern, excluding the quarantine dir itself.
  while IFS= read -r d; do
    [ -z "$d" ] && continue
    case "$d" in "$HIACC_EXCLUDED"*) continue ;; esac
    # A nested match (…/children/audio) moves with its parent, so by the time the
    # loop reaches it the path is already gone. Skip rather than error.
    [ -d "$d" ] || continue
    log "  quarantining: ${d}"
    mv "$d" "$HIACC_EXCLUDED"/ && moved=$((moved + 1))
    # -Ei is POSIX and works everywhere; -P needs a PCRE build and a UTF-8 locale.
  done < <(find "$HIACC_DIR" -mindepth 1 -type d | grep -Ei "$HIACC_CHILD_PATTERN" || true)

  write_exclusion_readme "$moved"

  # VERIFY, do not trust. The whole point of the 12 Aug failure is that a broken
  # scan is indistinguishable from a clean corpus unless you check afterwards.
  local leftover
  leftover="$(find "$HIACC_DIR" -mindepth 1 -type d \
    | grep -Ei "$HIACC_CHILD_PATTERN" \
    | grep -v "^${HIACC_EXCLUDED}" || true)"
  if [ -n "$leftover" ]; then
    warn "child-looking directories STILL outside quarantine after the sweep:"
    printf '%s\n' "$leftover" | while IFS= read -r d; do warn "    ${d}"; done
    die "quarantine failed -- refusing to continue. Move these into ${HIACC_EXCLUDED} by hand."
  fi

  if [ "$moved" -eq 0 ]; then
    warn "No child folders matched the pattern. HiACC's adult/child layout MUST be"
    warn "verified by hand against the dataset documentation -- do NOT assume the"
    warn "whole extract is adult. If the split is by manifest/filename rather than"
    warn "folder, update HIACC_CHILD_PATTERN or quarantine manually before any use."
  else
    log "quarantined ${moved} child folder(s) into ${HIACC_EXCLUDED}"
  fi

  log "run the audit before using this corpus:"
  log "    python -m src.data.quarantine --root ${HIACC_DIR}"
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

  if ! command -v wget >/dev/null 2>&1 && ! command -v curl >/dev/null 2>&1; then
    die "need either wget or curl on PATH"
  fi
  mkdir -p "$LOG_DIR" "$ASV_DIR" "$MUCS_DIR" "$HIACC_DIR"

  warn_if_cloud_synced
  # Archives + extracted copies roughly double the download size.
  case "$only" in
    asvspoof) check_disk_space 50 ;;
    mucs)     check_disk_space 17 ;;
    hiacc)    check_disk_space 2  ;;
    *)        check_disk_space 68 ;;
  esac

  log "RUN mode: downloads starting. Logs in ${LOG_DIR}"
  log "data root: ${DATA_ROOT}"

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
