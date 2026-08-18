# Reproduce

Exact commands to regenerate every number this project currently claims.

**Two halves.** The first is reproducible **today** — corpus counts, the channel
protocol, the spoof pilot, the transliteration audit. The second is the paper's
tables and figures, which do not exist yet; that half is finalised in Week 10
(owner M) once there are checkpoints to archive.

Nothing here needs a GPU except spoof generation and training.

---

## 0. Environment

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
# CUDA build FIRST, so the CPU wheels in requirements.txt do not win.
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Python 3.10–3.13. **Do not upgrade torch past 2.8** — from 2.9 torchaudio routes
audio IO through `torchcodec`, which fails against FFmpeg 8/9 and breaks XTTS.
`ffmpeg` must be on PATH or the AMR-NB channel condition silently falls back to
G.711 and is never actually tested.

Verify before anything expensive:

```bash
python -m src.utils.device      # must print a GPU name, not "cpu", for generation
python -m pytest -q             # 410 passed, 2 skipped
```

---

## 1. Corpora → the index (claims: 52,825 / 520 and 3,318 / 24)

```bash
# Download (~15 GB), or extract archives already on disk — same child quarantine:
bash scripts/01_download_data.sh --run
DATA_ROOT=/c/dfdata bash scripts/01_download_data.sh --extract-only

python -m src.data.corpora --data-root $DATA_ROOT
```

Expected, and **byte-comparable across machines**:

```
MUCS  {"clips": 52825, "speakers": 520, "hours": 89.55, "with_transcript": 52825}
HiACC {"clips": 3318,  "speakers": 24,  "hours": 3.22,  "with_transcript": 3318}
wrote 56143 rows across 544 speakers -> data/manifests/clip_index.csv
```

If these differ, an archive is incomplete — stop, do not generate anything.

`clip_index.csv` is **gitignored** (it holds machine-local absolute paths) and is
regenerated per machine. `speaker_pools.csv` is the opposite: **frozen and
committed**, SHA-256 `f57e0d85…`, and must never be regenerated or the two
machines stop measuring the same split.

## 2. The child-audio exclusion (claim: 1,858 quarantined, 0 reachable)

```bash
python -m src.data.quarantine --root $DATA_ROOT/raw/hiacc
```

Expected: `audio files: 5176 (quarantined: 1858)` — i.e. 3,318 adult + 1,858 child.

The audit **never returns a pass**; a human signs it. It also refuses to overwrite
a report carrying hand-written sections (an incident note, a ticked checkbox, a
signature) — pass `--force` only once that content is preserved elsewhere.

Verify nothing downstream can reach child audio:

```bash
pytest tests/test_splits.py tests/test_preprocess_quarantine.py tests/test_quarantine.py -q
```

## 3. Channel protocol (claims: 17.62 dB / 0.991 and 5.57 dB / 0.885)

```bash
python -m src.data.listening_test --codec g711  --snr-db 20
python -m src.data.listening_test --codec amr_nb --snr-db 20 \
    --out-dir $DATA_ROOT/processed/listening_test_amr \
    --sheet docs/qa/channel_sim_listening_sheet_amr.csv
python -m src.data.channel_qa            # objective 20/20 band-limiting check
```

Measurement alone is not the claim: three people rated 20 clean/channel pairs
(60/60 rows) at telephony 4.0/5 and intelligibility 4.0/5. The **AMR-NB listening
pass is still outstanding** — that column is verified by measurement only and must
not be reported as ear-checked.

## 4. Transliteration audit (claim: 0 unmapped across 56,143)

```bash
python -c "
import pandas as pd
from src.data import transliteration as tr
t = pd.read_csv('data/manifests/clip_index.csv').dropna(subset=['transcript'])['transcript'].astype(str)
print('unmapped:', tr.unmapped_devanagari(t))
print('leaked  :', sum(1 for r in tr.romanise_series(t) if tr.has_devanagari(r)))
"
```

Expected: `unmapped: {}` and `leaked : 0`. Run this before pointing the module at
any new corpus — anything reported is being silently dropped by the safety net.

## 5. The spoof pilot (claim: 40 clips, 0 failures, 13.9 vs 14.4 chars/sec)

Requires the signed ethics note in `docs/ethics/` — it is gitignored, so a fresh
clone will not have it and the gate will refuse.

```bash
python -m src.data.ethics_gate                 # must exit 0
export COQUI_TOS_AGREED=1

# Romanised pack (the decision of P-014), and the matched Devanagari control
python -m src.data.pilot_jobs --data-root $DATA_ROOT \
    --pack-dir $DATA_ROOT/generated/pilot_roman \
    --jobs-out data/manifests/pilot_generation_jobs_roman.csv --romanise

python -m src.data.spoof_generation \
    --jobs $DATA_ROOT/generated/pilot_roman/generation_jobs.csv \
    --pack-dir $DATA_ROOT/generated/pilot_roman \
    --out-dir  $DATA_ROOT/generated/pilot_roman/outputs
```

The control pack is built from the romanised pack's `transcript_source`, so both
render the **same sentences** — that is what makes the A/B a comparison rather than
two unrelated runs. Generation is resumable: re-running skips clips that exist.

Pre-fetch the XTTS weights first (Coqui's downloader has no resume, and omitting
`hash.md5` makes it re-download 1.9 GB) — see `docs/gpu_laptop_setup.md`.

### The rating sheet

```bash
python -m src.data.pilot_rating \
    --roman-pack $DATA_ROOT/generated/pilot_roman \
    --deva-pack  $DATA_ROOT/generated/pilot_deva \
    --stage-dir  $DATA_ROOT/generated/pilot_ab/clips
```

`--stage-dir` is not optional for a blind sheet: without it the pack folder in the
audio path (`pilot_roman/` vs `pilot_deva/`) tells the rater which script they are
hearing. Do not open `docs/qa/pilot_script_answer_key.csv` before rating.

---

## 6. Paper tables and figures — not yet reproducible

These need Stage-1 training, which cannot start: **ASVspoof 2019 LA is not on the
GPU machine**, and it is the only Stage-1 training corpus.

- [ ] Stage-1 baseline (`scripts/04_train_baseline.sh` + `configs/train_baseline.yaml`)
- [ ] Gap matrix (`scripts/05_eval_gap_matrix.sh` + `configs/eval_matrix.yaml`)
- [ ] Stage-3 LoRA (`scripts/06_train_lora.sh` + `configs/train_lora_codemix.yaml`)
- [ ] Figure + table regeneration from archived checkpoints
- [ ] Seeds, checkpoint SHA-256s, and the exact commit per table

Until this section is filled in, **no cross-lingual gap number from this repo
should be quoted** — see the limitations in `README.md`.
