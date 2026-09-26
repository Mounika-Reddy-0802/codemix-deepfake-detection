# Reproduce

Exact commands to regenerate every number this project currently claims.

**Two halves.** Sections 1-5 rebuild the corpus, the channel protocol, the spoof
pilot and the transliteration audit. Section 6 rebuilds every table and figure in
the paper from the trained checkpoints.

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

## 6. Paper tables and figures

Every number in the paper comes from one of the artefacts below. Training needs a
GPU; the reporting steps are pure numpy/pandas and run anywhere.

`$DATA_ROOT` for the code-mixed work is the **level-normalised** bundle, not the
first build: `lora_bundle_norm` for clean audio and `lora_bundle_norm_ch20` for the
G.711 @ 20 dB condition (Sec. 2 and `docs/results/bundle_normalisation_v1.md`).

### 6.1 S1, English only

```bash
bash scripts/04_train_baseline.sh            # configs/train_stage1_run.yaml
python -m src.training.evaluate     --checkpoint checkpoints/baseline/best.pt     --manifest data/manifests/asvspoof_eval.csv --device cuda     --out results/s1_norm/stage1__asvspoof_eval.json
```

Every adapter starts from this checkpoint. To confirm a checkpoint is the one an
adapter was built on, compare a frozen encoder tensor between the two: they must be
bit-identical (the deployed adapter stores its base weights under `.base.weight`).

### 6.2 S2 adapters and S3 native models

Four training configs each, identical but for data mix and condition:

```bash
for cfg in train_lora_norm_clean train_lora_norm_channel            train_lora_norm_rvc_clean train_lora_norm_rvc_channel            train_s3_native_clean train_s3_native_channel            train_s3_native_rvc_clean train_s3_native_rvc_channel; do
    python -m src.training.train --config configs/$cfg.yaml
done
```

Both W10 runs were done on Kaggle T4s from a clean clone:
`notebooks/kaggle_w10_retrain_normalised.ipynb` (S2) and
`notebooks/kaggle_w10_s3_native.ipynb` (S3). `src/training/config_guard.py` refuses
any config that puts code-mixed rows into a run that is neither LoRA nor
`s3_native`, and `tests/test_splits.py` refuses a Tortoise clip in any training
manifest. Both run before a launch.

### 6.3 Scoring

One command per (model, set, condition); per-clip scores go beside each JSON.

```bash
python -m src.training.evaluate --checkpoint <ckpt>     --manifest data/manifests/<manifest>.csv     --data-root $DATA_ROOT --device cuda     --out results/<run>__<set>.json --scores-out results/<run>__<set>_scores.csv     --partial results/_partial_<set>.csv
```

| Set | Clean manifest | Channel manifest |
|---|---|---|
| Eval pool (XTTS) | `codemix_eval.csv` | `codemix_eval_channel20.csv` |
| CM04 (Tortoise) | `score_cm04_norm.csv` | `score_cm04_norm_channel20.csv` |
| RVC test | `rvc_holdout_test.csv` | `rvc_holdout_test_channel20.csv` |
| English | `asvspoof_eval.csv` | same clips |

`--partial` is a resume cache: a run killed by a driver reset or a reboot continues
from where it stopped when you repeat the command. Delete the cache once the
`_scores.csv` exists.

### 6.4 Shortcut floors, then the tables

The floors must exist before any EER is read, one per set and condition:

```bash
python -m src.data.lowlevel_cue --help     # fit on adaptation pool, score eval pool
python -m src.reporting.ablation           # experiments/results/ablations.json
python -m src.reporting.systems            # systems_s1_s2_s3.json + reverse degradation
python -m src.reporting.figures --table    # experiments/figures/*.png
```

`src.reporting.systems` reads `results/s1_norm/` for S1's code-mixed cells and
leaves them out if that directory is incomplete, rather than filling them with
pre-normalisation numbers (P-032).

### 6.5 Live-call operating point and the probes

```bash
python -m src.inference.calibrate --help                    # threshold + ladder (P-031)
python -m src.inference.interpretability     --config configs/interpretability.yaml                  # what the model responds to
```

Outputs: `configs/threshold.yaml` with
`experiments/results/threshold_calibration.json`, and
`experiments/results/interpretability.json`
(`docs/results/interpretability_v1.md`).

### 6.6 What each paper table is built from

| Paper element | Artefact |
|---|---|
| S1 gap table | `results/s1_norm/stage1__*.json` |
| Shortcut floors | `experiments/results/lowlevel_cue_check_*.json` |
| Main systems table | `experiments/results/systems_s1_s2_s3.json` |
| RVC ablation | `experiments/results/ablations.json` |
| English retention | `experiments/results/s3_reverse_degradation.json` |
| Probe subsection | `experiments/results/interpretability.json` |
| Live-call behaviour | `experiments/results/threshold_calibration.json` |

### 6.7 Still open

- [ ] Checkpoint SHA-256s and the commit hash per table, recorded at the freeze.
- [ ] Checkpoints are in a private Kaggle dataset (`codemix-w10-results`); only the
      deployed adapter is on the demo machine.
