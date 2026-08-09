# Week 2 — preprocessing, channel simulation, dataset + metrics

**Goal:** the pipeline that turns raw corpora into model-ready, channel-matched
data, plus the evaluation machinery.

## What shipped

| Owner | Deliverable |
|-------|-------------|
| L | `preprocess` (resample 16 kHz mono → silero VAD → loudness norm → 2–10 s segments); `channel_sim` (16 k → 8 k → **G.711 μ-law** / AMR-NB → **noise @ SNR** → 16 k); `audio_utils`; `02_preprocess_all.sh` |
| M | `dataset` (manifest-driven Dataset + variable-length collate); `build_manifests` (schema + **speaker-disjoint splits** + anti-leakage checklist); `metrics` (EER / AUC / F1 + bootstrap CIs) |
| SK | adult-speaker selection (child audio excluded); XTTS-v2 clone driver with per-file metadata + held-out-tool firewall |

## Channel-matched protocol

Every eval clip passes through the same narrowband telephony chain the live Twilio
demo delivers, so the detector is tested under its deployment condition. G.711
μ-law + SNR mixing are exact pure-numpy implementations (unit-tested); AMR-NB uses
ffmpeg when present.

## Honest limitations

- G.711 + SNR are exact and tested; **AMR-NB depends on ffmpeg** and falls back to
  G.711 when it is absent.
- The dataset/metrics logic is unit-tested on synthetic inputs; end-to-end runs
  await downloaded corpora.

## Decisions

- P-001/P-002 carried forward (English-only Stage-1; HiACC child exclusion).
- Anti-leakage is enforced mechanically by `tests/test_splits.py` before any
  training launch.
