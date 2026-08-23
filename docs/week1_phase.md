# Week 1 — setup, datasets, accounts, ethics

**Goal:** a working repo, reproducible environment, dataset acquisition plan, a
free live-call harness, and the ethics/literature scaffolding.

## What shipped

| Owner | Deliverable |
|-------|-------------|
| M | repo bootstrap: folder structure, `.gitignore`, pinned `requirements.txt`, CI (ruff + pytest), commit-msg hook, the team rules doc; `env_check` notebook (loads wav2vec2-base / XLSR-53 / WavLM-base) |
| L | `01_download_data.sh` (ASVspoof 2019 LA, MUCS Hi-En, HiACC) with HiACC child-folder quarantine; HF streaming loaders for the eval-only Indic corpora; licence register |
| SK | free aiortc WebRTC harness with a 16 kHz PCM tap + the shared `PcmFrameSource` contract (reused by Twilio in Week 6); Twilio design note |
| ALL | related-work template (17 refs, 4 strands) + ethics sign-off requirements |

## Honest limitations

- Multi-GB downloads are gated behind an explicit "run downloads now"; nothing was
  pulled yet.
- Twilio is deliberately **not** activated until Week 6 (30-day trial timing).

## Decisions

- P-001 Stage-1 trains on ASVspoof 2019 LA (English) only — the whole gap-matrix
  measurement depends on it.
- P-002 HiACC child audio is excluded from the pipeline entirely.
