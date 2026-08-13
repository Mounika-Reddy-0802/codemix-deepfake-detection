# Quantifying and Closing the Code-Mixing Generalisation Gap in Audio Deepfake Detection

[![CI](https://github.com/Mounika-Reddy-0802/codemix-deepfake-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Mounika-Reddy-0802/codemix-deepfake-detection/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/phase-week%203%20complete-brightgreen)

Fine-tune wav2vec2 / WavLM on **ASVspoof 2019 LA (English — the only Stage-1
training corpus)**, measure the English → monolingual Hindi/Tamil → code-mixed
Hinglish generalisation gap under a **channel-matched telephony protocol**, close
part of the gap with **LoRA**, and demonstrate it live: a cloned voice on a phone
call triggers a beep in the receiver's ear, a dashboard alert, and an SMS.

> B.Tech final-year capstone. 12-week plan (`PROJECT_PLAN_12_WEEKS.md`);
> tech stack, folder structure, and dataset usage matrix in
> `TECH_STACK_AND_FOLDER_STRUCTURE.md`.

## Pipeline

```mermaid
flowchart LR
    subgraph data["data (L)"]
        raw[ASVspoof19 LA / MUCS / HiACC-adult] --> pre[preprocess:<br/>16k mono, VAD, norm, segment]
        pre --> chan[channel sim:<br/>8k, G.711/AMR, SNR]
        gen[XTTS-v2 clones + held-out Tortoise] --> chan
    end
    subgraph model["model (M)"]
        chan --> ds[manifest + dataset<br/>speaker-disjoint] --> det[wav2vec2/WavLM<br/>+ attentive pooling + MLP]
        det --> gap[gap matrix:<br/>EN → HI/TA → code-mixed]
        gap --> lora[LoRA adaptation]
    end
    subgraph demo["live demo (SK)"]
        det --> stream[streaming inference] --> call[WebRTC now / Twilio Wk6:<br/>beep + dashboard + SMS]
    end
```

## Team

| Code | Name | Role (primary ownership) |
|------|------|--------------------------|
| **L**  | M. Lahari (D006) | Data Lead — corpora, preprocessing, channel simulation, spoof generation, dataset docs, ethics/licence compliance |
| **M**  | S. Mounika Reddy (D032) | Model Lead — training pipeline, baseline, gap matrix, LoRA, ablations, experiment tracking |
| **SK** | K. Sai Krishna Reddy (D034) | Systems/Demo Lead — Twilio live-call pipeline, streaming inference, ONNX, dashboard, Gradio demo, deployment |

## The golden rule (do not break)

**Stage-1 trains on ASVspoof 2019 LA (English) only.** IndicSynth, IndicTTS-Deepfake,
IndicVoices, HiACC, and the Tortoise clones are **evaluation-only** and must never
enter a training manifest. HiACC **child** audio is excluded from the pipeline
**entirely** — never bonafide, never a cloning reference. Enforced by
`tests/test_splits.py`. Full rationale in `CLAUDE.md` and `docs/ethics/`.

## Repository layout

See `TECH_STACK_AND_FOLDER_STRUCTURE.md` Section 2 for the annotated tree. Key
points: `data/`, `checkpoints/`, and secrets are gitignored; notebooks contain
no pipeline logic (anything producing a paper number lives in `src/` behind a
config); one shared inference module powers both demos.

## Setup

```bash
# 1. Clone, then install the git hooks (strips any stray attribution trailer)
sh scripts/install_hooks.sh          # or: git config core.hooksPath githooks

# 2. Python environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
pip install -r requirements.txt
#    ffmpeg must be on PATH for AMR-NB channel simulation (Week 2).

# 3. Secrets
cp .env.example .env                  # fill in W&B / HF / (Twilio, from Week 6)
#    Git identities + PATs live in .env.git (gitignored) — see below.

# 4. Sanity check (Week 1 deliverable)
#    open notebooks/env_check.ipynb  -> loads wav2vec2-base / XLSR-53 / WavLM-base
```

### Git identity / branching (project rule)

Each member commits under their own GitHub identity. **One branch per member per
week**, named `week<N>-<name>-<work-topic>` (the topic is the work, not the role).
Branches fork from `dev`, and the weekly PR goes to **`dev`, never `main`**; a
teammate reviews and merges, and `dev` → `main` happens once per completed week
with a `week<N>-complete` tag. Full rules — commit style, PR template, reviewer
duties, the ethics and download gates — in **`GIT_RULES.md`**.

## Accounts, compute and dataset links (W3-T6)

Run the machine-checkable half with:

```bash
python -m src.utils.env_check              # offline: secrets present, packages, GPU
python -m src.utils.env_check --network    # also verifies the HF token authenticates
python -m src.utils.env_check --encoders   # also loads the three SSL encoders
```

Everything below needs a human to confirm and a link pasted in. **Data and
checkpoints never enter git** — record the Kaggle Dataset / Drive link plus the
SHA-256 hash instead.

| Resource | Owner | Link | Verified |
|---|---|---|---|
| Kaggle account 1 (GPU, 30 h/wk) | M | _paste_ | ☐ |
| Kaggle account 2 (GPU, 30 h/wk) | L | _paste_ | ☐ |
| Kaggle account 3 (GPU, 30 h/wk) | SK | _paste_ | ☐ |
| W&B project | M | _paste_ | ☐ |
| Hugging Face token owner | M | _paste_ | ☐ |
| IndicVoices (gated — terms accepted) | M | _paste_ | ☐ |
| IndicTTS-Deepfake (gated) | M | _paste_ | ☐ |
| ASVspoof 2019 LA (S1 anchor) + SHA-256 | L | _paste_ | ☐ |
| MUCS 2021 Hindi-English + SHA-256 | L | _paste_ | ☐ |
| HiACC adult subset + SHA-256 | L | _paste_ | ☐ |
| HF Spaces placeholder | SK | _paste_ | ☐ |

## Live-call demo — Twilio timing

Weeks 1–6 use the **free WebRTC (`aiortc`) harness** in `live_call/webrtc_harness/`.
**The Twilio trial is activated in Week 7** (W7-T4), so its 30-day window covers
Weeks 7–10 including the midterm demo. See `live_call/README.md`.

## Status

Planning is governed by **`PROJECT_PLAN_V2_AFFECTDF.md`** (the AffectDF-anchored
v2 plan). It supersedes `PROJECT_PLAN_12_WEEKS.md`, which is kept for history only.

**Weeks 1–2 complete.** Week 1: bootstrap, download scripts, eval-only streaming
loaders, WebRTC harness, ethics/licence templates. Week 2: preprocessing, channel
simulation, dataset + metrics modules, speaker-pool carving.

**Week 3 in progress** — clearing the human gates (PR reviews, mentor ethics
sign-off, downloads), AffectDF positioning and attack taxonomy, speaker selection.
Task log in `docs/progress.md`; blockers in `Works updates.md`; decisions in
`docs/problems_and_decisions.md`.

> The mentor ethics note was **signed on 12 Aug 2026**, so the generation gate is
> open (`python -m src.data.ethics_gate` exits 0). The signed PDF is gitignored —
> a fresh clone will not have it, and the gate will refuse until it is copied into
> `docs/ethics/` by hand.

## Honest limitations

- **No paper numbers yet.** Through Week 3 the repo is pipeline + scaffolding.
  Heavy paths (XTTS/Tortoise generation, wav2vec2 training) are validated by lint,
  compile, and pure-logic unit tests; end-to-end runs need the downloaded corpora,
  a GPU, and the mentor ethics sign-off (human steps in `Works updates.md`).
- **CI is intentionally light** (ruff + pytest + numpy/pandas). Tests needing
  torch/torchaudio/librosa skip in CI and run in the full environment.
- **AMR-NB needs ffmpeg**; without it the channel sim falls back to G.711 μ-law.
- The baseline EER gate (defensible published range) is a Week-5 checkpoint — until
  then, no cross-lingual gap number should be trusted.

## Licences

Per-corpus licence table in `docs/licences.md`. Note IndicSynth is CC-BY-NC-4.0
(research use) and the HiACC child-data exclusion. Code licence: TBD by the team.
