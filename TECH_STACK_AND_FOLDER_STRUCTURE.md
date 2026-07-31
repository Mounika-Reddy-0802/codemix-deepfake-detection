# Tech Stack & Project Folder Structure
## Code-Mixed Audio Deepfake Detection + Live Call Detection (Web App)

---

## 1. Tech Stack

### Research / ML core
| Layer | Choice | Why |
|-------|--------|-----|
| Language | Python 3.10+ | Ecosystem standard |
| DL framework | PyTorch 2.x | Standard for speech research |
| Models | `facebook/wav2vec2-base`, `facebook/wav2vec2-large-xlsr-53`, `microsoft/wavlm-base` via HuggingFace `transformers` | Proposal architecture; validated by ADD 2022 top systems |
| Audio I/O + DSP | `torchaudio`, `librosa`, `soundfile` | Resampling, codecs, feature handling |
| VAD | `silero-vad` | Fast, accurate, free, runs on CPU |
| Channel simulation | `torchaudio` codec transforms (G.711 μ-law), `ffmpeg` (AMR-NB), custom SNR noise-mixing module | Core of the channel-matched protocol |
| Parameter-efficient FT | `peft` (LoRA) | Stage-3 adaptation on free GPU budget |
| Metrics | `scikit-learn` (AUC, F1) + custom EER + `numpy` bootstrap CIs | ASVspoof-standard reporting |
| Experiment tracking | Weights & Biases (free tier) | Compare runs across 3 Kaggle accounts |
| Spoof generation | Coqui **XTTS-v2** (training-side attack), **Tortoise-TTS** (held-out unseen attack) | Open-source, multilingual voice cloning; strict tool separation for unseen-attack split |
| Datasets | See **Section 4: dataset usage matrix** — verified URLs, sizes, licences, and the train vs eval role of every corpus | The single most audit-sensitive part of the project |
| Compute | Kaggle (30 GPU-hrs/wk × 3 accounts), Google Colab | Free tier only, per budget |

### Live call detection system (the wow-factor demo)
| Layer | Choice | Why |
|-------|--------|-----|
| Telephony | **Twilio Programmable Voice: Conference + Media Streams** | Conference connects caller ↔ receiver while Media Streams taps live call audio (8 kHz μ-law) to your websocket server — and the Conference Participant API lets the server play alert audio **to one participant only**. Matches your telephony evaluation protocol exactly |
| Free dev harness (Weeks 1–5) | **WebRTC via `aiortc`** (Python) — browser-to-browser call through your FastAPI server | 100% free, no expiry, full server-side audio access. All streaming-inference and alert code is built and tested here first; Twilio is only swapped in at Week 6. Also the zero-cost live fallback if Twilio fails at the viva |
| Caller client | **Twilio Voice JS SDK** (browser dialer page) | Twilio can't issue Indian numbers to individuals; browser dialer → your US Twilio number avoids international call charges entirely |
| **Alert layer 1 — in-call audio (PRIMARY)** | Conference Participant API plays a beep / short whispered warning into the **receiver's ear only** | The receiver's phone is against their ear during a call — this is the one alert channel they cannot miss. Caller (potential fraudster) hears nothing |
| **Alert layer 2 — live dashboard** | Browser page over websocket | For demo screens and the real-world call-centre-agent framing in the paper |
| **Alert layer 3 — escalation SMS** | Twilio SMS API, fired when high confidence is sustained 10+ s | Timestamped warning waiting on their phone even if all else is ignored. Trial includes 100 SMS to verified numbers. (Report caveat: SMS *into* India in production requires a DLT-registered sender) |
| **Alert layer 4 — post-call summary** | SMS + dashboard log on call end, always | Covers the missed-everything case: voice-fraud harm (transferring money, sharing OTP) usually happens *after* the call, so a warning read 30 s post-hangup still prevents most damage |
| Escalation logic | `verdict_engine` state machine: detect → beep + red dashboard; sustained → repeat beep + SMS; call end → summary | System intensifies instead of firing once — the direct answer to "what if they don't notice?" |
| Backend | **FastAPI** + `websockets` + `uvicorn` | One async Python server handles Twilio webhooks, the media-stream websocket, and dashboard updates |
| Streaming inference | Rolling 4 s window / 2 s hop, exponential smoothing over window verdicts | Stable live verdict instead of flickering per-chunk output |
| Model serving | **ONNX Runtime** (int8 quantised export of the fine-tuned model) | wav2vec2-base inference in well under 1 s per window on CPU — no GPU server needed |
| Receiver dashboard | Plain HTML + JS + websocket (optionally Tailwind via CDN) | Live Real ✅ / Suspicious ⚠️ / Cloned 🚨 indicator, confidence meter, verdict timeline. No React build chain needed |
| Dev tunneling | `ngrok` free tier | Exposes local FastAPI to Twilio webhooks during development |
| Demo hosting | Hugging Face **Spaces** (Gradio upload/record demo) + free-tier VM or persistent Space for the call server | Free; Gradio demo doubles as viva fallback |

### Writing & delivery
| Purpose | Tool |
|---------|------|
| Paper | Overleaf, IEEE conference LaTeX template |
| Report | Institute .docx template |
| Version control | GitHub (private repo, 3 collaborators, branch-per-feature, PR review) |
| Data storage | Kaggle Datasets (private) for processed splits — mountable in notebooks, saves re-uploading |

### Budget map (₹1–2k + Twilio trial)
| Item | Cost |
|------|------|
| Twilio trial (activate **Week 6**, not Week 1) | ₹0 — 75 free voice minutes + 100 free SMS, no credit card, **expires 30 days after signup**; calls/SMS only to verified teammate numbers (fine for dev + demo) |
| Weeks 1–5 development (WebRTC harness) | ₹0 — aiortc is open source |
| Paid upgrade (only if viva falls outside the trial window) | from ₹1–2k: US number ~$1.15/mo + inbound voice ~$0.0085/min + SMS cents each — the budget covers all demo usage several times over |
| Everything else (datasets, models, Kaggle, HF, W&B, ngrok, Overleaf) | ₹0 |

---

## 2. Project Folder Structure

One GitHub monorepo. `data/`, `checkpoints/`, and secrets are gitignored.

```
codemix-deepfake-detection/
│
├── README.md                     # Project overview, setup, links to demo & paper
├── REPRODUCE.md                  # Exact steps to regenerate every paper table
├── requirements.txt              # Pinned Python dependencies
├── .env.example                  # Template for secrets (Twilio SID/token) — never commit real .env
├── .gitignore                    # data/, checkpoints/, .env, wandb/, __pycache__
│
├── configs/                      # All experiment configuration (YAML)
│   ├── data/
│   │   ├── channel_sim.yaml      # Codec choice, SNR levels, resample chain
│   │   └── splits.yaml           # Speaker-disjoint split definitions
│   ├── train_baseline.yaml       # Stage 1: ASVspoof English baseline
│   ├── train_lora_codemix.yaml   # Stage 3: LoRA adaptation
│   └── eval_matrix.yaml          # All evaluation columns & conditions
│
├── data/                         # GITIGNORED — local/Kaggle only
│   ├── raw/                      # As-downloaded corpora
│   │   ├── asvspoof2019_LA/
│   │   ├── mucs2021/
│   │   ├── hiacc/
│   │   ├── indicsynth_subset/
│   │   ├── indictts_deepfake_subset/
│   │   └── indicvoices_subset/
│   ├── generated/                # Project-generated spoofs
│   │   ├── xtts_v2/              # Training-side attack
│   │   └── heldout_tool/         # Unseen-attack split (tool kept separate!)
│   ├── processed/                # After preprocessing + channel sim
│   │   ├── clean/
│   │   └── channel_matched/
│   └── manifests/                # CSVs: filepath,label,language,speaker,tool,condition
│       ├── train_*.csv
│       ├── dev_*.csv
│       └── eval_{english,hindi,tamil,codemixed}_{clean,channel}.csv
│
├── src/
│   ├── data/
│   │   ├── download.py           # Scripted corpus downloads / HF streaming
│   │   ├── preprocess.py         # Resample, VAD trim, loudness norm, segment
│   │   ├── channel_sim.py        # 8 kHz → codec → noise@SNR → 16 kHz chain
│   │   ├── spoof_generation.py   # XTTS-v2 / Tortoise cloning drivers + metadata logging
│   │   ├── build_manifests.py    # Speaker-disjoint split assembly + QA checks
│   │   └── dataset.py            # PyTorch Dataset / collate for variable-length audio
│   │
│   ├── models/
│   │   ├── encoder.py            # wav2vec2 / WavLM / XLSR wrapper (freeze control)
│   │   ├── pooling.py            # Attentive statistics pooling (multi-layer)
│   │   ├── classifier.py         # 2-layer MLP head
│   │   ├── detector.py           # Full model = encoder + pooling + head
│   │   └── lora.py               # PEFT/LoRA configuration helpers
│   │
│   ├── training/
│   │   ├── train.py              # Stage 1 & 3 training loop (AMP, checkpointing, W&B)
│   │   ├── evaluate.py           # Runs one eval column; writes results JSON
│   │   └── metrics.py            # EER, AUC, F1, bootstrap confidence intervals
│   │
│   ├── inference/
│   │   ├── predict.py            # File → probability + verdict (shared by demos)
│   │   ├── streaming.py          # Rolling-window streaming inference + smoothing
│   │   └── export_onnx.py        # ONNX export + int8 quantisation + latency test
│   │
│   └── utils/
│       ├── audio_utils.py
│       ├── seed.py               # Reproducibility (fixed seeds everywhere)
│       └── logging_utils.py
│
├── live_call/                    # SK's domain — the live-call detection system
│   ├── server.py                 # FastAPI app: Twilio webhook (TwiML <Conference> +
│   │                             #   <Start><Stream> tap), media ws, dashboard ws
│   ├── media_handler.py          # μ-law decode → buffer → rolling windows → inference
│   ├── verdict_engine.py         # Smoothing, thresholds, Real/Suspicious/Cloned states
│   │                             #   + escalation state machine (beep → SMS → summary)
│   ├── alerts.py                 # Alert delivery: Conference Participant API audio
│   │                             #   injection (receiver-only), Twilio SMS, post-call
│   │                             #   summary. Provider-agnostic interface
│   ├── webrtc_harness/           # FREE dev/fallback path (Weeks 1–5 + viva backup)
│   │   ├── rtc_server.py         # aiortc: browser↔browser call, server audio tap
│   │   └── call.html             # Simple two-party WebRTC call page
│   ├── audio_assets/
│   │   ├── warning_beep.wav      # Distinctive double-beep
│   │   └── warning_whisper.wav   # "Warning: this voice may be AI-generated."
│   ├── static/
│   │   ├── dashboard.html        # Receiver's live-warning screen
│   │   ├── dialer.html           # Caller page (Twilio Voice JS SDK)
│   │   └── app.js / style.css
│   └── README.md                 # ngrok setup, Twilio console config, trial-timing
│                                 #   notes (activate Week 6!), run instructions
│
├── demo/                         # Gradio upload/record demo → HF Spaces
│   ├── app.py
│   └── requirements.txt          # Slim deps for the Space
│
├── scripts/                      # One-command entry points
│   ├── 01_download_data.sh
│   ├── 02_preprocess_all.sh
│   ├── 03_generate_spoofs.sh
│   ├── 04_train_baseline.sh
│   ├── 05_eval_gap_matrix.sh
│   ├── 06_train_lora.sh
│   └── 07_export_onnx.sh
│
├── notebooks/                    # Exploration only — no pipeline logic lives here
│   ├── eda_corpora.ipynb
│   ├── spoof_quality_audit.ipynb
│   └── error_analysis.ipynb
│
├── experiments/                  # Results (small files — committed)
│   ├── results/                  # JSON per run
│   ├── gap_matrix/               # The headline tables + heatmap figures
│   └── figures/                  # Publication-quality plots (DET curves etc.)
│
├── checkpoints/                  # GITIGNORED — store on Kaggle Datasets / Drive
│
├── paper/                        # Overleaf-synced LaTeX (IEEE template)
│   ├── main.tex
│   ├── sections/
│   └── figures/
│
├── report/                       # Department deliverables
│   ├── report_draft.docx
│   ├── logbook/                  # Weekly entries per member
│   └── presentation/
│
└── tests/
    ├── test_channel_sim.py       # Codec chain sanity (sample rates, no clipping)
    ├── test_splits.py            # CRITICAL: asserts zero speaker overlap across splits
    ├── test_metrics.py           # EER against hand-computed toy cases
    └── test_streaming.py         # Windowing math on a synthetic stream
```

### Structure rules the team agrees to
1. **Notebooks never contain pipeline logic** — anything that produces a paper number lives in `src/` behind a config, so results are reproducible.
2. **`test_splits.py` runs before every training launch.** Speaker leakage between train and eval is the single bug that could invalidate the whole paper.
3. **The held-out TTS tool's outputs never enter any training manifest.** Enforced by keeping `data/generated/heldout_tool/` matchable only to `eval_*` manifests.
4. **Secrets stay in `.env`** (Twilio SID/auth token), never in code or git history.
5. **One shared inference module** (`src/inference/predict.py`) powers both the Gradio demo and the live-call server — fix a bug once, both demos benefit.

### System architecture of the live-call demo (for your report diagram)
```
Caller (browser dialer,       Twilio Cloud                    Your FastAPI server
Twilio Voice JS SDK) ─PSTN─▶  US number → CONFERENCE  ──ws──▶ media_handler.py
                              (caller + receiver legs)        8kHz μ-law frames
Receiver ◀────────────────────┘      ▲                        → decode → 4s rolling window
(phone / browser)                    │                        → ONNX detector
                                     │ Participant API:       → verdict_engine
                                     │ play warning_beep.wav     (smoothing + escalation)
                                     │ to RECEIVER ONLY ◀────────┤
                                     │ (Alert layer 1)           │
                                                                 ├─ ws ──▶ Dashboard 🟢🟡🔴
                                                                 │         (Alert layer 2)
                                                                 ├─ Twilio SMS on sustained
                                                                 │  detection (Alert layer 3)
                                                                 └─ post-call summary SMS +
                                                                    log (Alert layer 4)
```
Design rules: never auto-disconnect the call, never alert the caller (a false positive on a genuine caller would be harmful, and warning a real fraudster teaches evasion) — warn the potential victim, log everything, let the human decide. Note the alignment with the research: Twilio delivers 8 kHz narrowband telephony audio — the exact condition your channel-matched evaluation protocol trains and tests under. The live demo is your evaluation protocol, deployed.

---

## 3. Free alternatives to Twilio (honest comparison)

The blunt truth first: **no provider carries real phone-network (PSTN) calls for free forever** — connecting to telecom carriers costs real money, so every "free" telephony offer is a limited trial. The only permanently-free option is to leave the phone network and use internet calling (WebRTC), which is exactly why the plan builds a WebRTC harness first.

| Option | Truly free? | What you get | Verdict for this project |
|--------|------------|--------------|--------------------------|
| **WebRTC self-hosted (`aiortc`)** | ✅ Forever | Browser-to-browser calls through your own server; full audio access; alert injection is trivial (you own the audio path) | **Use it** — primary dev environment Weeks 1–5 and the zero-cost live fallback at the viva. Only limitation: it's an internet call, not a real phone number |
| **Twilio trial** | ✅ 30 days | 75 voice minutes + 100 SMS, no credit card, verified recipients only | **Use it** — activate Week 6 so the window covers integration + demo recording; best docs + the Conference/Media Streams features this design needs |
| **Telnyx** | Trial credit (~$10) | Similar programmable voice + media streaming; cheapest per-minute rates among major providers | Backup #1 if Twilio blocks you — API concepts port over |
| **Plivo** | Trial credit | Closest API design to Twilio (easiest migration), cheaper rates | Backup #2 |
| **Vonage (Nexmo)** | Small trial credit (~€2) | Voice + WebSocket audio streaming | Credit too small to be primary |
| **Exotel / Ozonetel (Indian CPaaS)** | Free trial, but business KYC | Actual Indian phone numbers, DLT-compliant SMS | Mention in the report as the *production* path for India; KYC makes it impractical for a student demo |
| **Asterisk / FreeSWITCH (self-hosted PBX)** | Software free; PSTN access is not | Full open-source telephony server | Overkill — weeks of telecom engineering, and you'd still pay a SIP trunk provider to reach real phones |

**Recommended stack, final:** aiortc WebRTC (free, always) + Twilio trial timed to Weeks 6–9 + ₹1–2k pay-as-you-go upgrade only if needed. Total realistic spend: **₹0 – ₹1,000**.

---

## 4. Dataset usage matrix (all links verified 21 Jul 2026)

The golden rule the whole project stands on: **the Stage-1 detector is trained on English ASVspoof only.** Everything Indian and code-mixed stays out of Stage-1 training so the gap matrix measures true generalisation. Only Stage 3 (adaptation) touches code-mixed training data, and only from speaker-disjoint splits.

| # | Dataset | Link | Size / subset used | Licence | ROLE |
|---|---------|------|--------------------|---------|------|
| 1 | ASVspoof 2019 LA | https://datashare.ed.ac.uk/handle/10283/3336 | ~24 GB (LA partition only) | Research | **TRAIN (Stage 1)** + English eval column (its own held-out eval split) |
| 2 | ASVspoof 2021 DF eval | https://zenodo.org/records/4835108 | **34.5 GB in 4 parts** + separate keys from asvspoof.org | ODC-ODbL | **EVAL ONLY — OPTIONAL.** At 34.5 GB this busts free-tier storage; download only if college storage arrives, else drop it (proposal already marks it optional) |
| 3 | IndicSynth | https://huggingface.co/datasets/vdivyasharma/IndicSynth | Stream Hindi (206k rows) + Tamil (282k rows) subsets only — never the full 4,000 hrs | CC-BY-NC-4.0 | **EVAL ONLY** — monolingual Indic spoof column. NC licence = fine for research, note it in the ethics table |
| 4 | IndicTTS Deepfake Challenge | https://huggingface.co/datasets/SherryT997/IndicTTS-Deepfake-Challenge-Data | Stream Hindi + Tamil + English subsets | Research | **EVAL ONLY** — monolingual Indic real+fake column; its public wav2vec2 baselines double as a sanity comparison |
| 5 | MUCS 2021 subtask 2 | https://www.openslr.org/104/ (Hindi-English_train.tar.gz 7.3 GB + Hindi-English_test.tar.gz 443 MB; skip Bengali-English) | 89.9 hrs train / 5.2 hrs test, 16 kHz | CC BY-SA 4.0 | **BOTH, speaker-disjoint:** (a) primary code-mixed bonafide for eval, (b) source of speaker reference clips + transcripts for spoof generation, (c) a carved-out speaker-disjoint slice becomes the Stage-3 adaptation training split |
| 6 | HiACC | https://zenodo.org/records/15551669 | 531.6 MB zip; **ADULT subset only (3,318 utterances)** | CC-BY-4.0 | **EVAL ONLY** — spontaneous code-mixed bonafide. **HARD RULE: the 1,858 child utterances are excluded from the pipeline entirely — never used as bonafide, never used as cloning references.** State this exclusion explicitly in the ethics section |
| 7 | IndicVoices | https://huggingface.co/datasets/ai4bharat/indicvoices | Stream filtered Tamil + code-switched subsets only (never the full 7,348 hrs) | Research (AI4Bharat) | **EVAL ONLY** — Tamil bonafide column + supplementary code-mixed bonafide |
| 8 | Project XTTS-v2 clones | Generated Weeks 2–3 from MUCS/HiACC-adult speaker refs | 1,500–2,500 utterances | Own data (never released standalone) | **BOTH, speaker-disjoint:** seen-attack spoofs — eval majority + the Stage-3 adaptation spoof half |
| 9 | Project Tortoise-TTS clones | Generated Week 3, tool firewalled from training | 400–600 utterances | Own data | **EVAL ONLY** — unseen-attack split. Zero Tortoise audio may ever appear in any training manifest |
| 10 | IITG-HingCoS | Request from authors (arXiv:1810.00662) | 25 hrs, on-request | Author permission | **BONUS ONLY** — email the authors in Week 1, but nothing in the plan depends on a reply |

### The gap matrix, expressed in datasets
| Eval column | Bonafide from | Spoof from |
|-------------|--------------|-----------|
| English | ASVspoof 2019 LA eval | ASVspoof 2019 LA eval (+ 2021 DF if storage allows) |
| Monolingual Hindi | IndicTTS-Deepfake (real) | IndicSynth + IndicTTS-Deepfake (fake) |
| Monolingual Tamil | IndicVoices + IndicTTS-Deepfake | IndicSynth + IndicTTS-Deepfake |
| Code-mixed Hindi-English | MUCS test + HiACC adult + IndicVoices code-switched | Project XTTS-v2 (seen) + Tortoise (unseen) |

All four columns pass through the identical channel simulation before evaluation; Stage-3 adaptation uses only the speaker-disjoint MUCS slice + matching XTTS-v2 clones.

### Anti-leakage checklist (enforced by tests/test_splits.py)
1. No speaker appears in both a training manifest and any eval manifest — across bonafide AND spoof (a cloned voice of speaker X counts as speaker X).
2. No Tortoise-generated file in any training manifest, ever.
3. No IndicSynth / IndicTTS / IndicVoices audio in any training manifest — they are eval columns.
4. No HiACC child audio anywhere in the pipeline.
5. Transcripts used to generate a spoof must not pair with that same speaker's bonafide recording of the same sentence in eval (near-duplicate leakage).
