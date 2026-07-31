# 12-Week Project Plan
## Code-Mixing Generalisation Gap in Audio Deepfake Detection + Live Call Detection Demo

**Team**
| Code | Name | Role (primary ownership) |
|------|------|--------------------------|
| **L** | M. Lahari (D006) | **Data Lead** — corpora, preprocessing, channel simulation, spoof generation, dataset documentation, ethics/licence compliance |
| **M** | S. Mounika Reddy (D032) | **Model Lead** — training pipeline, baseline reproduction, gap matrix, LoRA adaptation, ablations, experiment tracking |
| **SK** | K. Sai Krishna Reddy (D034) | **Systems/Demo Lead** — Twilio live-call pipeline, streaming inference server, ONNX optimisation, dashboard, Gradio demo, deployment |
| **ALL** | Combined work | Literature survey, paper writing, error analysis, report, presentation, viva prep |

Each member owns roughly one-third of the technical work, and the paper/report/presentation are split equally by section. Roles are primary ownership, not silos — everyone reviews everyone's outputs weekly.

**Decision locked:** Web app (not mobile app). Live-call demo = real Twilio call (Conference-based) → Media Streams → your inference server → **layered alerts**: (1) in-call beep/whisper into the receiver's ear only, (2) live browser dashboard, (3) escalation SMS during the call, (4) post-call summary SMS — so a warning lands even if the receiver never looks at a screen. Caller side uses Twilio Voice JS SDK (browser dialer) because Twilio cannot issue Indian numbers to individuals. Budget fits: Twilio trial = 75 free voice minutes + 100 SMS, expires 30 days after activation → develop free on a WebRTC harness (Weeks 1–5), activate the trial in Week 6, upgrade pay-as-you-go from the ₹1–2k budget only if the viva outlasts the trial. Training on free Kaggle GPU (30 hrs/week) + Colab.

**Weekly rhythm (fixed, all 12 weeks):** one 30-min sync meeting, one shared log-book entry per member, code merged to the shared GitHub repo by Sunday night. The log book is a department deliverable — writing it weekly costs 10 minutes; reconstructing it in Week 12 costs days.

---

## PHASE 1 — Foundation & Data (Weeks 1–4)

### Week 1 — Setup, datasets, accounts, approvals
**Goal:** every dataset downloading, every account created, ethics cleared. Nothing blocks later weeks.

| Who | Tasks |
|-----|-------|
| **L** | Download + verify per the **dataset usage matrix** (Section 4 of the tech-stack file): ASVspoof 2019 LA from Edinburgh DataShare (~24 GB — the ONLY Stage-1 training corpus); MUCS 2021 Hindi-English from openslr.org/104 (7.3 GB train + 443 MB test — skip the Bengali-English files); HiACC from Zenodo 15551669 (531 MB — extract the ADULT subset only and quarantine the child folders immediately). Set up HF streaming for IndicSynth (Hindi + Tamil subsets), IndicTTS-Deepfake, and IndicVoices filtered subsets — these three are eval-only, never downloaded in full, never trained on. **Skip ASVspoof 2021 DF for now** (34.5 GB — revisit only if college storage arrives). Email IITG-HingCoS authors (bonus, no dependency). Create the per-corpus licence table (note IndicSynth's CC-BY-NC and the HiACC child-data exclusion) for the ethics note. |
| **M** | Set up GitHub repo with the agreed folder structure (see tech-stack file). Set up Kaggle + Colab environments, verify GPU access, confirm wav2vec2-base / WavLM-base / XLSR-53 load and run a forward pass. Set up Weights & Biases (free tier) project. Request college GPU access formally (email in Week 1 even if answer comes later). |
| **SK** | **Do NOT activate Twilio yet** — its trial now expires 30 days after signup (75 free voice minutes, 100 SMS), so save it for Week 6 when integration actually starts. This week: read Twilio Media Streams + Conference docs thoroughly; build the **free WebRTC dev harness** (`aiortc` in Python) — a browser-to-browser call through your own FastAPI server that streams audio server-side. This costs ₹0, has no expiry, and lets all streaming/alert code be developed and tested for free before Twilio ever enters. Set up ngrok. Create Hugging Face account + Spaces placeholder. |
| **ALL** | Finalise literature list (15+ references — proposal already has 17). Each member deep-reads ~6 papers and writes 3–4 sentence summaries into a shared doc (this becomes the related-work section later). Get mentor sign-off on the ethics/licence note **before** any spoof generation. |

**Checkpoint:** all datasets on disk / streaming; WebRTC dev harness passes audio browser→server→browser; mentor ethics sign-off obtained.

---

### Week 2 — Preprocessing + channel simulation pipeline
**Goal:** the audio pipeline that everything downstream depends on.

| Who | Tasks |
|-----|-------|
| **L** | Build preprocessing: resample to 16 kHz mono, VAD trimming (silero-vad), loudness normalisation, 2–10 s segmentation. Build the **channel simulation chain**: 16 kHz → 8 kHz → AMR-NB or G.711 μ-law encode/decode (torchaudio) → additive noise at controlled SNR (5/10/15/20 dB) → back to 16 kHz. Unit-test on 20 sample clips; listen to outputs manually. |
| **M** | Build the dataset/dataloader layer: manifest CSV format (filepath, label, language, speaker, condition), PyTorch Dataset + collate for variable-length audio, train/dev/eval split logic with **speaker-disjoint splits** (critical — leakage here invalidates the paper). Write the metrics module: EER, AUC-ROC, F1, with bootstrap confidence intervals. |
| **SK** | Select speakers + transcripts from MUCS + HiACC **adult subset** with L (child speakers never used, for anything): filter for audio quality, pick 30–50 adult speakers with enough clean reference audio (≥30 s each), extract code-mixed transcripts for synthesis. With M, carve the MUCS speaker list into two disjoint pools now — an eval pool and a Stage-3 adaptation pool — so speaker-disjointness is baked in before any generation. Set up Coqui XTTS-v2 locally/on Colab; generate 20 pilot clones; team listens and rates quality. |
| **ALL** | 1-hour session: listen to pilot clones + channel-simulated audio together; agree quality bar for the spoof set. |

**Checkpoint:** channel-sim pipeline produces correct, audible telephony-grade audio; pilot clones judged usable; speaker/transcript list frozen.

---

### Week 3 — Spoof generation at scale
**Goal:** the project-generated code-mixed spoof corpus — your core data contribution.

| Who | Tasks |
|-----|-------|
| **L** | Run XTTS-v2 generation at scale on Kaggle/Colab: target 1,500–2,500 code-mixed utterances across the selected speakers (training-side attack). Log generation metadata (tool, speaker ref, transcript, settings) per file. |
| **SK** | Set up the **held-out second TTS tool** (Tortoise-TTS or another open model, excluded from any training exposure) and generate the unseen-attack split (~400–600 utterances). Keep tool identity strictly separated in the manifests. |
| **M** | Start Stage-1 training scaffolding while generation runs: encoder + attentive-stats pooling + 2-layer MLP head, training loop with mixed precision, checkpointing, W&B logging. Smoke-test on a 1%-subset of ASVspoof to confirm the loop learns. |
| **ALL** | Manual quality audit: each member listens to a random 50-clip sample of generated spoofs, tags failures (wrong language, garbled code-switch points, artifacts). Report generation-quality statistics — reviewers will ask. |

**Checkpoint:** full spoof set generated with metadata; audit statistics documented; training loop verified on subset.

---

### Week 4 — Dataset v1 freeze + baseline training starts
**Goal:** evaluation corpus locked; English baseline training underway.

| Who | Tasks |
|-----|-------|
| **L** | Assemble **Dataset v1** exactly per the dataset usage matrix: speaker-matched bonafide/spoof pairs, all evaluation audio passed through identical channel simulation (clean copies retained separately). Final manifests for every evaluation column — English (ASVspoof eval), mono-Hindi (IndicTTS-Deepfake + IndicSynth), mono-Tamil (IndicVoices + IndicSynth), code-mixed (MUCS test + HiACC adult + XTTS/Tortoise spoofs). Run the full anti-leakage checklist (test_splits.py: speaker disjointness incl. cloned voices, no Tortoise in training, no eval-only corpora in training, no child audio anywhere). Write the dataset datasheet (composition, sources, licences, generation process). |
| **M** | Launch Stage-1 baseline: fine-tune wav2vec2-base (+ head) on ASVspoof 2019 LA on Kaggle. Iterate on LR/schedule until dev EER trends toward published ranges (~1–5% EER territory for this class of system). This may take multiple Kaggle sessions — start early in the week. |
| **SK** | Build the offline inference module: load checkpoint → predict on a wav file → probability + verdict. This is the shared foundation for both the Gradio demo and the live-call server. Begin the FastAPI skeleton for the streaming server. |
| **ALL** | Review 1 prep if scheduled around here: slides showing dataset v1 statistics, pipeline diagram, pilot results. |

**Checkpoint:** **Dataset v1 frozen** (no more changes without team decision + changelog). Baseline training in progress.

---

## PHASE 2 — Core Research Results (Weeks 5–8)

### Week 5 — Baseline validated
**Goal:** trust the model before trusting any cross-lingual number.

| Who | Tasks |
|-----|-------|
| **M** | Finish baseline training; evaluate on ASVspoof held-out eval. **Gate: EER must be in a defensible published range.** If not, debug (features, splits, pooling) before anything else — every later number depends on this. |
| **L** | Run the full evaluation-set QA pass: verify no speaker overlap across splits, verify channel simulation applied uniformly, verify label balance per column. Fix + re-freeze as v1.1 if issues found (with changelog). |
| **SK** | Prototype **streaming inference**: rolling 4-second window with 2-second hop over a live audio stream, per-window verdict, exponential smoothing over windows. Test by piping a wav file through it as a fake stream, then live over the free WebRTC harness. Measure CPU latency; if > ~1.5 s per window, export to ONNX + int8 quantisation. |
| **ALL** | Mid-project risk review (30 min): is the baseline solid? Is compute burn sustainable? Adjust Week 6–8 scope if needed. |

**Checkpoint:** validated English baseline with documented EER; streaming inference works on simulated streams.

---

### Week 6 — Stage 2: THE GAP MATRIX (headline result)
**Goal:** the paper's central empirical claim.

| Who | Tasks |
|-----|-------|
| **M** | Run the English-trained model, unmodified, across all evaluation columns: English held-out / monolingual Hindi / monolingual Tamil / code-mixed Hindi-English — each in clean **and** channel-matched conditions. Compute EER/AUC/F1 + bootstrap CIs per cell. Produce the gap-matrix table and plots. |
| **L** | Slice the results: performance vs. SNR, vs. attack tool (seen XTTS vs unseen tool), vs. corpus source. Start the error-analysis notebook: which utterances fool the detector, which bonafide clips get flagged. |
| **SK** | **Activate the Twilio trial NOW** (its 30-day window now covers Weeks 6–9, exactly the integration phase). Get the trial number, verify teammates' phones as allowed recipients. Stand up the Twilio media pipeline for real: webhook answers inbound call → **Conference** with `<Start><Stream>` tap → websocket server receives 8 kHz μ-law frames → decode → buffer into rolling windows → run streaming inference (port over the code already proven on the WebRTC harness). Test end-to-end with a real call playing recorded audio; log verdicts. Track the 75-free-voice-minutes budget from day one — dev calls should be short. (No UI yet — terminal output is fine.) |
| **ALL** | **Review 2 / Midterm package:** gap matrix v1 + demo of live-call verdicts in terminal. This is a strong midterm: headline research result + working core of the wow-factor demo. |

**Checkpoint:** gap matrix v1 exists — the project's central result is now in hand. A real phone call gets classified live.

---

### Week 7 — Stage 3: LoRA adaptation + live-call dashboard
**Goal:** the "how much can we close the gap" result + visible demo.

| Who | Tasks |
|-----|-------|
| **M** | LoRA / top-layer adaptation on the code-mixed training split (PEFT). Re-evaluate on held-out code-mixed data (clean + channel-matched, seen + unseen attacks). Produce the gap-closure table: before vs after adaptation, all columns (to show adaptation doesn't wreck English performance). |
| **SK** | Build the **receiver dashboard**: simple web page (served by FastAPI) showing live call status, rolling verdict indicator (Real ✅ / Suspicious ⚠️ / Cloned 🚨), confidence meter, and verdict history timeline — updated over websocket. Build the **in-call audio alert** (primary alert channel): when the verdict crosses threshold, the server uses the Conference Participant API to play a warning beep / short whispered announcement **to the receiver only** (caller hears nothing). Wire the browser caller page with Twilio Voice JS SDK so demos need no international calling. |
| **L** | Curate the demo audio arsenal: 10–15 clips — genuine code-mixed speech, XTTS clones, unseen-tool clones — that will be played into demo calls. Verify each produces the expected verdict through the full Twilio path; document any surprises (these become paper error-analysis material). |
| **ALL** | Decide the operating threshold for the live demo from the DET curve (e.g., favour low false-alarm on genuine speech — flagging a real caller as fake is the worse demo failure). |

**Checkpoint:** adaptation numbers final-draft quality; a live Twilio call shows a real-time warning on the dashboard.

---

### Week 8 — Ablations, error analysis, results freeze
**Goal:** everything the paper needs, locked.

| Who | Tasks |
|-----|-------|
| **M** | Primary ablation: frozen vs fine-tuned encoder. If Kaggle hours allow, secondary: wav2vec2 vs WavLM. Final consolidated results tables, publication-quality figures (DET curves, gap matrix heatmap, adaptation deltas). |
| **L** | Complete error analysis: breakdown by code-switch density, SNR, utterance length, attack tool. Write it up as draft paper text directly (Section: Analysis). Finalise the dataset datasheet as a paper appendix / supplementary. |
| **SK** | Build the **alert escalation ladder** in `verdict_engine`: first detection → single beep + dashboard red; sustained high-confidence 10+ s → beep repeats + **SMS alert** fired via Twilio ("Alert: your call at HH:MM showed signs of AI voice cloning. Do not share OTPs or make payments."); call end → **post-call summary SMS + dashboard log** regardless (covers the missed-everything case — voice-fraud harm usually happens *after* the call). Trial includes 100 SMS to verified numbers — plenty. Harden the live system: reconnect logic, call-end handling, graceful degradation, deploy inference server to a persistent host. Also build the simple **Gradio upload/record demo** on HF Spaces (the proposal's promised deliverable — 1–2 days reusing the inference module). **Decide now:** if the viva date falls after the 30-day trial expires, upgrade the Twilio account this week using the ₹1–2k budget (pay-as-you-go: number ~$1.15/mo + cents/min — the budget covers it several times over). |
| **ALL** | **Results freeze meeting:** agree these are the numbers the paper reports. Later reruns only for bug fixes, logged in the changelog. |

**Checkpoint:** all experimental results frozen. Both demos (live-call + Gradio upload) functional.

---

## PHASE 3 — Paper, Polish, Delivery (Weeks 9–12)

### Week 9 — Paper draft
**Goal:** full IEEE-format draft.

| Who | Tasks (writing split — each owns their sections end-to-end) |
|-----|-------|
| **L** | Sections: Dataset & Corpus Construction, Channel-Matched Protocol, Ethics/Licence statement. |
| **M** | Sections: Methodology (model, training stages), Experiments & Results (gap matrix, adaptation, ablations), figures/tables. |
| **SK** | Sections: Introduction, Related Work (from the Week-1 summaries), System/Deployment (live telephony detection — a genuine differentiator: your evaluation protocol *matches* your deployment channel), Conclusion & Future Work. |
| **ALL** | Venue shortlist (2–3 legitimate Scopus-indexed conferences — INDICON/TENCON-class or Springer speech/AI venues with verifiable peer review; check submission deadlines NOW and let deadlines reorder Weeks 9–10 if needed). Cross-review: each member edits both others' sections. |

**Checkpoint:** complete draft in Overleaf, internally reviewed once.

---

### Week 10 — Paper submission + demo polish
**Goal:** rubric tier 2 secured (submitted, <10% similarity).

| Who | Tasks |
|-----|-------|
| **ALL (primary task)** | Revision pass with mentor feedback → plagiarism/AI-similarity check → fix → **submit to chosen venue**. This is the week's non-negotiable outcome. |
| **SK** | Demo polish: dashboard styling, a rehearsed 3-minute demo script (dial → play genuine clip → verdict Real → play clone → **beep sounds in the panel member's ear** → dashboard goes red → escalation SMS arrives → post-call summary SMS after hang-up), backup screen-recording *with audio* of a successful run (Twilio/network can fail during a viva — always have the recording). |
| **L** | Final licence/ethics documentation package; README for the dataset; verify Twilio spend vs budget and top up (₹1–2k buffer) only if trial credit is exhausted. |
| **M** | Reproducibility pass: configs + seeds + a `REPRODUCE.md` that regenerates every paper table from checkpoints. |

**Checkpoint:** **paper submitted.** Demo runs clean start-to-finish + backup recording exists.

---

### Week 11 — Final evaluation matrix + viva material
**Goal:** everything frozen, viva-grade.

| Who | Tasks |
|-----|-------|
| **M** | Final full evaluation matrix (all columns × clean/channel-matched × seen/unseen attacks) from frozen checkpoints; final model freeze + archived checkpoints. |
| **L** | Failure-case exhibit for the viva: 5–6 curated examples (audio + model output + explanation) showing both successes and honest failures — panels trust teams who show failures. |
| **SK** | End-to-end demo dress rehearsal ×3 with the whole team; fix anything flaky; finalise the backup recording; confirm HF Spaces Gradio demo link is public and stable. |
| **ALL** | Start the department report skeleton (institute template) — drop in paper content reformatted to department structure; assemble log book entries. |

**Checkpoint:** final numbers locked; demo bulletproof; report ~50% assembled.

---

### Week 12 — Report, presentation, viva
**Goal:** Review 3 delivery.

| Who | Tasks |
|-----|-------|
| **L** | Report chapters: Introduction, Literature Survey, Dataset & Methodology (data half). Log book compilation + formatting. |
| **M** | Report chapters: Methodology (model half), Results & Analysis, all figures/tables. |
| **SK** | Report chapters: System Design & Implementation (live-call architecture diagram), Conclusion & Future Work. Final presentation deck (each member presents their own third). |
| **ALL** | Viva rehearsal ×2 using the proposal's Section-10 anticipated-questions list, plus new ones: "Why is the live demo not out of scope anymore?" → *because Twilio Media Streams delivers exactly the 8 kHz telephony audio our channel-matched protocol was designed for — the demo is the evaluation protocol, deployed.* Hard-bound report submission. |

**Checkpoint:** Review 3 done. 🎓

---

## Workload balance summary

| Member | Technical core (Wks 1–8) | Writing core (Wks 9–12) |
|--------|--------------------------|--------------------------|
| **L** | Data pipeline, channel sim, spoof generation, dataset freeze, error analysis | Dataset/protocol/ethics sections; report intro + lit survey + data chapters; log book |
| **M** | Training infra, baseline, gap matrix, LoRA, ablations, final matrices | Method/results sections + figures; report results chapters; reproducibility |
| **SK** | Twilio pipeline, streaming inference, dashboard, Gradio, deployment | Intro/related-work/system sections; report system chapter; presentation deck |

## Top risks in the 12-week (vs 14-week) version
1. **Baseline doesn't reproduce (Week 5).** Buffer: the gap matrix run itself (Week 6) is cheap — only training is expensive. If baseline slips 1 week, compress Week 8 ablations to the single primary ablation.
2. **Conference deadline mismatch.** Check deadlines in Week 9 Day 1; if the best venue closes early, swap Weeks 9–10 forward and push demo polish to Week 11.
3. **Twilio flakiness at viva.** Mitigated by the mandatory backup screen recording (Week 10), the free WebRTC harness (same detection pipeline, zero-cost, can be demoed live if Twilio dies), and the Gradio upload demo as a third fallback.
3b. **Twilio 30-day trial expiry / 75-minute cap.** Mitigated by: activating the trial only in Week 6 (window covers Weeks 6–9), doing all pre-integration development on the free WebRTC harness, keeping dev calls short, and the ₹1–2k budget earmarked for a pay-as-you-go upgrade in Week 8 if the viva falls outside the trial window.
4. **Kaggle 30 hr/week ceiling.** Mitigated by frozen-encoder-first, LoRA-not-full-finetune, and using all three members' Kaggle accounts for parallel independent runs (baseline on one, ablation on another).
