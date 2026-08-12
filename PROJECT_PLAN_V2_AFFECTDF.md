# Updated Project Plan (v2) — AffectDF-Anchored

**Quantifying and Closing the Code-Mixing Generalisation Gap in Audio Deepfake
Detection + Live Call Detection Demo**

B.Tech CSE (Data Science), Semester 7 — Team L / M / SK
Plan revised: **12 August 2026** (start of Week 3)

> **This is the only active plan.** It supersedes `PROJECT_PLAN_12_WEEKS.md` and
> `Project_Plan_12_Weeks.docx` for all task IDs, owners and deliverables. Where
> they conflict, v2 wins. The older files are kept for history only.
>
> Markdown conversion of `Updated_Project_Plan_v2_AffectDF.docx` (the .docx
> remains the signed original).

---

## 1. Strategy: our position relative to the base paper

**Base paper:** AffectDF (Mahapatra et al., arXiv:2608.05507, Aug 2026) — the most
comprehensive benchmark for emotionally expressive speech deepfake detection. It
shows that SDD systems trained on conventional benchmarks collapse under
distribution shift (e.g. AASIST: 0.83% EER in-domain → 56.40% on AffectDF), that
retraining on the shifted domain gives inconsistent gains and damages
conventional-benchmark performance, and that it stops at measurement — no
mitigation strategy is tested.

We do not compete with AffectDF on scale (260 hours, 21 attacks, A100/H100
clusters). We contribute on the axes it does not cover. Our value-over-baseline
claim, stated once and defended everywhere:

> "AffectDF measures the emotional generalisation gap and stops. We measure the
> code-mixing gap under channel-matched telephony conditions — and then close it,
> and deploy it live."

### 1.1 Gap-contribution map

| AffectDF gap / omission | Our contribution |
|---|---|
| Entirely English (ESD + MSP-Podcast); no code-mixed or non-English speech | **Hinglish code-mixed axis**: gap matrix across English / mono-Hindi / mono-Tamil / code-mixed columns — headline result |
| Measures failure only; no mitigation tested | **3-system comparison**: S1 English-only (gap) → S2 LoRA-adapted (cheap fix) → S3 native code-mixed (ceiling) |
| Clean-condition audio; no telephony/codec simulation | **Channel-matched protocol**: 8 kHz AMR-NB/G.711 + SNR noise on every eval cell, clean copies retained |
| No system/deployment component | **Live Twilio call detection**: beep in receiver ear, dashboard, SMS escalation — the protocol deployed |
| Emotion-wise slicing (their Table 3) | Our analogue: **switch-density slicing** + read vs spontaneous (HiACC) fixed-tool contrast |
| Their finding: domain retraining wrecks conventional performance | We test the same trade-off on our axis: does S3 lose English performance where S2 (LoRA) preserves it? |

**Methodology adopted from AffectDF** (makes our paper look mature): attack-ID
taxonomy table with disjoint speakers AND disjoint attack systems across splits;
rich per-clip protocol files; the low-level-cue logistic-regression shortcut check
(their Appendix G); fixed-generator style-contrast design; min t-DCF alongside EER
where applicable; and zero-shot cross-evaluation of our systems on an AffectDF
test subset (CC BY-NC 4.0) as an external benchmark column.

---

## 2. Where we are (12 Aug 2026)

Weeks 1–2 code is written and pushed on per-task branches (repo scaffold, download
scripts, preprocessing, channel sim, dataset/metrics modules, WebRTC harness,
pool-carving code).

**Blocked human gates, in order:**

1. Review + merge the 4 open PRs.
2. **Mentor ethics sign-off** — nothing generates before this.
3. Run the MUCS + HiACC downloads and preprocessing.
4. Channel-sim listening test.
5. Speaker selection + pool freeze.

**Timeline:** 10 weeks remain (Weeks 3–12). Spoof generation is the critical path
and it sits behind the gates above. Clear them inside Week 3 or every downstream
week slips.

---

## 3. The three systems

| System | Definition | Role in the paper |
|---|---|---|
| **S1** | wav2vec2/WavLM fine-tuned on ASVspoof 2019 LA only (zero code-mixed exposure) | Establishes the gap; anchors us to published literature |
| **S2** | S1 + LoRA adaptation on the disjoint code-mixed adaptation pool (~1% trainable params) | The cheap fix; the mitigation AffectDF never tested |
| **S3** | Trained directly on MUCS/HiACC bonafide + XTTS/RVC clones | The ceiling; also tested for reverse degradation on English |

**Fallback rule (agreed now):** if Kaggle hours run short, drop S3 ablations first,
then S2 — **never S1**. If S1 fails its published-EER gate by end of Week 6,
invert: anchor with pretrained AASIST/RawNet2 zero-shot on our eval set and
reframe around cross-tool generalisation.

---

## 4. Week-by-week plan (Weeks 3–12)

### Week 3 (this week) — Unblock everything + AffectDF integration

**Goal:** every human gate cleared; base-paper positioning written; pilots rated.

| ID | Owner | Task | Pushed to GitHub by Sunday |
|---|---|---|---|
| **W3-T1** | ALL | Clear the gates: each member reviews + merges one teammate's open PR (4 total); obtain mentor ethics sign-off on the REVISED scope (larger spoof volume, RVC added, AffectDF cross-eval); log-book entries for Weeks 1–2. | Merged PRs; `docs/ethics/mentor_signoff.pdf` |
| **W3-T2** | L | Run MUCS + HiACC downloads; verify HiACC child quarantine against the real folder/manifest layout BY HAND; run preprocessing to 16 kHz segments; channel-sim listening test on 20 clips. | download logs + `docs/qa/child_quarantine_check.md` |
| **W3-T3** | M | Deep-read AffectDF; write the related-work anchor + precise gap statement ("multilingual, channel, emotional shift covered; intra-utterance code-switching unexamined"); draft OUR attack-taxonomy table (attack IDs, tool, pool, split) in AffectDF Table-1 format. | `docs/related_work_notes.md` + `docs/attack_taxonomy.md` |
| **W3-T4** | L+SK | Speaker selection: scripted SNR/duration ranking, team listening pass; select 30–50 adult speakers; freeze THREE disjoint pools (train / adaptation / eval) and commit. | `data/manifests/speaker_pools.csv` (frozen) |
| **W3-T5** | SK | **After sign-off:** XTTS-v2 pilot (20 clips; test Devanagari vs romanised, hi vs en) + RVC pilot (1 speaker model, 10 conversions); team rates all 30, agrees quality bar. | `notebooks/spoof_quality_audit.ipynb` (pilot) |
| **W3-T6** | M | Verify Kaggle GPU on all 3 accounts, W&B project, HF tokens/gated datasets; download ASVspoof 2019 LA for S1 (keep — S1 is the anchor). | `notebooks/env_check.ipynb` run + links in README |

**Checkpoint:** All gates cleared; pools frozen; pilots rated; AffectDF positioning
committed. If sign-off has not happened by end Wednesday, escalate to the mentor
in person.

### Week 4 — Spoof generation at scale (now the core data contribution)

**Goal:** the full code-mixed spoof corpus, train + eval + held-out, with
AffectDF-grade metadata.

| ID | Owner | Task | Deliverable |
|---|---|---|---|
| **W4-T1** | L | XTTS-v2 generation at scale from `generation_jobs.csv`: train-pool speakers, ~4,000+ utterances targeting rough parity with bonafide hours; idempotent/resumable; per-clip metadata (tool, version, speaker, reference, transcript, settings, pool). | `src/data/spoof_generation.py` + metadata CSV |
| **W4-T2** | SK | RVC track: train per-speaker models for 10–15 train-pool targets (overnight batches); convert same-pool source utterances; log source + target + model checksum. | `src/data/rvc_generation.py` + metadata |
| **W4-T3** | SK | Held-out attacks, eval-pool speakers only: Tortoise-TTS 400–600 utterances (+ optionally kNN-VC); firewalled directory; never in any training manifest. | `data/generated/heldout_tool/` metadata |
| **W4-T4** | M | Model modules (encoder wrapper, attentive stats pooling, MLP head) + training loop (AMP, checkpointing, resume-from-latest, W&B); smoke-test on 1% of the new data. | `src/models/` + `src/training/train.py` + smoke run |
| **W4-T5** | M | Rewrite anti-leakage suite for the new regime: speaker-pool disjointness (incl. cloned voices AND RVC source/target), tool firewall, transcript near-duplicates, no child audio; wire into CI. | `tests/test_splits.py` passing in CI |
| **W4-T6** | L | Quality audit: auto-filters (duration mismatch, near-silence, clipping) + 50-clip listening per member + ECAPA-TDNN speaker-similarity and Whisper transcript-check statistics. | audit report + stats in notebook |

**Checkpoint:** Full spoof corpus generated with metadata; audit statistics
documented; leakage tests green.

### Week 5 — Dataset v1 freeze + S1/S3 launched in parallel

| ID | Owner | Task | Deliverable |
|---|---|---|---|
| **W5-T1** | L | Assemble Dataset v1: manifests for all four eval columns (English / mono-Hindi / mono-Tamil / code-mixed), identical channel simulation on all eval audio, clean copies retained; FREEZE with changelog. | `src/data/build_manifests.py` + manifests |
| **W5-T2** | L | Dataset datasheet in AffectDF style: attack-taxonomy table, per-corpus licences, generation process, exclusions, spoof:real ratios per split. | `docs/datasheet.md` |
| **W5-T3** | M | Launch S1 (ASVspoof baseline) on account 1 and S3 (code-mixed native) on account 2, in parallel; iterate LR/schedule; all runs in W&B. | configs + W&B runs + checkpoints on Kaggle Datasets |
| **W5-T4** | M | AffectDF Appendix-G replica: logistic regression on 8 low-level signal features (RMS, clipping, ZCR, rolloff, etc.) over OUR dataset — must be near chance, else fix the pipeline before trusting anything. | `experiments/results/lowlevel_cue_check.json` |
| **W5-T5** | SK | FastAPI server skeleton + streaming windowing prototype (4 s window / 2 s hop over faked stream). | `live_call/server.py` + `src/inference/streaming.py` |
| **W5-T6** | ALL | Review 1 package (if scheduled): dataset statistics, taxonomy table, pipeline diagram, pilot results, AffectDF positioning slide. | `report/presentation/review1.pptx` |

**Checkpoint:** Dataset v1 FROZEN. S1 and S3 training in progress. Low-level-cue
check passed (near-chance).

### Week 6 — Validation gates (trust before numbers)

| ID | Owner | Task | Deliverable |
|---|---|---|---|
| **W6-T1** | M | **S1 GATE:** EER on ASVspoof held-out eval must land in a defensible published range. If off, debugging takes priority over everything. | `experiments/results/baseline.json` |
| **W6-T2** | M | **S3 GATE:** sane EER on eval pool; near-0% on seen tools + collapse on unseen = shortcut learning — investigate (channel sim on training data? augmentation on?). | `experiments/results/s3_validation.json` |
| **W6-T3** | L | Eval-set QA (speaker overlap zero, uniform channel sim, label balance) + slicing infrastructure: SNR bins, tool tags, switch-density annotation, read/spontaneous tags (HiACC). | `docs/qa_report.md` + `notebooks/eda_corpora.ipynb` |
| **W6-T4** | SK | Streaming inference live over WebRTC harness (screen capture); ONNX export + int8; CPU latency benchmark (< 1.5 s per 4 s window). | demo capture + benchmark table |
| **W6-T5** | ALL | Mid-project risk review: both gates status, GPU burn, Twilio window planning; adjust Weeks 7–9 scope; minutes committed. | `docs/minutes/week6_risk_review.md` |

**Checkpoint:** Both training gates passed (or contingency invoked). Streaming
works on live audio.

### Week 7 — THE GAP MATRIX + reverse-degradation + Twilio live

| ID | Owner | Task | Deliverable |
|---|---|---|---|
| **W7-T1** | M | Run S1 unmodified on all four eval columns, clean AND channel-matched; EER/AUC/F1 per cell with bootstrap CIs; publication heatmap. **This is the headline result**, in hand by midterm. | `experiments/gap_matrix/` + heatmap figure |
| **W7-T2** | M | Reverse check (our version of AffectDF's key finding): evaluate S3 on English/ASVspoof eval — does native code-mixed training wreck conventional performance? | `experiments/results/s3_reverse_degradation.json` |
| **W7-T3** | L | Slice gap-matrix results by SNR, attack tool (seen vs unseen), corpus, switch density; begin error analysis. | `notebooks/error_analysis.ipynb` |
| **W7-T4** | SK | **ACTIVATE Twilio trial** (30-day window covers Weeks 7–10): webhook → Conference → Media Streams tap → websocket → streaming inference; track 75-min budget from day one. | `live_call/server.py` + `media_handler.py` |
| **W7-T5** | SK | End-to-end real-call test: dial in, play genuine + spoofed clips, log verdicts. | verdict logs + short recording in `docs/` |
| **W7-T6** | ALL | Review 2 / midterm: gap matrix v1 + live terminal call demo + AffectDF comparison slide. | `report/presentation/review2.pptx` |

**Checkpoint:** Gap matrix v1 exists. A real phone call gets classified live.
Reverse-degradation result recorded.

### Week 8 — S2 (LoRA) + AffectDF cross-eval + dashboard

| ID | Owner | Task | Deliverable |
|---|---|---|---|
| **W8-T1** | M | S2: LoRA/top-layer adaptation (PEFT) on the adaptation pool; re-evaluate on all columns (clean + channel, seen + unseen); verify English performance preserved; produce the 3-way S1/S2/S3 table draft. | `configs/train_lora_codemix.yaml` + `adaptation.json` |
| **W8-T2** | M+L | External benchmark column: zero-shot S1/S2/S3 on a sampled AffectDF test subset (CC BY-NC; check size/practicality first). | `experiments/results/affectdf_crosseval.json` |
| **W8-T3** | L | Fixed-tool style contrast (their acted-vs-spontaneous move, our data): same generator, read vs spontaneous HiACC subsets; write up. | analysis section in `error_analysis.ipynb` |
| **W8-T4** | SK | Receiver dashboard (live status, rolling verdict, confidence, timeline) + browser dialer; alert layer 1: warning beep/whisper into RECEIVER leg on threshold cross. | `live_call/static/` + `alerts.py` |
| **W8-T5** | L | Demo audio arsenal (10–15 verified clips through full Twilio path); operating threshold from DET curve, ALL sign off. | `demo_assets/` + `configs/threshold.yaml` |

**Checkpoint:** 3-way table draft exists. AffectDF cross-eval done. Live call shows
beep + red dashboard.

### Week 9 — Ablations, escalation, RESULTS FREEZE

| ID | Owner | Task | Deliverable |
|---|---|---|---|
| **W9-T1** | M | Primary ablation: XTTS-only vs XTTS+RVC training for S3 (does attack diversity improve unseen-tool EER?); secondary (frozen vs fine-tuned encoder) only if hours allow. | `experiments/results/ablations.json` |
| **W9-T2** | M | Consolidate final tables + publication figures (gap heatmap, 3-way table, DET curves, cross-eval, reverse-degradation). | `experiments/figures/` complete |
| **W9-T3** | L | Complete error analysis by switch density, SNR, length, tool — written DIRECTLY as draft paper text; datasheet finalised as supplementary. | `paper/sections/analysis.tex` + supplementary |
| **W9-T4** | SK | Escalation ladder (beep → repeat + SMS → call-end summary); Gradio demo on HF Spaces; persistent host deploy; decide Twilio pay-as-you-go top-up. | `verdict_engine.py` + live Space URL |
| **W9-T5** | ALL | **RESULTS FREEZE** meeting: these are the paper's numbers; later reruns only for logged bug fixes. | `CHANGELOG.md` freeze entry + minutes |

**Checkpoint:** All experimental results frozen. Both demos functional.

### Week 10 — Paper draft (full IEEE draft, internally reviewed)

| ID | Owner | Task | Deliverable |
|---|---|---|---|
| **W10-T1** | ALL | Venue shortlist DAY 1 (2–3 legitimate Scopus-indexed: INDICON/TENCON-class or Springer speech/AI); check deadlines; predatory venues excluded. | `docs/venues.md` |
| **W10-T2** | L | Write: Dataset & Corpus Construction (taxonomy table front and centre), Channel-Matched Protocol, Ethics/Licence statement. | `paper/sections/` (L) |
| **W10-T3** | M | Write: Methodology (S1/S2/S3), Experiments & Results, all figures placed. | `paper/sections/` (M) |
| **W10-T4** | SK | Write: Introduction + Related Work ANCHORED ON AffectDF, System & Deployment, Conclusion. | `paper/sections/` (SK) |
| **W10-T5** | M | REPRODUCE.md draft: configs + seeds + commands regenerating every table from checkpoints. | `REPRODUCE.md` (draft) |
| **W10-T6** | ALL | Cross-review via PR comments; one full integration pass; `main.tex` compiles. | review comments resolved |

**Checkpoint:** Complete draft in Overleaf/repo, internally reviewed once.

### Week 11 — SUBMIT + demo polish + final matrix

| ID | Owner | Task | Deliverable |
|---|---|---|---|
| **W11-T1** | ALL | Mentor revision pass; plagiarism/AI-similarity check (< 10%); **SUBMIT** to chosen venue. Non-negotiable outcome of the week. | submission receipt + tagged version |
| **W11-T2** | M | Final full evaluation matrix from frozen checkpoints; model freeze; archive checkpoints to Kaggle Datasets/Drive with hashes. | `final_matrix.json` + archive links |
| **W11-T3** | SK | Demo polish + rehearsed 3-minute script; record the backup screen-recording WITH audio of a full successful run. | `docs/viva/demo_script.md` + recording |
| **W11-T4** | L | Failure-case exhibit (5–6 curated successes AND honest failures, incl. switch-boundary examples); final licence/ethics package; Twilio spend check. | `docs/viva/failure_cases.md` + `budget.md` |
| **W11-T5** | ALL | Department report skeleton in institute template; drop in reformatted paper content; assemble log-book entries. | `report/report_draft.docx` (~50%) |

**Checkpoint:** PAPER SUBMITTED. Final numbers locked. Backup recording exists.
Report ~50%.

### Week 12 — Report, presentation, viva (Review 3)

| ID | Owner | Task | Deliverable |
|---|---|---|---|
| **W12-T1** | L | Report chapters: Introduction, Literature Survey (AffectDF-anchored), Dataset & Methodology (data half); complete log book. | `report/` chapters + logbook |
| **W12-T2** | M | Report chapters: Methodology (model half), Results & Analysis, all figures/tables; REPRODUCE.md fresh-clone test; tag `v1.0`. | `report/` chapters + git tag |
| **W12-T3** | SK | Report chapters: System Design (live-call architecture diagram), Conclusion; final presentation deck; 3x dress rehearsal. | `report/` chapters + `final.pptx` |
| **W12-T4** | ALL | Viva Q&A prep including the two anticipated hard questions: "How is this different from AffectDF?" and "Why trust results on self-generated spoofs?" | `docs/viva/qa_prep.md` |
| **W12-T5** | ALL | Final report assembly, print, hard-bound submission; repo cleanup; tag `v1.0-final`. | final PDF + git tag |

**Checkpoint:** Review 3 done. Project complete.

---

## 5. The paper's claims table (write toward this from day one)

| Claim | Evidence that proves it | Produced in |
|---|---|---|
| English-trained SDD fails on Hinglish telephony speech | Gap matrix: S1 EER per column, clean vs channel, with CIs | Week 7 |
| LoRA adaptation closes part of the gap cheaply | S1 vs S2 delta on all columns; English performance preserved | Week 8 |
| Native code-mixed training is the ceiling — at a cost | S3 results + reverse-degradation on English | Weeks 6–7 |
| Results are not shortcut artifacts | Low-level-cue check near chance; held-out-tool EER; channel sim on both classes | Weeks 5–6 |
| Results are not dataset-local | Zero-shot cross-eval on AffectDF subset + pretrained AASIST comparison row | Week 8 |
| The protocol works in the real world | Live Twilio call detection with latency numbers + demo recording | Weeks 7–9 |

---

## 6. Top risks (revised)

- **Gates slip past Week 3.** Ethics sign-off and downloads are human steps with no
  technical workaround; if not cleared by Wednesday of Week 3, escalate in person
  and compress Week 4 generation into split day/night sessions across all three
  accounts.
- **S1 baseline does not reproduce (Week 6 gate).** Contingency: anchor with
  pretrained AASIST/RawNet2 zero-shot rows; reframe around cross-tool
  generalisation. Decided in the Week 6 risk review, not later.
- **Shortcut learning in S3.** Detected by the Week 5 low-level-cue check and the
  seen/unseen tool split; mitigated by channel sim + RawBoost on training data.
- **Kaggle 30 hr/week ceiling.** Frozen-encoder-first, LoRA over full fine-tuning,
  parallel runs across three accounts; cut order: S3 ablations → S2 → never S1.
- **AffectDF cross-eval impractical (download size).** Fallback: sampled subset;
  if still impractical, the pretrained-AASIST comparison row alone carries the
  external-validity claim.
- **Conference deadline mismatch.** Checked Week 10 Day 1; if the best venue closes
  early, swap Weeks 10–11 forward and push demo polish into Week 12.
- **Twilio trial window / viva date.** Trial activated Week 7 so the 30-day window
  covers Weeks 7–10; Rs 1–2k earmarked for pay-as-you-go if the viva falls outside
  it; WebRTC harness and Gradio as fallbacks; backup recording mandatory in Week 11.

---

## 7. One-liners the team should memorise

- **Vs the base paper:** "AffectDF measures the emotional gap and stops; we measure
  the code-mixing gap under telephony conditions, close it, and deploy it."
- **On self-generated spoofs:** "Held-out unseen tools, an external cross-evaluation
  on AffectDF, a pretrained-detector comparison row, and a low-level-cue sanity
  check — four independent defences."
- **On scale:** "They have 260 hours and H100 clusters; we have an untouched axis, a
  mitigation result, and a phone that beeps. Scale is their contribution; ours is
  the axis and the fix."
