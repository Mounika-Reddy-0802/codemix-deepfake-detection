# HUMAN_TODO — things only the team can do

Week 1 code is done and committed on per-task branches under the correct
identities. The items below need a human. **Bold = blocks pushing / Week 2.**

## 0. Unblock git push (nothing is on GitHub yet)

All commits exist locally; **pushing is blocked** until these are done:

- [ ] **Create the GitHub repo** `codemix-deepfake-detection` (Private). Do **not**
      initialise it with a README/.gitignore/licence (we already have them).
      Confirm the owner — `.env.git` currently assumes
      `github.com/Mounika-Reddy-0802/codemix-deepfake-detection`; change `REPO_URL`
      if it lives elsewhere (e.g. an org).
- [ ] **Add the other two members as collaborators** with write access
      (`Lahari-66`, `krishna-2-005`).
- [ ] **Create a fine-grained PAT per member** and paste into `.env.git`
      (replace the `REPLACE_WITH_*` placeholders). Scope: the repo only;
      Contents = Read/write, Pull requests = Read/write. Account passwords **do
      not work** for git (GitHub removed that in 2021) — it must be a token.
- [ ] **SECURITY:** the three GitHub account passwords were shared in chat —
      **change all three passwords and enable 2FA** now.
- [ ] After cloning elsewhere, install the hook once: `sh scripts/install_hooks.sh`.

Once PATs are in `.env.git`, push each branch as its owner, e.g.:
```bash
set -a; . ./.env.git; set +a
git push "https://$L_GH_USER:$L_GH_PAT@$REPO_URL"  week1/lahari-download-scripts
git push "https://$L_GH_USER:$L_GH_PAT@$REPO_URL"  week1/lahari-streaming-licences
git push "https://$M_GH_USER:$M_GH_PAT@$REPO_URL"   main
git push "https://$M_GH_USER:$M_GH_PAT@$REPO_URL"   week1/mounika-env-check
git push "https://$SK_GH_USER:$SK_GH_PAT@$REPO_URL" week1/saikrishna-webrtc-harness
git push "https://$SK_GH_USER:$SK_GH_PAT@$REPO_URL" week1/saikrishna-live-call-design
# team task — push under whichever identity you like; commits keep their authors:
git push "https://$M_GH_USER:$M_GH_PAT@$REPO_URL"   week1/team-related-work-ethics
```
Then open a PR for each branch (compare URLs printed in the session report).

## 1. Accounts & compute (W1-T3/T4)

- [ ] Verify **Kaggle** accounts (all 3) + GPU access (30 hrs/wk each).
- [ ] Verify **Google Colab** works (GPU runtime).
- [ ] Create the **Weights & Biases** project (free tier); put `WANDB_API_KEY`
      in `.env` / Kaggle Secrets. Run `notebooks/env_check.ipynb` — confirm the
      three encoders load and a W&B run appears.
- [ ] Create/verify **Hugging Face** account + token (`HF_TOKEN`); accept dataset
      terms on the HF pages (IndicVoices is **gated**).
- [ ] Create the **HF Spaces** placeholder (SK).
- [ ] **Send the college GPU-access request email** (M) — even though the reply
      comes later, send it in Week 1.

## 2. Ethics & licences (W1-T7 / L)

- [ ] Mentor reviews `docs/licences.md` + `docs/ethics/README.md` and **signs the
      ethics note**. Scan to `docs/ethics/mentor_signoff_<date>.pdf` and log it.
      **No spoof generation (incl. Week-2 XTTS pilots) until this exists.**
- [ ] (Bonus) Send the **IITG-HingCoS** access email
      (`docs/outreach/iitg_hingcos_request.md`) — not on the critical path.

## 3. Literature (W1-T7 / ALL)

- [ ] Each member writes their **~6 paper summaries** (3–4 sentences each) into
      `docs/related_work_notes.md`: M → 1,2,3,6,7,8; L → 10,11,12,13,14,15;
      SK → 4,5,9,16,17. Verify each citation's exact DOI/arXiv id.

## 4. PR review (project requirement)

- [ ] Each member opens GitHub and **reviews + merges one teammate's PR** (never
      your own). Suggested merge order: `main` is already the base; merge the six
      task PRs in any order. `docs/progress.md` will have a trivial append
      conflict between branches — keep **all** lines when resolving.

## 5. Log book (weekly deliverable)

- [ ] Each member writes their **Week-1 log-book entry** in `report/logbook/`.

## 6. Verify before the big downloads (W1-T1 / L)

- [ ] Confirm the pinned versions in `requirements.txt` install on the Kaggle/Colab
      base image (adjust `torch`/`torchaudio` to the image's CUDA build if needed).
- [ ] **HiACC child quarantine:** the download script auto-moves folders matching
      `child|kid|minor`. **Verify this matches HiACC's real adult/child layout**
      before trusting it — if the split is by manifest/filename, quarantine
      manually. The child data must never be used.
- [ ] Only when the team says **"run downloads now"** should the multi-GB pulls
      start: `nohup bash scripts/01_download_data.sh --run > data/raw/logs/download_main.log 2>&1 &`

## 7. Connectors (optional, not needed for Week 1)

- [ ] The claude.ai Gmail / Google Calendar / Google Drive connectors are
      unauthorized in this environment. Authorize them in claude.ai connector
      settings if you want them available later.

---

# WEEK 2 — human-only steps (code is done + pushed)

Week 2 code is on `week2/lahari`, `week2/mounika`, `week2/saikrishna`. These need a
human (data, a GPU, or ears):

- [ ] **Ethics sign-off must exist before any generation** (carried over from Week 1).
      No XTTS pilots until `docs/ethics/` has the signed note.
- [ ] Install the full ML stack + **ffmpeg** on the training box (ffmpeg is needed
      for AMR-NB; without it channel_sim falls back to G.711 only).
- [ ] **Run downloads** (say "run downloads now"), then
      `bash scripts/02_preprocess_all.sh` to produce 16 kHz clean segments.
- [ ] **L:** run channel simulation on ~20 sample clips and **listen** — confirm the
      G.711/AMR + SNR output sounds like telephony audio (a paper-quality check).
- [ ] **SK+L:** from preprocessed MUCS/HiACC-adult, build the clip index
      (speaker, source, duration) and run `select_reference_speakers` to pick 30–50
      adult speakers (≥30 s each); extract their code-mixed transcripts.
- [ ] **SK+M:** carve the MUCS speakers into eval vs Stage-3 adaptation pools with
      `build_manifests.carve_pools` (done in code) and freeze the pool lists.
- [ ] **SK:** set up Coqui XTTS-v2 (Colab/Kaggle GPU) and generate **20 pilot clones**
      via `scripts/03_generate_spoofs.sh --pilot`.
- [ ] **ALL:** 1-hour session — listen to the pilot clones + channel-sim audio and
      **agree the quality bar** for the spoof set; record the decision.
- [ ] Each member: Week-2 log-book entry + review/merge a teammate's Week-2 PR.

---

# WEEK 3 — HUMAN_TODO (v2 plan, `PROJECT_PLAN_V2_AFFECTDF.md`)

Week 3 code is on `week3-lahari-preprocess-mucs-hiacc`,
`week3-mounika-affectdf-taxonomy` and `week3-krishna-xtts-rvc-pilot`, all branched
from `dev`. Everything below needs a human — data, a browser, a GPU, ears, or a
signature. **Ordered by what unblocks the most.**

## 0. BLOCKING EVERYTHING — GitHub credentials

**Nothing can be pushed.** All three `*_GH_PAT` values in `.env.git` are still the
`REPLACE_WITH_*` placeholders, so every push fails with
*"Invalid username or token. Password authentication is not supported."*

- [ ] **Create a fine-grained PAT per member** (Contents: read/write, Pull
      requests: read/write, scoped to this repo only) and paste into `.env.git`.
      An account password will not work — GitHub removed that in 2021.
- [ ] Confirm `Lahari-66` and `krishna-2-005` still have write access to
      `Mounika-Reddy-0802/codemix-deepfake-detection`.
- [ ] Then push, each branch under its own account (commands in the session report).

## 1. ~~BLOCKING ALL GENERATION~~ — mentor ethics sign-off (W3-T1) — **CLEARED**

- [x] Mentor reviewed `docs/licences.md` + `docs/ethics/README.md` and **signed the
      ethics note on the REVISED v2 scope**: larger spoof volume (~4,000+ XTTS
      utterances), **RVC added** as a second attack family, and **AffectDF
      cross-evaluation** (CC BY-NC 4.0, research use).
- [x] Scanned to `docs/ethics/mentor_signoff_2026-08-12.pdf` (12 Aug 2026).

`python -m src.data.ethics_gate` exits 0: *"ethics gate OPEN: signed note found
(mentor_signoff_2026-08-12.pdf)"*. The XTTS pilot, the RVC pilot, Tortoise and
Week-4 generation are all permitted.

> The signed PDF is **gitignored** (`.gitignore` line 15 — it carries real
> signatures), so it does not appear in `git status` and a fresh clone will not
> have it. **Run the gate, do not read this checklist**, when you need to know
> whether generation is permitted — and copy the PDF into `docs/ethics/` by hand
> on any new machine (Colab included; the notebook's step 3 does this from Drive).

**Generation is now gated on compute, not ethics** — see section 8.

## 2. Review + merge the open PRs (W3-T1)

- [ ] Each member reviews and merges one teammate's PR. Never your own.
      Krishna ↔ Lahari review each other; Mounika reviewed by whoever is free.
- [ ] PRs target **`dev`**, not `main` (`GIT_RULES.md` section 4).
- [ ] `docs/progress.md` will have an append conflict between branches —
      **keep all lines** when resolving.

## 2b. Machine state (17 Aug 2026)

| | dev laptop | GPU laptop |
|---|---|---|
| MUCS / HiACC | extracted | **extracted, counts identical** |
| ASVspoof 2019 LA | 25 % partial | **absent — Stage-1 cannot train** |
| CUDA | none | torch 2.8.0+cu128, RTX 3050 6 GB |
| Tests | green | **green (410 passed, 2 skipped)** |
| Ethics gate | open | **open** |

`.env.git` exists on the GPU laptop but **all three `*_GH_PAT` values are still
`REPLACE_WITH_*` placeholders**, so per-owner pushes still cannot run — see §0.

## 3. Downloads (W3-T2) — only on "run downloads now"

- [ ] Say the word, then:
      `nohup bash scripts/01_download_data.sh --run > data/raw/logs/download_main.log 2>&1 &`
      (ASVspoof 2019 LA 7.12 GB, MUCS ~7.7 GB, HiACC ~532 MB.)
- [ ] Confirm the pinned versions in `requirements.txt` install on the Kaggle/Colab
      base image; adjust `torch`/`torchaudio` to the image's CUDA build if needed.
- [ ] Install **ffmpeg** on the training box — without it channel sim silently
      falls back to G.711 and the AMR-NB condition is never really tested.

## 4. HiACC child quarantine — verify BY HAND (W3-T2, ethics-critical)

The preprocessing walk has been fixed so quarantined folders are never read, and
`python -m src.data.quarantine` now writes the evidence report. **The tool cannot
close this gate** — it pattern-matches; a human confirms the real layout.

- [ ] After download: `python -m src.data.quarantine --root data/raw/hiacc`
- [ ] Read HiACC's documentation and record how adult and child recordings are
      **actually** distinguished — folder, manifest column, or filename field.
      If the folder regex quarantined nothing but the audit reports age-like
      manifest columns, **the split is by metadata and must be done by hand.**
- [ ] Confirm the adult count matches the published HiACC adult subset size.
- [ ] Sign `docs/qa/child_quarantine_check.md`.

## 5. Listening tests (W3-T2) — ears required

- [x] ~~`bash scripts/02_preprocess_all.sh`~~ — superseded. MUCS is a Kaldi corpus
      (521 long recordings, speakers in `utt2spk`), so blind chunking destroyed
      speaker identity. Replaced by `python -m src.data.corpora`, which indexes
      both corpora by their real speaker ids.
- [x] `python -m src.data.listening_test` → 20 clean/channel pairs + rating sheet.
- [x] **L, M, SK listened to all 20 pairs (13 Aug 2026).** Verdict: clearly
      telephony, fully intelligible in both languages. Telephony 4.0/5,
      intelligibility 4.0/5, 60/60 rows rated. Written up in
      `docs/qa/channel_sim_verdict.md`; objective measurement passed 20/20.
- [x] **AMR-NB objective half done (13 Aug 2026).** ffmpeg 9.0 installed
      (`winget install Gyan.FFmpeg`), so `channel_sim` runs real AMR-NB instead of
      falling back. 20/20 pass. AMR-NB is materially harsher than G.711: 5.57 dB
      SNR vs 17.62, correlation 0.885 vs 0.991. Numbers in
      `docs/qa/channel_sim_verdict.md` §3.
- [ ] **AMR-NB listening pass outstanding** — L, M, SK rate the 20 pairs at
      `<DATA_ROOT>/processed/listening_test_amr/` into
      `docs/qa/channel_sim_listening_sheet_amr.csv`. Until then an AMR-NB column
      is verified by measurement only and must not be reported as ear-checked.

> **ffmpeg note for the other two machines.** `winget install Gyan.FFmpeg` gives
> the CLI (enough for AMR-NB). If you also generate spoofs locally you need
> `winget install Gyan.FFmpeg.Shared` too — `torchcodec`, which the TTS stack
> requires from torch 2.9, loads FFmpeg's shared DLLs and the default build is
> statically linked. Put its `bin/` on PATH.

## 6. Speaker selection + pool freeze (W3-T4)

- [x] Ranking built over **520 MUCS speakers**; top 50 shortlisted (8.34 hours).
- [x] **Pools FROZEN**: 25 train / 10 adaptation / 15 eval, zero speakers in more
      than one pool. SHA-256 `f57e0d85dbd3c8f5b96ad6af59edae7218104b0167807eae6433314947063e7b`,
      recorded in `docs/progress.md`, `data/manifests/speaker_pools.csv` committed.
- [ ] **Confirm the 50 by ear before Week-4 generation.** The shortlist was ranked
      on a dynamic-range proxy (24.5–92.4 dB spread), *not* a calibrated SNR, and
      the eligibility gate passed all 520 speakers — so the ordering is a hint,
      not a quality guarantee. Spot-check the top of
      `data/manifests/speaker_shortlist.csv`; re-freeze if any are unusable.

## 7. Accounts and compute (W3-T6)

Run `python -m src.utils.env_check` — it reports what it can and labels the rest
`manual`. Current state on the dev machine: **ok=3, missing=7, manual=8**.

- [ ] Kaggle GPU (30 hrs/wk) verified on **all three** accounts.
- [ ] W&B project created; `WANDB_API_KEY` in `.env` / Kaggle Secrets.
- [ ] HF token created; `HF_TOKEN` set; **gated dataset terms accepted** for
      IndicVoices and IndicTTS-Deepfake (browser-only).
- [ ] Colab GPU runtime confirmed.
- [ ] Paste every link into the resource table in `README.md`.

## 8. Pilots — XTTS DONE on the GPU laptop, waiting on ears (W3-T5)

**Platform: the team's own GPU laptop** (RTX 3050 6 GB), 17 Aug 2026 — replaces
Colab, which replaced Kaggle. No quota, no idle disconnect, and the corpora are
already there. `notebooks/pilot_xtts_colab.ipynb` still works if Colab is wanted.

- [x] Corpora extracted with the child quarantine; counts match the dev laptop
      exactly (MUCS 52,825/520, HiACC 3,318/24, 1,858 quarantined).
- [x] **40 clips generated, 0 failures.** Not the original 4-cell matrix: the
      team fixed the script to Latin (P-014), so the pilot became a matched A/B —
      the same speaker saying the same sentence under the same language tag,
      once from romanised Hinglish and once from Devanagari. That isolates the
      one variable still in question.
- [ ] **All three rate all 40.** Blind sheet at
      `docs/qa/pilot_script_rating_sheet.csv` (40 clips × 3 raters = 120 rows),
      audio staged under neutral names in `<DATA_ROOT>/generated/pilot_ab/clips`.
      Score with `src.data.pilot_rating.summarise_ab` against the withheld key —
      **do not open `pilot_script_answer_key.csv` before rating.**
- [ ] Record the remaining decisions in `docs/problems_and_decisions.md`
      (language tag, quality bar, RVC viability). Script is settled by P-014.
- [ ] RVC pilot: 1 speaker model, 10 conversions. **Not started.**

> **What the pre-screen already says.** Median speech rate is 13.9 chars/sec for
> Devanagari against 14.4 for romanised, and paired durations track the ~13 %
> text expansion — so romanisation does not appear to rush or truncate speech.
> That is a measurement, not a verdict; it tells you the A/B is worth listening
> to rather than which side wins.
>
> **One clip of 40 came out at 0.83 s from a 150-character transcript** and
> raised nothing. Generation catches crashes, not empty output. Week 4 needs the
> W4-T6 duration filter *before* the 4,000-clip run.

> **Compute-plan consequence of moving to Colab.** The v2 plan sizes Weeks 5–9
> around *3 Kaggle accounts × 30 GPU-hours/week*, with S1 and S3 training in
> parallel on separate accounts (W5-T3). Colab has no comparable weekly quota and
> disconnects idle sessions, so that parallelism no longer holds. **Revisit the
> Week-5 training plan before launching S1/S3** — decide whether to buy Colab Pro,
> keep Kaggle purely for the long training runs, or serialise S1 then S3.

> **ASVspoof via Colab too.** The local download stalled at 25% (1.75 GB of
> 7.11 GB) — `datashare.ed.ac.uk` resets the connection after sustained
> transfers. Since S1 trains on Colab anyway, fetch it there or from a Drive
> mirror instead of fighting the throttle.

## 9. Log book

- [x] Week 1–2 entries written for all three members (`report/logbook/`).
- [ ] Week-3 entries after the gates clear.

## 10. Housekeeping

- [ ] **`Research/` (72 MB of PDFs) is untracked and uncommitted.** Decide: add to
      `.gitignore`, or move to shared Drive and link it. It should not go into git.
- [ ] **`parked/v1-week4-superseded`** is a local branch holding work written
      against the old plan (dataset-v1 assembly, LR schedule + resume, evaluate,
      offline inference, FastAPI skeleton). Under v2 that maps to W4-T4, W5-T1 and
      W5-T5. Not pushed. Cherry-pick from it when those weeks arrive, or delete it.
- [ ] **SECURITY (carried over from Week 1):** the three GitHub account passwords
      were shared in chat — change all three and enable 2FA.
