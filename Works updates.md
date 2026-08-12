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

## 1. BLOCKING ALL GENERATION — mentor ethics sign-off (W3-T1)

- [ ] Mentor reviews `docs/licences.md` + `docs/ethics/README.md` and **signs the
      ethics note on the REVISED v2 scope**: larger spoof volume (~4,000+ XTTS
      utterances), **RVC added** as a second attack family, and **AffectDF
      cross-evaluation** (CC BY-NC 4.0, research use).
- [ ] Scan to `docs/ethics/mentor_signoff_<date>.pdf`.

Verify the gate opens: `python -m src.data.ethics_gate` must exit 0. Until then
the XTTS pilot, the RVC pilot, Tortoise, and all Week-4 generation refuse to run —
there is no override flag.

**Plan escalation rule:** if this has not happened by **end of Wednesday**,
escalate to the mentor in person. Every downstream week slips otherwise.

## 2. Review + merge the open PRs (W3-T1)

- [ ] Each member reviews and merges one teammate's PR. Never your own.
      Krishna ↔ Lahari review each other; Mounika reviewed by whoever is free.
- [ ] PRs target **`dev`**, not `main` (`GIT_RULES.md` section 4).
- [ ] `docs/progress.md` will have an append conflict between branches —
      **keep all lines** when resolving.

## 3. Downloads (W3-T2) — only on "run downloads now"

- [ ] Say the word, then:
      `nohup bash scripts/01_download_data.sh --run > data/raw/logs/download_main.log 2>&1 &`
      (ASVspoof 2019 LA ~23 GB, MUCS ~7.7 GB, HiACC ~532 MB.)
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

- [ ] `bash scripts/02_preprocess_all.sh` (refuses HiACC if the audit warns).
- [ ] `python -m src.data.listening_test` → 20 clean/channel pairs + rating sheet.
- [ ] **L, M, SK each listen to all 20 pairs** and fill in
      `docs/qa/channel_sim_listening_sheet.csv`. Confirm the channel output sounds
      like a phone call, not like broken audio. This validates a load-bearing
      claim in the paper.

## 6. Speaker selection + pool freeze (W3-T4)

- [ ] `python -m src.data.speaker_selection --root data/processed/clean/mucs2021 --source mucs2021`
- [ ] **Team listening pass** over the top of the ranking; confirm the final
      30–50 adult speakers. The shortlist is never padded to reach 30 — if the
      corpus comes up short, that is a real finding, not a bug to work around.
- [ ] `python -m src.data.speaker_pools --shortlist data/manifests/speaker_ranking.csv`
- [ ] **Record the printed SHA-256** in `docs/progress.md` and commit
      `data/manifests/speaker_pools.csv`. Pools may not move after this.

## 7. Accounts and compute (W3-T6)

Run `python -m src.utils.env_check` — it reports what it can and labels the rest
`manual`. Current state on the dev machine: **ok=3, missing=7, manual=8**.

- [ ] Kaggle GPU (30 hrs/wk) verified on **all three** accounts.
- [ ] W&B project created; `WANDB_API_KEY` in `.env` / Kaggle Secrets.
- [ ] HF token created; `HF_TOKEN` set; **gated dataset terms accepted** for
      IndicVoices and IndicTTS-Deepfake (browser-only).
- [ ] Colab GPU runtime confirmed.
- [ ] Paste every link into the resource table in `README.md`.

## 8. Pilots — AFTER the sign-off only (W3-T5)

- [ ] XTTS-v2 pilot: 20 clips (Devanagari vs romanised script, `hi` vs `en` tag).
- [ ] RVC pilot: 1 speaker model, 10 conversions.
- [ ] **All three rate all 30** and agree the quality bar; record the four
      decisions in `docs/problems_and_decisions.md`
      (script, language tag, quality bar, RVC viability).

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
