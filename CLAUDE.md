# Project working rules (read before committing anything)

This file is the standing contract for how work is committed to this repo. It
applies to every contributor and every task, every week.

---

## 1. Git identity rules (CRITICAL — follow exactly)

Every task in `PROJECT_PLAN_12_WEEKS.md` has an owner code (**L**, **M**, **SK**,
or **ALL**). **All git activity for a task is done under that owner's identity
only.** Never configure a global git identity; always use per-command `-c` flags
so identities never bleed between tasks.

Identities and fine-grained GitHub tokens live in **`.env.git`** (gitignored,
line 1 of `.gitignore`). Source it at the start of a work session:

```bash
set -a; . ./.env.git; set +a
```

`.env.git` defines, for each member `<X>` in {`L`, `M`, `SK`}:
`<X>_GIT_NAME`, `<X>_GIT_EMAIL`, `<X>_GH_USER`, `<X>_GH_PAT`, plus `REPO_URL`.

Commit as the task owner:

```bash
git -c user.name="$L_GIT_NAME" -c user.email="$L_GIT_EMAIL" commit -m "feat: ..."
```

Push as the task owner (token is injected inline, never stored in a remote):

```bash
git push "https://$L_GH_USER:$L_GH_PAT@$REPO_URL" week1/lahari-download-scripts
```

- **`.env.git` is never printed, echoed, logged, or committed.** No `*_PAT`
  value may appear in any command output, file, or commit.
- GitHub disabled password auth for git in 2021 — `*_GH_PAT` **must** be a
  fine-grained personal access token (Contents: read/write, Pull requests:
  read/write), not an account password.

## 2. "ALL"-owned tasks

Split the deliverable into three parts and commit each part under a different
member, so all three contribute (e.g. one section each of a shared doc).

## 3. Branching & PRs

- One branch per task: `week<N>/<firstname>-<short-task-name>`
  (e.g. `week1/lahari-download-scripts`).
- Roughly **2–4 focused commits per task** — real incremental work, not one
  giant commit and not 50 trivial ones.
- Push the branch and open a PR. **Do not self-merge to `main`.** A different
  member reviews and merges — PR review is a project requirement.
- Print the GitHub compare/PR URL when a branch is pushed.

## 4. No AI / tool attribution — anywhere

The words **Claude, AI, assistant, generated**, or any tool attribution must
**never** appear in any commit message, PR title, PR body, code comment,
docstring, README, or file. **No `Co-Authored-By:` trailers of any kind.**
Commit messages are conventional style ("feat: ...", "docs: ...", "test: ...")
in plain first-person engineering voice.

Safety net: `.git/hooks/commit-msg` (tracked master copy in `githooks/`,
installed via `scripts/install_hooks.sh`) strips any `Co-Authored-By:` line or
line containing "Generated with" before a commit is created. Keep it installed.

## 5. Progress log

After finishing each task, append one line to `docs/progress.md`
(date, task ID, owner, branch, one-sentence summary) and commit it under the
same owner.

---

## 6. The golden dataset rule (audit-critical)

- **Stage-1 trains on ASVspoof 2019 LA (English) ONLY.** This is what makes the
  gap matrix a measurement of true generalisation.
- **IndicSynth, IndicTTS-Deepfake, IndicVoices, HiACC, and Tortoise clones are
  evaluation-only** — they never appear in any training manifest.
- **Tortoise never appears in training manifests, ever** (held-out unseen attack).
- **HiACC child audio is never used for anything** — not as bonafide, not as a
  cloning reference. The child folders are quarantined into
  `data/raw/hiacc/_EXCLUDED_children/` on extraction.
- Only **Stage-3 (LoRA adaptation)** touches code-mixed training data, and only
  from **speaker-disjoint** MUCS slices + matching XTTS-v2 clones.
- ASVspoof 2021 DF is skipped (storage); revisit only if college storage arrives.

Enforced mechanically by `tests/test_splits.py` (the anti-leakage checklist).
`test_splits.py` runs before every training launch.

## 7. Secrets

Runtime secrets (W&B, Hugging Face, Twilio) live only in `.env` (gitignored);
`.env.example` documents every variable with placeholders. Git identities/PATs
live only in `.env.git`. Never hardcode a secret in code, notebook, or history.

## 8. Current-week protocol

Work **strictly week by week**. Only the current week's tasks (as stated at the
start of the session) may be worked on, plus explicitly carried-over tasks.

- **We are in Week 1.** Scope: W1-T1 … W1-T7 (setup, datasets, accounts, ethics).
- **Twilio stays off until Week 6.** Weeks 1–5 use the free WebRTC (`aiortc`)
  harness. Do not create or activate the Twilio trial before Week 6 — its 30-day
  window is timed to cover Weeks 6–9.
- Large multi-GB downloads run only on explicit instruction ("run downloads
  now"), as background jobs logging to `data/raw/logs/`.

## 9. Code quality bars

- Python 3.10+, type hints, docstrings, **ruff-clean** (`ruff check .` and
  `ruff format --check .`).
- Every module in `src/` gets at least a smoke test in `tests/`.
- Notebooks contain **no pipeline logic** — anything that produces a paper
  number lives in `src/` behind a config.
