# GIT_RULES.md — codemix-deepfake-detection

Repository workflow rules for the 3-member team: **Lahari (L) · Mounika (M) ·
Sai Krishna (SK)**.

These rules are mandatory. They exist so that (a) faculty can see each member's
weekly contribution at a glance, and (b) the repo itself becomes a portfolio
artifact. Task IDs and owners come from **`PROJECT_PLAN_V2_AFFECTDF.md`** — the
only active plan.

`CLAUDE.md` remains the standing contract for identity handling, the golden
dataset rule, and secrets. Where this file and `CLAUDE.md` overlap, they agree;
where this file is more specific (branch naming, merge cycle), this file wins.

---

## 1. Repository structure and folder ownership

```
codemix-deepfake-detection/
├── README.md                     # overview, architecture, quickstart, results
├── GIT_RULES.md                  # this file
├── CLAUDE.md                     # identity / dataset / secrets contract
├── PROJECT_PLAN_V2_AFFECTDF.md   # the active plan (v2)
├── .env.example                  # names of required keys, never real values
├── requirements.txt
├── configs/                      # every experiment is config-driven
├── src/
│   ├── data/                     # L — download, preprocess, channel sim, manifests
│   ├── models/                   # M — encoder, pooling, head, LoRA
│   ├── training/                 # M — train, evaluate, metrics
│   ├── inference/                # SK — predict, streaming, ONNX export
│   └── utils/
├── live_call/                    # SK — FastAPI server, WebRTC/Twilio, alerts
├── scripts/                      # numbered pipeline entry points (01..07)
├── tests/                        # every src/ module gets at least a smoke test
├── docs/                         # week-wise documentation (see §2)
├── notebooks/                    # exploration only; no pipeline logic
├── experiments/                  # results JSON, figures, gap matrix
├── paper/ , report/              # writing
└── data/ , checkpoints/          # gitignored; links + SHA-256 hashes only
```

**Data and checkpoints never enter git.** Commit Kaggle Datasets/Drive links plus
SHA-256 hashes instead. Never commit `.env`, `.env.git`, keys, model binaries
over 50 MB, or notebook output cells (clear outputs before committing).

## 2. `docs/` — week-wise documentation

One markdown file per member per week:

```
docs/
├── W3_lahari_preprocess_mucs_hiacc.md
├── W3_mounika_affectdf_taxonomy.md
├── W3_krishna_xtts_rvc_pilot.md
├── ...
├── progress.md          # one line per finished task
└── problems_and_decisions.md
```

Each weekly doc is short (half a page is fine) and answers: **what I built, how to
run it, what the numbers/outputs are, what's next.** Written *before* opening the
weekly PR — **the PR is not reviewable without it.**

For an **ALL**-owned task, split the deliverable three ways so each member has
their own commits on their own branch. Never one member committing on behalf of
the team.

## 3. Branch model

**Permanent branches**

- `main` — stable, demo-ready, reviewed code only. **Nobody commits directly. Ever.**
- `dev` — integration branch; the week's PRs land here first.

**Weekly work branches** — one per member per week:

```
week<N>-<name>-<work-topic>
```

Week 3:

```
week3-lahari-preprocess-mucs-hiacc
week3-mounika-affectdf-taxonomy
week3-krishna-xtts-rvc-pilot
```

Rules:

- The topic is **the work, not the person's role**.
- Lowercase, hyphen-separated; no spaces or underscores.
- Branch from the **latest `dev`** at the start of the week. Never branch from
  another member's weekly branch.
- Branches are **never deleted after merge** — the branch list is the timeline.
- **No shared branches.** Pair work (e.g. W3-T4, owner `L+SK`) is split so each
  half lives on its owner's branch; the partner reviews the PR.

> Branches `week1/*`, `week2/*`, `week3/*` (slash-style) predate this file and are
> kept as history. From Week 3 onward the hyphen form above is the rule.

## 4. Weekly merge cycle (completion-based, not calendar-based)

1. **Start of week:** create `week<N>-<name>-<topic>` from the latest `dev`.
2. **During the week:** commit to your own branch every day you work, and push the
   same day — unpushed work does not exist.
3. **When your week's work is complete:** write `docs/W<N>_<name>_<topic>.md`,
   update `experiments/` if you produced numbers, then open a PR → `dev`.
4. **Review:** your reviewer approves (Krishna ↔ Lahari review each other;
   Mounika reviewed by whoever is free). Merge with a **merge commit** — no
   squash (the daily history is the contribution evidence), no rebase on shared
   branches.
5. **When all three weekly PRs are merged and the week's gate passes:** the team
   confirms, then the rotating member opens `dev` → `main` titled
   `Week N: <gate summary>`, merges, and tags:

```bash
git tag -a week3-complete -m "Gate 3: human gates cleared, pools frozen, AffectDF positioning"
git push --tags
```

Rotation for `dev` → `main`: W3 Lahari, W4 Mounika, W5 Krishna, repeat.

If a member finishes early they review others' PRs or read for the paper — they
do **not** start next week's branch until `dev` → `main` for the current week is
merged, so every new branch starts from the same baseline.

## 5. Authorship and pushing from three accounts

Every commit is authored by the **task's owner** per the v2 plan, using
per-command `-c` flags. Never configure a global identity.

```bash
set -a; . ./.env.git; set +a

git -c user.name="$L_GIT_NAME" -c user.email="$L_GIT_EMAIL" commit -m "add mucs segment cutter"
git push "https://$L_GH_USER:$L_GH_PAT@$REPO_URL" week3-lahari-preprocess-mucs-hiacc
```

- `L` = `Lahari-66`, `M` = `Mounika-Reddy-0802`, `SK` = `krishna-2-005`.
- **Never commit or push everything from a single account.** Each branch is
  pushed with its owner's credentials.
- `.env.git` is never printed, echoed, logged, or committed. No `*_PAT` value may
  appear in any output, file, or commit.
- **Use real timestamps.** Never set `GIT_AUTHOR_DATE` or `GIT_COMMITTER_DATE`.
- Keep the hook installed: `sh scripts/install_hooks.sh`.

## 6. Commit quality

**Style:** short, lowercase, imperative, one logical change, ~50 characters.

Good:

```
add mucs segment cutter
fix hiacc child quarantine check
add speaker pool carving script
update channel sim snr levels
add xtts pilot generation script
write affectdf related work notes
xtts pilot: 17/20 clips pass quality bar
```

**Banned — instant PR rejection:**

```
update            changes          final            final2
work done         minor fix        commit           asdf
updated code      week5 work       lahari changes   pushed files
```

Also banned: AI-style phrasing — "Implement comprehensive…", "Refactor
architecture…", "Introduce advanced…", "Enhance robust…", "Add extensive…",
"Optimize pipeline…".

Rules:

- One logical change per commit. A big deliverable is a **branch**, not a commit —
  break it into its natural steps (skeleton → parsing → feature → test → docs).
- The description must let a teammate know what changed without opening the diff.
- **If a commit changes a number, put the number in the message.**
- After each meaningful change: verify the project still runs, review the diff,
  commit, and push before starting the next task.
- Several meaningful commits per working session — not one giant commit, not
  twenty one-liners.
- Commit working code. End-of-session WIP is allowed once
  (`wip: sink schema mismatch, see TODO`) and may not be a branch's final commit.

## 7. No AI / tool attribution — anywhere

The words **Claude, AI, assistant, generated**, or any tool attribution must never
appear in a commit message, PR title, PR body, code comment, docstring, README or
file. **No `Co-Authored-By:` trailers of any kind.**

Safety net: `githooks/commit-msg` (installed via `scripts/install_hooks.sh`)
strips any co-author trailer or "Generated with" line before the commit is
created. Keep it installed.

## 8. Pull requests

**Title:** `[W<N>] <Name>: <work-topic summary>`
e.g. `[W3] Lahari: MUCS/HiACC preprocessing + child-quarantine verification`

**Description template** (`.github/pull_request_template.md`):

```markdown
## What this week's branch delivers
-

## How to run / verify
-

## Numbers produced (link to experiments/ or docs/)
-

## Docs
- [ ] docs/W<N>_<name>_<topic>.md written

## Checks
- [ ] runs end-to-end on a fresh pull of dev
- [ ] ruff check . and ruff format --check . clean
- [ ] pytest green (tests/test_splits.py in particular)
- [ ] no data / secrets / large binaries / notebook outputs committed
- [ ] commit messages follow GIT_RULES §6
```

**PRs go to `dev`, not `main`.** **Never self-merge.**

The reviewer must (15 minutes, not a formality): pull the branch and run the
stated verify step; scan commit messages for §6 compliance; confirm the weekly doc
exists and matches what the code does; leave at least one substantive comment.
Rubber-stamp approvals defeat the purpose — review comments are contribution
evidence too.

## 9. Contribution visibility

This workflow produces four proofs of individual weekly contribution:

1. **Branch list** — `week<N>-<name>-<topic>`, readable as a per-member timeline.
2. **Commit history** — daily, self-describing commits under each member's own name.
3. **`docs/`** — one signed weekly writeup per member.
4. **PRs + reviews** — who built what, who verified what, every week.

GitHub's Insights → Contributors graph reflects true effort only if §6 is
followed, which is exactly why vague bulk commits are banned.

## 10. Conflicts, gates and emergencies

- Merge conflicts are resolved by the branch owner, with the file's area owner
  (§1) consulted — never resolved blind. `docs/progress.md` conflicts are append
  conflicts: **keep all lines.**
- **Ethics gate:** no spoof generation of any kind — including pilots — until a
  signed mentor note exists in `docs/ethics/`. This is not a process preference;
  it is the condition under which the corpora may be used at all.
- **Download gate:** multi-GB pulls start only on an explicit team instruction
  ("run downloads now"), as background jobs logging to `data/raw/logs/`.
- **Hotfix on `main`** (demo-day breakage only): branch `hotfix-<topic>` from
  `main`, PR back to both `main` and `dev`, reviewed by any teammate. Not a
  backdoor around the weekly cycle.
- **If a member's week slips:** the PR still opens when the team closes the week,
  with whatever is done plus the doc stating what is missing; the gap moves to
  next week's branch. An honest partial PR beats a silent missing week.

---

Any rule change requires agreement of all three members and a commit to this file
explaining the change.
