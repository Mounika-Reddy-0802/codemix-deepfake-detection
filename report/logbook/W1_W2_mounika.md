# Log book — Mounika, Weeks 1–2

## Week 1

**W1-T3 — repository bootstrap.** Set up the whole skeleton: package tree under
`src/` with empty `__init__` files so imports stay stable from day one,
`.gitignore` with `.env.git` on line 1 (git identities and PATs must never be
committable), `requirements.txt` with pinned versions, ruff + pytest config in
`pyproject.toml`, CI, and the `commit-msg` hook that strips attribution trailers.
CI installs only a light subset — ruff, pytest, numpy, pandas, pyyaml — so lint
and tests stay fast; the heavy ML wheels are for training boxes only. That
constraint is why nearly every module imports torch lazily.

**W1-T4 — environment check notebook.** `notebooks/env_check.ipynb` loading
wav2vec2-base, XLSR-53 and WavLM-base, running a forward pass, and initialising
a W&B run.

**W1-T7 (ALL, my third) — related-work scaffolding.**

## Week 2

**W2-T2 — dataset, splits and metrics.** `src/data/dataset.py` (map-style dataset
over a manifest CSV plus a padding collate, torch imported only inside
`collate_fn`), `src/data/build_manifests.py` (manifest schema, speaker-disjoint
splitting, the anti-leakage checklist) and `src/training/metrics.py` (EER, ROC-AUC,
F1, bootstrap CIs — all pure numpy, no sklearn).

The decision I would defend at a viva: a **cloned voice of speaker X carries
speaker id X**, not the tool's id. Without that convention, speaker-disjointness
silently fails the moment spoofs enter a manifest, because the clone looks like a
different speaker to the split logic while sounding like the same person to the
model. `tests/test_splits.py` pins it.

## Reflection

Writing the metrics by hand rather than pulling sklearn felt slow at the time, but
it forced the score convention to be explicit — higher = more bonafide, label 1 =
bonafide — in one place. Every later module (predict, evaluate) now cites that one
line, which is the sort of thing that silently inverts a whole results table if
it is left implicit.

**Blocked on:** W&B project, HF token, Kaggle GPU verification on all three
accounts; ASVspoof 2019 LA not yet downloaded.
