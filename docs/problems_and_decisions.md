# Problems & decisions (ADR log)

Numbered decisions and the problems that motivated them, versioned alongside the
code. Newest entries can be appended; existing ones are not rewritten.

| ID | Decision | Why |
|----|----------|-----|
| **P-001** | Stage-1 trains on **ASVspoof 2019 LA (English) only** | The English → Hindi/Tamil → code-mixed gap is only a *measurement of generalisation* if nothing else leaks into training. Enforced by `tests/test_splits.py`. |
| **P-002** | **HiACC child audio excluded entirely** — never bonafide, never a cloning reference, never in a manifest | Ethics. Quarantined into `_EXCLUDED_children/` on extraction; checked in `build_manifests` + `select_reference_speakers` + `test_splits`. |
| **P-003** | **Tortoise held-out tool lives in `heldout_tts.py`**, separate from `spoof_generation.py` | Keeps the unseen-attack tool firewalled *and* lets L (scale) and SK (held-out) edit different files in the same week without merge conflicts. |
| **P-004** | **Heavy imports are lazy** (torch/transformers/TTS/torchaudio/librosa/silero inside functions) | CI installs only ruff/pytest/numpy/pandas/pyyaml and stays fast + green; modules still import for pure-logic unit tests. |
| **P-005** | **Metrics in pure numpy** (custom EER, rank AUC, F1, bootstrap CI) — no sklearn | Removes a CI dependency; results are exact and unit-tested against known cases. |
| **P-006** | **G.711 μ-law + SNR mixing in pure numpy**; AMR-NB via ffmpeg with G.711 fallback | The lossy pieces that matter for the paper are exact + testable; AMR-NB stays optional so the pipeline runs without ffmpeg. |
| **P-007** | **One branch per person per week** (`week<N>/<firstname>`), merged into `main` after the week | Individual contribution is graded from git history; merging keeps `main` complete so the next week branches off a full tree. |
| **P-008** | **Pushes run in the background** | Pushing from the OneDrive-synced working copy is slow (sync contends on `.git` pack files); a foreground push can exceed a 5-min timeout while the same push completes in the background. |
| **P-009** | **Twilio activated in Week 6, not before** | The 30-day trial window is timed to cover Weeks 6–9 (integration + demo). Weeks 1–5 use the free WebRTC harness. |
