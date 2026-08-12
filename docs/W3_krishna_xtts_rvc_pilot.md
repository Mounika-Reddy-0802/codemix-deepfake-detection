# W3 — Krishna: ethics gate + pilot harness + speaker pool freeze

Branch: `week3-krishna-xtts-rvc-pilot` · Tasks: **W3-T5** (blocked), **W3-T4** (freeze half)

## What I built

**1. `src/data/ethics_gate.py` — the rule made mechanical.**
"No spoof generation, including pilots, until the mentor signs" has been in the
plan since Week 1, but it was enforced by whoever remembered it. Now every
generation entry point calls `require_signoff()` *before a model is loaded*, so a
blocked run fails in milliseconds with a readable reason instead of after a
20-minute XTTS download.

Three deliberate properties, each with a test:

- **No override.** No `--force`, no `SKIP_ETHICS=1`. The module does not import
  `os` at all, so there is no env-var route past it. A bypass flag would make the
  gate decorative.
- **`docs/ethics/README.md` does not count.** It describes what the sign-off must
  cover; only a scanned artefact matching `mentor_signoff*.{pdf,png,jpg}` opens
  the gate.
- **An empty placeholder does not count** — a zero-byte `mentor_signoff.pdf` is
  not a signature.

Wired into `load_xtts`, `generate_batch`, `load_tortoise` and
`generate_heldout_batch`; a parametrised test asserts each one still calls it, so
a future refactor cannot quietly drop the gate.

**2. `src/data/speaker_pools.py` — three pools, frozen and hashed.**
The v2 plan needs **train / adaptation / eval** (the Week-2 `carve_pools` gave
two). Carve is deterministic per seed, every speaker lands in exactly one pool,
and the rounding remainder goes somewhere rather than being dropped.

The part that matters later is `freeze()` + `verify_frozen()`: the CSV is hashed
(SHA-256 over the canonical row order, so re-sorting does not change the hash) and
verified against that hash before use. If a speaker ever moves pool after
generation has started, a clone of a training voice ends up in the eval column and
every generalisation number in the paper is void. The test
`test_verify_fails_when_a_speaker_moved_pool` is that scenario.

**3. `notebooks/spoof_quality_audit.ipynb` — the pilot protocol, not the pilot.**
Cell 1 checks the gate and stops. The rest is the rating design: the 20-clip XTTS
matrix (Devanagari vs romanised × `hi` vs `en` tag), the 10 RVC conversions, the
three rating axes, and the four decisions the pilot has to produce. Outputs
cleared; no pipeline logic in the notebook.

## How to run it

```bash
python -m src.data.ethics_gate          # prints gate status, exits 1 when closed
python -m src.data.speaker_pools --shortlist data/manifests/speaker_ranking.csv
python -m src.data.speaker_pools --verify <sha256>   # confirm pools unchanged
```

## Numbers

- **39 tests** added: `test_ethics_gate.py` (20), `test_speaker_pools.py` (19).
  All green; `ruff` clean.
- **Ethics gate status: CLOSED.** `python -m src.data.ethics_gate` exits 1.
  `test_real_repo_gate_is_currently_closed` asserts this — when the mentor signs,
  that test is the one to update.
- **0 clips generated.** No XTTS pilot, no RVC pilot, no Tortoise. That is the
  correct outcome this week, not a shortfall.
- **0 speakers pooled.** The carve code is ready and tested; it needs the
  shortlist, which needs downloads + the team listening pass.

## What's next (blocked, in order)

1. **Mentor ethics sign-off** — blocks everything below. Scan to
   `docs/ethics/mentor_signoff_<date>.pdf`.
2. **Downloads + preprocessing** (L) → the clip index → the ranking → the team
   listening pass → the shortlist.
3. **Then** carve and freeze the pools, commit `data/manifests/speaker_pools.csv`,
   and record its SHA-256 in `docs/progress.md`.
4. **Then** the XTTS + RVC pilots, and the 30-clip rating session.

If the sign-off has not happened by end of Wednesday, the plan says escalate to
the mentor in person rather than let Week 4 slip.
