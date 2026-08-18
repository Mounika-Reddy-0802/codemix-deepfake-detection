# W3 — Krishna: ethics gate, pool freeze, and the XTTS pilot

Branch: `week3-krishna-xtts-rvc-pilot` · Tasks: **W3-T5**, **W3-T4** (freeze half)

**Outcome:** the gate opened, the pools froze, and 40 clips generated on real GPU
with zero failures. The pilot's script question got an answer; its language-tag and
quality-bar questions still need ears. RVC did not run — blocked on toolchain.

---

## What I built

### 1. `src/data/ethics_gate.py` — the rule made mechanical

"No spoof generation, including pilots, until the mentor signs" had been in the plan
since Week 1, enforced by whoever remembered it. Now every generation entry point
calls `require_signoff()` **before a model is loaded**, so a blocked run fails in
milliseconds with a readable reason instead of after a 20-minute XTTS download.

Three deliberate properties, each with a test:

- **No override.** No `--force`, no `SKIP_ETHICS=1`. The module does not import `os`
  at all, so there is no env-var route past it. A bypass flag makes a gate decorative.
- **`docs/ethics/README.md` does not count.** It describes what the sign-off must
  cover; only a scanned artefact matching `mentor_signoff*.{pdf,png,jpg}` opens it.
- **An empty placeholder does not count** — a zero-byte PDF is not a signature.

Wired into `load_xtts`, `generate_batch`, `load_tortoise` and
`generate_heldout_batch`; a parametrised test asserts each still calls it, so a
refactor cannot quietly drop the gate.

### 2. `src/data/speaker_pools.py` — three pools, frozen and hashed

The v2 plan needs **train / adaptation / eval** (Week 2's `carve_pools` gave two).
The carve is deterministic per seed, every speaker lands in exactly one pool, and
the rounding remainder goes somewhere rather than being dropped.

What matters later is `freeze()` + `verify_frozen()`: the CSV is hashed (SHA-256
over canonical row order, so re-sorting does not change the hash) and verified
before use. If a speaker moves pool after generation starts, a clone of a training
voice lands in the eval column and every generalisation number is void.
`test_verify_fails_when_a_speaker_moved_pool` is exactly that scenario.

### 3. `src/data/transliteration.py` — the corpus in one script

Team decision (**P-014**): the corpus is written in Latin. Hand-rolled rather than
`indic-transliteration`, for two load-bearing reasons — CI installs only
ruff/pytest/numpy/pandas/pyyaml, and the spelling is an *input to XTTS*, so a
library upgrade changing one vowel would generate half the corpus from a different
romanisation, invisibly.

Targets Hinglish, not IAST: digraphs, no diacritics, and both schwa-deletion rules,
because an extra written vowel is an extra **syllable** to XTTS (`karnaa`, not
`karanaa`).

### 4. `src/data/pilot_rating.py` — the blind A/B sheet

Same speaker, same sentence, same language tag, differing only in script. Blinding
took three passes and the first two only *looked* blind — see Numbers below.

---

## Numbers

| | |
|---|---|
| Ethics gate | **OPEN** — signed 12 Aug 2026, `ethics_gate` exits 0 |
| Speakers pooled | **50** frozen: 25 train / 10 adaptation / 15 eval, SHA-256 `f57e0d85…` |
| Clips generated | **40** (20 romanised + 20 Devanagari), **0 failures**, RTX 3050 |
| Median speech rate | 13.9 chars/sec Devanagari vs **14.4** romanised |
| Transliteration audit | **0** unmapped characters, **0** rows leaking Devanagari, across 56,143 |
| Tests added | transliteration 35, pilot rating 15, pilot-job reproducibility 12 |

## What the pilot answered, and what it did not

**Answered — script.** Not by the original 4-cell matrix but by P-014: the corpus is
Latin. MUCS is 98.2 % Devanagari-bearing and only **17** usable Devanagari-free
transcripts exist inside the train pool, so Hinglish had to be produced, not
selected. The pilot became a matched A/B to check what that costs.

**Answered — RVC viability (P-017).** Blocked on toolchain, not GPU or ethics.
`rvc-python` → `fairseq` → a numpy build calling `pkgutil.ImpImporter`, removed in
Python 3.12. This machine has only 3.12.7 and no conda. Deferred to W4-T2. **Do not
install it into the project venv** — the resolver downgrades torch and breaks XTTS.

**Not answered — language tag, quality bar.** Both need the rating session.

## Three bugs worth recording

- **The pack was not reproducible across machines.** MUCS durations are whole
  seconds so ties are the rule; with duration as the only sort key the winner fell
  out of filesystem walk order. The CPU and GPU laptops built the *same* pilot with
  speakers 4 and 5 swapped in every cell — `deva_hi_04` was a different person on
  each, and a position-keyed rating sheet would have mis-attributed every score.
  Ties now break on `utt_id` (**P-016**).
- **Romanisation re-opened the truncation trap.** It inflates length ~13 % (up to
  58 %), pushing **14 of 20** transcripts over XTTS's 150-char Hindi limit, which
  XTTS truncates silently. The limit now applies to the romanised text — the P-010
  failure re-entering by the back door (**P-015**).
- **The blinding leaked twice before it worked.** `clip_id` encodes the script in
  its a/b suffix, so `blind_id` is assigned *after* the shuffle; and the pack folder
  in the audio path (`pilot_roman/` vs `pilot_deva/`) gave the answer away as
  plainly as a label, so `--stage-dir` restages both runs under neutral names.

## How to run it

```bash
python -m src.data.ethics_gate                       # exits 0 when signed
python -m src.data.speaker_pools --verify <sha256>   # confirm pools unchanged

python -m src.data.pilot_jobs --data-root $DATA_ROOT \
    --pack-dir $DATA_ROOT/generated/pilot_roman --romanise
python -m src.data.spoof_generation --jobs <pack>/generation_jobs.csv \
    --pack-dir <pack> --out-dir <pack>/outputs
python -m src.data.pilot_rating --roman-pack <roman> --deva-pack <deva> \
    --stage-dir $DATA_ROOT/generated/pilot_ab/clips
```

## Honest limitations

- **One clip of 40 came out empty** — 0.83 s of audio from a 150-character
  transcript, no exception raised. The batch runner catches crashes, not silent
  emptiness. At Week-4 scale that is ~200 dead files; the W4-T6 duration filter has
  to land before the 4,000-clip run.
- **The A/B is unrated.** The pre-screen says romanisation does not rush or truncate
  speech; three people still have to confirm the clips say the right words.
- **Two romanisation choices are judgement calls.** `फ` → `ph` gives `sirph` where a
  speaker would type `sirf`; `ड़` → `r` gives `thoraa` for `thoda`.
- **RVC is untested end to end** — no model trained, no conversion run.

## What's next

1. **Rate the 40 clips** (all three) → language tag + quality bar → Week 4 unblocks.
2. **Duration auto-filter** before generation at scale.
3. **RVC**: side-by-side Python 3.10, a fairseq-free fork, or drop it — a team call,
   since dropping it narrows the scope the mentor signed.
