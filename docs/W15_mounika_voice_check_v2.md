# Week 15 — Mounika — the voice check, and voice check v2

The project's final output is the voice check: any recording in, "real" or "cloned" out. This
week I built the page around the scoring path, wrote the field-recording protocol it collects
under, designed the robustness sweep, and measured two candidate detectors in it, each under a
rule committed before its numbers existed. The first (SSL-AASIST retrained with spontaneous
speech) was not adopted; the second, voice check v2, was, and the page now runs it.

## What I did

**Field recordings, with consent first** (`b85a411`). `docs/field_validation/PROTOCOL.md`, its
take plan and `docs/ethics/field_recording_consent_form.md` say what is recorded and under what
consent. The page saves a labelled recording only when the speaker's consent box is ticked, into
`data/field_validation/`, which git ignores.

**Robustness sweep design** (`b1f3921`). The transforms (`src/data/robustness.py`: MP3, Opus,
AAC, a second G.711 hop, white noise, babble, room reverberation, speed and pitch), the sweep plan
(`data/manifests/robustness_plan.csv`) and its Kaggle notebook; the results are
`docs/results/robustness_v1.md`.

**The voice check page** (`0a95cb6`). `voice_check/app.py` calls the same `judge` in
`src/inference/any_audio.py` that the accuracy numbers come from, so the page cannot disagree
with them. The detector, its operating point and its front end are read from
`configs/voice_check.yaml` only.

**The retrained detector, measured and not adopted** (`b15dd9a`, P-040). Both seeds of
SSL-AASIST retrained with spontaneous speech, each at its own dev-chosen point, against the rule
written first: seed 2025 judged 97.6% of read genuine speech real against 99% required, so the
page kept its detector.

**Voice check v2, measured and adopted** (P-042, P-043). The same recipe plus IndicF5 and
Chatterbox clones of the generator study's 31 training speakers, two seeds on Kaggle. On the
laptop: the dev-only operating points (0.073 and 0.591, the unchanged rule), the 30 real calls,
then the committed rule. Every number leaves out the 31 training speakers, the reference's too.

| Without the 31 training speakers | Before (0.082) | v2 seed 1234 | v2 seed 2025 |
|---|---|---|---|
| Genuine spontaneous / read speech | 99.3% / 99.9% | 99.8% / 100.0% | 99.7% / 99.9% |
| Genuine live calls / replays | 12/12, 6/6 | 12/12, 6/6 | 12/12, 6/6 |
| IndicF5 / Chatterbox clones caught | 12.3% / 4.2% | 89.3% / 93.8% | 90.2% / 93.5% |
| Tortoise (new speakers / CM04) | 80.0% / 83.7% | 94.1% / 91.9% | 98.5% / 98.2% |
| RVC | 45.4% | 73.5% | 77.6% |
| XTTS-v2 | 99.7% | 100.0% | 100.0% |

Both seeds met all five conditions, so `configs/voice_check.yaml` now names seed 1234 at 0.073.

## What I found that the rule does not judge

- **RVC paid for it.** `src/reporting/voice_check_v2_table.py` puts the spontaneous recipe alone
  beside v2 on the same clips: the IndicF5 and Chatterbox rows took RVC from 85.8/89.0% to
  73.5/77.6% in both seeds. Tortoise barely moved; most of its gain over the old detector was
  already in the recipe.
- **RVC is not an unseen tool here.** The plan page called it one; every recipe, the old
  detector's included, trains on CM02's speaker-held-out RVC half (714 rows, P-025). The
  condition was applied as written and met, so only the wording is corrected. Tortoise is the
  one tool no model heard.
- **A Chatterbox clone played into a phone still passes** (0 of 6, every model). v2 learned
  Chatterbox through the simulated line, not through a speaker and a microphone.
- **The per-call files.** The rule reads one row per call; `real_calls score` writes windows
  only, and the first run of the rule came out "incomplete" until the per-call files were
  written from those windows (`src.inference.real_calls.report`). The rule's code was not
  touched.

## How to reproduce

```bash
python -m src.reporting.voice_check_v2 --out experiments/results/voice_check_v2.json
python -m src.reporting.voice_check_v2_table --out experiments/results/voice_check_v2_table.md
```

Both read committed files only: the head-crop scores (`experiments/results/voice_check_v2/`),
the operating points (`configs/threshold_ssl_aasist_vcv2_s*.yaml`, from
`experiments/calibration_ssl_vcv2_s*/`) and the real-call files
(`experiments/results/real_calls_v1/ssl_vcv2_s*/`).

## Still open

Clean wideband files with v2 are measured in Lahari's week record (`voice_check_v2_clean.md`); field recordings from people using the page; a replayed
Chatterbox clone.
