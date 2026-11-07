# Voice check demo for the 13 November review (about 7 minutes)

The project's final output is the voice check: any recording in, "real" or "cloned" out
(`docs/voice_check.md`). This script shows it on recordings whose verdict is already known.
Every clip below was judged by the page's own scoring path (`src.inference.any_audio.judge`,
telephone front end, frozen threshold 0.082) in the clean-file measurement, and the expected
verdict is that run's (`experiments/any_audio/clean_scored.csv`). Nothing here is cherry-picked
to hide a weakness: scenes 6 and 7 are the clones it misses, shown on purpose.

## Before the evaluators arrive (15 minutes)

1. Close every browser and app you do not need: the detector needs about 2.5 GB of free memory.
2. Copy the demo folder to the demo laptop: `D:\voice_check_demo\` (7 clips, 7 backups and
   `expected.csv`). The clips are evaluation recordings; they are shown, never shared.
3. **Re-check every clip on this laptop** (one command in Git Bash, about 3 minutes; it reads the
   page's own settings from `configs/voice_check.yaml`):
   ```bash
   python -m src.inference.any_audio D:/voice_check_demo/*.wav --out D:/voice_check_demo/today.csv
   ```
   Each verdict must match `expected.csv`. If one does not, use its backup from `backup/`.
4. Start the page: `python -m voice_check.app`, open `http://127.0.0.1:7860`, and check one clip so
   the model is loaded before the evaluators watch.

## The scenes

| # | Clip (in `D:\voice_check_demo\`) | Expected | P(genuine) | What to say |
|---|---|---|---|---|
| 1 | `01_genuine_read__…_0027.wav` | real | 1.00 | A real Hindi-English speaker reading. Most detectors trained on English call this fake; ours does not. |
| 2 | `02_genuine_spontaneous__AD09004.wav` | real | 1.00 | A real person speaking freely, the hardest genuine case. The final recipe calls 43.9% of such clips fake; the voice check, whose operating point was chosen with spontaneous development speech, judges 99.3% of them real. |
| 3 | `03_xtts_clone__…_00775.wav` | cloned | 0.001 | An XTTS-v2 clone: the tool the model was trained on. 96.7% caught on clean files. |
| 4 | `04_tortoise_clone__…_00310.wav` | cloned | 0.004 | Tortoise: a tool the model never heard. 66.7% caught on clean files. |
| 5 | `05_rvc_conversion__…_01486.wav` | cloned | 0.002 | Voice conversion, not text-to-speech: a different kind of attack. |
| 6 | `06_indicf5_clone_misses__…_00575.wav` | **real** | 0.80 | The honest limit: IndicF5, a recent Hindi TTS, passes. On clean files it catches none of these. |
| 7 | `07_chatterbox_clone_misses__…_00066.wav` | **real** | 1.00 | Chatterbox passes too. This is the gap the generator study and voice check v2 address. |

For each clip: drag it onto the page, press **Check this voice**, point at the verdict and at the
plot of every 4-second window's score against the threshold line.

## After scene 7 (1 minute): what we did about it

- The generator study asks whether training on one more tool catches a tool never heard
  (`docs/results/generator_study_v1.md`; rule committed before training). Its answer is
  **partly**: trained also on Chatterbox, the final recipe's error on IndicF5, never heard, falls
  from 20.44% to 12.43% in every seed (confirmed); trained also on IndicF5, its error on
  Chatterbox falls from 24.32% to 15.75% in every seed, but it calls slightly more genuine speech
  fake than the rule allows, so that half is not confirmed. Show
  `experiments/figures/generator_study.png`.
- Voice check v2 is the same model trained also on IndicF5 and Chatterbox clones of other
  speakers (`docs/results/voice_check_v2.md`). Scenes 6 and 7 use speakers it never heard. If the
  rule adopted it, re-run scenes 6 and 7 on v2 and say what changed; if not, say which condition
  it missed. Do not switch models in front of the evaluators unless the rule adopted v2.

## Live microphone (only if the field recordings are measured)

Speaking into the laptop is the most convincing scene and the least measured one. Do it only
after the field recordings (`docs/field_validation/PROTOCOL.md`) are scored and their accuracy
is in `docs/voice_check.md`; then quote that number before speaking. A replayed real voice
counts as real: the check answers "cloned or not", not "live or recorded".

## If something fails

The recorded demo video (made on the freeze day from this script) and the verdict table in
`docs/voice_check.md` show the same path. Questions and answers: `VIVA_QA.md` (Q52-Q55 cover
this demo).
