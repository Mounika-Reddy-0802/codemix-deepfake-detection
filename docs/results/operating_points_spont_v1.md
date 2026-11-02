# Operating points with spontaneous dev speech (decision D1, part a)

**Method written on 1 November 2026 while the `iv_dev` clips were being scored, before any
operating point was chosen on them.**

## Why

SSL-AASIST already separates spontaneous genuine speech from clones of the same speakers
(2–3% EER on the 30 IndicVoices test speakers, `indicvoices_data_v1.md`), yet calls 15–40%
of their genuine clips fake at its fixed rule. That is a misplaced operating point, not a
ranking failure: every model's live-path threshold was chosen on a dev set of read lectures
only. Team decision D1 part (a) (P-038) lets the 20 `iv_dev` speakers into the dev set for
choosing operating points, with no retraining.

## Method (fixed before any `iv_dev` clip was scored)

- **Scores.** The live path (`src.inference.calibrate score`: every 4 s window, level-
  normalised, smoothed) on the `iv_dev` speakers' 694 genuine extempore clips and 398 XTTS-v2
  clones over the G.711 line, for the final adapter (seed 1234) and both SSL-AASIST
  checkpoints; the two-checkpoint average is the window-by-window mean
  (`src.inference.ensemble`). The 598 clones of the `iv_test` speakers are scored the same
  way, so the test speakers have both genuine clips and clones
  (`data/indicvoices/run_operating_points.sh`).
- **Rule, unchanged.** P-031's rule in `src.inference.calibrate choose`: candidate thresholds
  are low quantiles of the genuine dev windows (and the window EER point); the chosen point is
  the most sensitive threshold and ladder with at most 5% of stitched genuine 60 s dev calls
  alerted. Run twice per model: on the old dev windows (506 clips; it must reproduce the
  tracked threshold file exactly, and does for all four models) and on the old dev windows
  plus the `iv_dev` windows (1,598 clips). The new points go to
  `configs/threshold_*_spont.yaml`; the tracked files are untouched
  (`data/indicvoices/choose_operating_points.sh`).
- **Measured once, both points, the same sets** (`src/reporting/operating_points.py`): HiACC
  (24 adult speakers, spontaneous, genuine only), the `iv_test` speakers (genuine and clones),
  CM04 Tortoise, CM07 IndicF5, CM09 Chatterbox, the final adapter's eval pool, new speakers
  and RVC test half, and the 30 real calls over G.711. Two readings each: the live path (a real
  call that raises any alert is a false alarm; per clip and as stitched 60 s calls) and the
  voice check (one decision per file, real when its mean window P(genuine) reaches the
  threshold, as `src.inference.any_audio.judge` decides).
- **Adoption.** The new point is adopted whatever the held-out sets show; the clones of
  unseen tools that it lets through are reported beside the false alarms it removes. The
  voice check reads `configs/threshold_ssl_aasist.yaml`; switching it to the new file is the
  adoption. The recorded live-call demo results stay on the old points until they are rerun.

One measure was added after the first results, and it chooses nothing: the threshold-free
separation of genuine spontaneous files from each tool's clones (last table).

## Results (2 November 2026)

The rule's new points, all satisfying it (at most 5% of genuine 60 s dev calls alerted):

| Model | Old point (read-speech dev) | New point (dev + `iv_dev`) |
|---|---:|---:|
| Final adapter, seed 1234 | 0.687, ladder 3/5 | 0.0016, ladder 2/4 |
| SSL-AASIST seed 1234 (the voice check's model) | 0.580 | 0.082 |
| SSL-AASIST seed 2025 | 0.334 | 0.014 |
| Average of the two | 0.540 | 0.103 |

**SSL-AASIST seed 1234, every held-out set** (`experiments/results/operating_points_spont_v1.json`):

| Held-out audio | Files or calls | Old point | New point |
|---|---:|---:|---:|
| *Genuine spontaneous called cloned (voice check)* | | | |
| HiACC (24 speakers) | 1,000 files | 7.6% | 0.7% |
| IndicVoices `iv_test` (30 speakers) | 1,024 files | 15.3% | 5.0% |
| *Genuine 60 s calls alerted (live path)* | | | |
| HiACC | 41 calls | 26.8% | 0.0% |
| IndicVoices `iv_test` | 94 calls | 84.0% | 6.4% |
| The 18 genuine real calls (one call at a time) | 18 calls | 11.1% | 0.0% |
| *Clones passing as real (voice check)* | | | |
| XTTS-v2 clones of the `iv_test` speakers (seen tool) | 598 files | 0.0% | 1.0% |
| Tortoise, CM04 | 68 files | 4.4% | 32.4% |
| IndicF5, CM07 | 134 files | 70.9% | 95.5% |
| Chatterbox, CM09 | 134 files | 78.4% | 96.3% |
| The 12 cloned real calls (XTTS-v2, Chatterbox) | 12 calls | 50.0% | 58.3% |
| *Cloned 60 s calls alerted (live path)* | | | |
| XTTS-v2, `iv_test` speakers | 55 calls | 100% | 100% |
| Tortoise, CM04 | 6 calls | 100% | 100% |
| IndicF5, CM07 | 6 calls | 100% | 0% |
| Chatterbox, CM09 | 6 calls | 83.3% | 0% |

**Every model, voice check** (genuine called cloned; clones passing as real):

| Model | HiACC genuine | `iv_test` genuine | `iv_test` XTTS-v2 clones | Tortoise | IndicF5 | Chatterbox |
|---|---|---|---|---|---|---|
| Final adapter | 53.8 → 2.3 | 48.8 → 5.7 | 1.5 → 35.8 | 60.3 → 100 | 85.1 → 98.5 | 85.8 → 100 |
| SSL-AASIST seed 1234 | 7.6 → 0.7 | 15.3 → 5.0 | 0.0 → 1.0 | 4.4 → 32.4 | 70.9 → 95.5 | 78.4 → 96.3 |
| SSL-AASIST seed 2025 | 22.8 → 5.1 | 29.5 → 5.4 | 0.0 → 1.2 | 8.8 → 57.4 | 76.9 → 100 | 50.8 → 86.6 |
| Average of the two | 17.1 → 1.5 | 23.1 → 5.5 | 0.0 → 0.0 | 5.9 → 33.8 | 74.6 → 94.8 | 63.4 → 93.3 |

**Threshold-free: can any single threshold separate them?** EER (%) of the voice check's file
score, the 2,024 genuine spontaneous files (HiACC and `iv_test`) against each tool's clones:

| Model | XTTS-v2, same speakers | Tortoise | IndicF5 | Chatterbox |
|---|---:|---:|---:|---:|
| Final adapter | 13.7 | 55.9 | 71.6 | 74.6 |
| SSL-AASIST seed 1234 | 1.5 | 8.8 | 39.6 | 44.0 |
| SSL-AASIST seed 2025 | 2.0 | 22.1 | 55.3 | 38.2 |
| Average of the two | 1.3 | 10.3 | 52.3 | 38.8 |

The genuine files and the Tortoise, IndicF5 and Chatterbox clones come from different corpora
and speakers, so these numbers say what one file-score threshold can and cannot do, not why.
HiACC here is the 1,000-clip subset scored through the live path earlier.

## Reading

1. **Part (a) did what it was built to do.** For every model, genuine spontaneous speech is
   called cloned 3–23 times less often, and genuine spontaneous 60 s calls are rarely warned
   (none of HiACC's 41, 5–10% of the 94 `iv_test` calls). Clones of the seen tool (XTTS-v2), even of the same speakers, are still caught.
2. **It costs most of the detection of unseen generators.** At the new points the voice check
   lets through a third (seed 1234, the average) to over half (seed 2025) of Tortoise clones
   and nearly all IndicF5 and Chatterbox clones, and the live path stops warning IndicF5 and Chatterbox calls.
3. **The cause is ranking, not the operating point.** Against XTTS-v2 clones the file score
   separates genuine spontaneous speech well (1.3–2.0% EER for SSL-AASIST), which is what the
   `iv_test` diagnosis showed. Against the newest generators it does not (38–55%): their clones
   score as genuine as unscripted speech, so any threshold that accepts real callers who speak
   freely also accepts them. The earlier reading that SSL-AASIST's false alarms were "an
   operating-point problem" holds for the seen tool only.
4. **The final adapter cannot serve spontaneous speech at any point.** Its new point (0.0016)
   lets through a third of the seen tool's clones and nearly every unseen-tool clone; against IndicF5 and
   Chatterbox its file score is worse than chance.
5. **Adopted as fixed beforehand.** The voice check now reads
   `configs/threshold_ssl_aasist_spont.yaml` (0.082; `configs/voice_check.yaml` in the
   voice-check work, where its accuracy table shows both points). The recorded live-call demo
   results stay on the old points until they are rerun.
6. **What follows.** Only a change to the model can separate unscripted genuine speech from the
   newest clones; part (b) tests whether spontaneous training speech does
   (`spontaneous_training_v1.md`). For the voice check, reporting "uncertain" between the two
   development-chosen points (0.082 and 0.580) would avoid a confident wrong answer exactly
   where the two overlap; that is a team decision.

## How to reproduce

```bash
# the iv_dev clips (filepath, label, tool) are the iv_ rows of codemix_adapt_dev_spont_channel20.csv
python -m src.inference.calibrate score --kind ssl_aasist \
    --checkpoint <ssl_aasist_final_s1234_fp16.pt> --manifest <iv_dev clips> \
    --data-root <IndicVoices G.711 clips> --out experiments/calibration_ssl/iv_dev_windows.csv
# (the same for the iv_test clones, the seed-2025 checkpoint and the adapter, --kind adapter)
python -m src.inference.ensemble --a experiments/calibration_ssl/iv_dev_windows.csv \
    --b experiments/calibration_ssl_s2025/iv_dev_windows.csv \
    --out experiments/calibration_ssl_avg/iv_dev_windows.csv
# dev_spont_windows.csv = dev_windows.csv + iv_dev_windows.csv (rows appended);
# iv_test_all_windows.csv = iv_test_windows.csv + iv_test_clones_windows.csv
python -m src.inference.calibrate choose --dev experiments/calibration_ssl/dev_spont_windows.csv \
    --validate experiments/calibration_ssl/{cm04,cm07,cm09,hiacc,iv_test_all}_windows.csv \
    --checkpoint-name ssl_aasist_final_s1234 --out configs/threshold_ssl_aasist_spont.yaml \
    --report experiments/results/threshold_calibration_ssl_spont.json
python -m src.reporting.operating_points   # -> experiments/results/operating_points_spont_v1.json
```
