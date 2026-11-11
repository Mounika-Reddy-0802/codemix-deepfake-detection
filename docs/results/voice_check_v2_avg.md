# Voice check v2 averaged with the spontaneous recipe (v1)

**Status: measured on 10 November 2026: not recommended.** Registered the same day before any
averaged number existed, so the rule cannot be fitted to its results. Team decision P-046. The rule is coded in
`src/reporting/rvc_average.py` and committed with this page.

## Why

Voice check v2 traded RVC for the two newest tools. Against the spontaneous recipe's SSL-AASIST
alone on the same clips, RVC conversions caught fell from 85.8% and 89.0% to 73.5% and 77.6%
(seeds 1234 and 2025), while IndicF5 and Chatterbox clones caught rose to 89-94%
(`voice_check_v2.md`). The two models are good at different things, so their average may keep
both.

## What is measured

No training and no new scoring. For each seed, voice check v2 and the spontaneous recipe's
SSL-AASIST of the same seed are averaged exactly as the project's earlier two-checkpoint average
(`src/inference/ensemble.py`, `ensemble_v1.md`): for every live-path window and every clip, the
mean of the two models' P(genuine). Both models already scored every input (dev windows, test
clips, the 30 real calls). The average's operating point is chosen on the same dev windows (the
506 read-speech dev clips and the 1,092 IndicVoices dev clips) by the unchanged rule (P-031),
never on a test set or a call.

## Rule (fixed now)

For each seed pair, measured exactly as voice check v2 was (the 31 `gen_train` speakers left
out of the sets that hold them, simulated G.711 line, one head crop per clip, each model at its
own dev-chosen point), the average must meet all five:

1. genuine speech kept to the v2 rule's bars: HiACC 97.3%, read speech (new speakers and
   language control) 99.0%, 11 of 12 genuine live calls, 5 of 6 genuine replays;
2. XTTS-v2 clones caught at least 98.0%;
3. RVC conversions caught at least 5.0 points above v2 of the same seed (about half the 11-12
   points v2 lost);
4. IndicF5 and Chatterbox clones caught, each, at most 5.0 points below v2 of the same seed;
5. Tortoise (new speakers and CM04), each, at most 5.0 points below v2 of the same seed.

Answer: **recommended** if both pairs meet all five, **not recommended** otherwise,
**incomplete** if a quantity is missing. Replayed clones, AMR-NB and clean files are reported,
not judged.

**What the answer does.** The average runs two models, so it doubles the voice check's scoring
time and memory. Whatever the answer, the page keeps v2 seed 1234 for the 13 November review; a
recommended average is put to the team after it, with its scoring time on the demo laptop.

## Run

```
python -m src.reporting.rvc_average build
python -m src.reporting.rvc_average judge --out experiments/results/rvc_average.json
```

`build` writes the averaged clip scores (`experiments/results/voice_check_v2_avg/`), the
averaged dev windows and the operating point (`configs/threshold_ssl_aasist_vcv2avg_s*.yaml`),
and the averaged call windows and verdicts (`experiments/results/real_calls_v1/ssl_vcv2avg_s*`).

## Results

**Answer: not recommended.** `python -m src.reporting.rvc_average judge`
(`experiments/results/rvc_average.json`): both seed pairs meet conditions 1, 2, 3 and 5 and miss
condition 4. The voice check keeps v2 seed 1234.

Operating points chosen on dev only: 0.496 (seed 1234) and 0.529 (seed 2025).

| Measure (percent judged correctly) | v2 s1234 | average s1234 | v2 s2025 | average s2025 |
|---|---|---|---|---|
| 1. Spontaneous genuine (HiACC) | 99.8 | 99.6 | 99.7 | 99.6 |
| 1. Read genuine, new speakers / language control | 100.0 / 100.0 | 99.9 / 100.0 | 99.9 / 100.0 | 99.1 / 99.1 |
| 1. Genuine live calls / genuine replays | 12/12, 6/6 | 12/12, 6/6 | 12/12, 6/6 | 12/12, 6/6 |
| 2. XTTS-v2 clones caught | 100.0 | 100.0 | 100.0 | 100.0 |
| 3. RVC conversions caught (at least +5) | 73.5 | **90.2** (+16.7) | 77.6 | **87.4** (+9.8) |
| 4. IndicF5 clones caught (at most -5) | 89.3 | 70.5 (-18.8) | 90.2 | 77.5 (-12.7) |
| 4. Chatterbox clones caught (at most -5) | 93.8 | 86.5 (-7.3) | 93.5 | 86.5 (-7.0) |
| 5. Tortoise, new speakers | 94.1 | 97.8 | 98.5 | 97.0 |
| 5. Tortoise, CM04 | 91.9 | 97.7 | 98.2 | 98.2 |

Reported, not judged: replayed XTTS-v2 clones 6 of 6 for both averages (v2: 4 and 6);
replayed Chatterbox clones 1 and 0 of 6 (v2: 0 and 0); IndicF5 through the real AMR-NB codec
69.3% and 42.6% (v2: 68.0% and 54.9%); genuine read speech over AMR-NB 99.7-99.8%.

### What it means

- **Averaging moves the balance; it does not give both.** RVC rises past even the spontaneous
  recipe alone (85.8% and 89.0%), and Tortoise rises in seed 1234, but IndicF5 and Chatterbox
  fall back part of the way v2 had raised them: 70-78% and 86.5%, still far above the 12.3% and
  4.2% of the detector before v2.
- **Genuine speech is unaffected**: every genuine live call and replay is still judged real.
- **Which model is better depends on the attack that matters more.** For the newest TTS tools
  v2 alone is better; for voice conversion the average is. The rule asked for both and the
  average does not deliver both, so the page keeps v2.

### Limits

- RVC rests on 6 speakers and CM04 Tortoise on 7 (`voice_check_v2_intervals.md`).
- Two seed pairs; the average was not tried with other weights, because any weight chosen on
  these test sets would be tuned on them.
