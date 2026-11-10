# Voice check v2 averaged with the spontaneous recipe (v1)

**Status: registered, not measured.** Written on 10 November 2026 before any averaged number
existed, so the rule cannot be fitted to its results. Team decision P-046. The rule is coded in
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

Not measured yet.
