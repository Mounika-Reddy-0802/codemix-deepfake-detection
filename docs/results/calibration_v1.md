# DET curves and calibration of the final systems (v1)

Plan task A7. Every result document so far quotes the equal error rate (EER). The EER
describes one point of the error trade-off, the threshold where both errors are equal,
and that point is chosen after seeing the test labels. A deployed detector runs at a
fixed rule instead. This document adds, from the same per-clip scores the published
tables come from:

- **DET curves**: every trade-off between false alarms (genuine speech called fake)
  and misses (fakes called genuine);
- **calibration**: whether P(genuine) can be read as a probability, by the Brier score
  (mean squared error of P(genuine); 0 is perfect) and the expected calibration error
  (ECE, 10 equal-width bins);
- **accuracy, false alarms and misses at the fixed clip rule** every result document
  uses: fake when P(genuine) < 0.5. Nothing is chosen here.

SSL-AASIST emits log-odds, which are mapped to P(genuine) by the logistic function (the
rule is then log-odds below 0); the EER does not depend on that mapping. Recomputed EERs
equal the committed summaries to the last digit (checked on four files across all three
systems before this table was built).

Code: `src/reporting/det_curves.py` (tests in `tests/test_det_curves.py`). Run:
`python -m src.reporting.det_curves --scores-root D:/dfdata/gen_results` (the unzipped
Kaggle outputs of runs 4, 7 and 8; the English scores are committed in
`experiments/results/english_channel_v1/`). Output: `experiments/results/calibration_v1.json`
(146 score files, every seed) and `experiments/figures/det_final_systems.{png,pdf}`.

## The table (mean ± sd over training seeds)

| Set | System | Seeds | EER (%) | Brier | ECE | Accuracy (%) | False alarm (%) | Miss (%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Original test pool (XTTS-v2), G.711 | Deployed adapter | 5 | 3.50 ± 0.75 | 0.045 ± 0.016 | 0.046 ± 0.020 | 94.8 ± 1.8 | 0.8 ± 0.6 | 8.1 ± 3.0 |
| Original test pool (XTTS-v2), G.711 | Final adapter | 5 | 1.13 ± 0.18 | 0.010 ± 0.003 | 0.012 ± 0.009 | 98.6 ± 0.4 | 1.0 ± 0.9 | 1.6 ± 0.9 |
| Original test pool (XTTS-v2), G.711 | SSL-AASIST | 5 | 0.25 ± 0.12 | 0.002 ± 0.002 | 0.004 ± 0.004 | 99.7 ± 0.2 | 0.6 ± 0.6 | 0.1 ± 0.1 |
| New speakers (XTTS-v2, Tortoise), G.711 | Deployed adapter | 5 | 14.42 ± 1.99 | 0.173 ± 0.016 | 0.173 ± 0.019 | 81.3 ± 1.4 | 4.3 ± 1.6 | 36.1 ± 4.9 |
| New speakers (XTTS-v2, Tortoise), G.711 | Final adapter | 5 | 2.59 ± 0.33 | 0.035 ± 0.013 | 0.040 ± 0.018 | 95.9 ± 1.5 | 0.6 ± 0.8 | 8.4 ± 3.8 |
| New speakers (XTTS-v2, Tortoise), G.711 | SSL-AASIST | 5 | 0.31 ± 0.02 | 0.003 ± 0.000 | 0.004 ± 0.002 | 99.7 ± 0.0 | 0.2 ± 0.2 | 0.5 ± 0.3 |
| New speakers (XTTS-v2, Tortoise), AMR-NB | Deployed adapter | 5 | 24.89 ± 0.96 | 0.233 ± 0.009 | 0.223 ± 0.009 | 74.6 ± 1.0 | 30.5 ± 2.8 | 19.3 ± 2.1 |
| New speakers (XTTS-v2, Tortoise), AMR-NB | Final adapter | 5 | 1.94 ± 0.26 | 0.025 ± 0.012 | 0.028 ± 0.016 | 97.1 ± 1.5 | 0.5 ± 0.5 | 5.8 ± 3.6 |
| New speakers (XTTS-v2, Tortoise), AMR-NB | SSL-AASIST | 5 | 0.22 ± 0.08 | 0.003 ± 0.001 | 0.005 ± 0.001 | 99.6 ± 0.1 | 0.2 ± 0.2 | 0.6 ± 0.5 |
| CM04 Tortoise, G.711 | Deployed adapter | 5 | 36.18 ± 5.48 | 0.196 ± 0.005 | 0.193 ± 0.010 | 79.4 ± 0.3 | 0.8 ± 0.6 | 88.5 ± 2.7 |
| CM04 Tortoise, G.711 | Final adapter | 5 | 17.73 ± 2.27 | 0.134 ± 0.025 | 0.135 ± 0.031 | 85.0 ± 2.5 | 1.0 ± 0.9 | 62.9 ± 12.8 |
| CM04 Tortoise, G.711 | SSL-AASIST | 5 | 1.75 ± 0.71 | 0.025 ± 0.014 | 0.032 ± 0.015 | 96.7 ± 2.1 | 0.6 ± 0.6 | 12.5 ± 10.4 |
| CM04 Tortoise, AMR-NB | Deployed adapter | 5 | 37.93 ± 3.83 | 0.531 ± 0.038 | 0.535 ± 0.037 | 44.4 ± 3.5 | 66.0 ± 5.3 | 20.1 ± 5.7 |
| CM04 Tortoise, AMR-NB | Final adapter | 5 | 6.90 ± 2.06 | 0.081 ± 0.041 | 0.085 ± 0.047 | 90.7 ± 4.6 | 0.6 ± 0.3 | 39.2 ± 20.9 |
| CM04 Tortoise, AMR-NB | SSL-AASIST | 5 | 1.36 ± 0.31 | 0.017 ± 0.008 | 0.021 ± 0.011 | 97.8 ± 1.0 | 0.6 ± 0.6 | 7.8 ± 5.9 |
| IndicF5, G.711 | Deployed adapter | 5 | 34.90 ± 1.43 | 0.378 ± 0.028 | 0.382 ± 0.028 | 60.7 ± 2.2 | 2.0 ± 0.9 | 85.6 ± 6.0 |
| IndicF5, G.711 | Final adapter | 5 | 21.68 ± 3.74 | 0.315 ± 0.065 | 0.325 ± 0.066 | 66.7 ± 5.8 | 0.7 ± 0.7 | 73.7 ± 13.9 |
| IndicF5, G.711 | SSL-AASIST | 5 | 15.79 ± 2.61 | 0.250 ± 0.039 | 0.276 ± 0.039 | 72.2 ± 4.0 | 0.4 ± 0.3 | 61.7 ± 8.7 |
| IndicF5, AMR-NB | Deployed adapter | 5 | 46.47 ± 1.17 | 0.430 ± 0.014 | 0.419 ± 0.017 | 53.9 ± 1.1 | 44.8 ± 3.0 | 47.7 ± 2.7 |
| IndicF5, AMR-NB | Final adapter | 5 | 21.32 ± 2.27 | 0.333 ± 0.052 | 0.342 ± 0.051 | 65.0 ± 5.0 | 0.7 ± 0.2 | 77.6 ± 11.4 |
| IndicF5, AMR-NB | SSL-AASIST | 5 | 21.75 ± 5.58 | 0.328 ± 0.077 | 0.345 ± 0.073 | 64.5 ± 7.0 | 0.5 ± 0.3 | 78.9 ± 15.8 |
| Chatterbox, G.711 | SSL-AASIST | 5 | 12.86 ± 3.83 | 0.242 ± 0.056 | 0.272 ± 0.053 | 72.2 ± 5.1 | 0.4 ± 0.3 | 61.7 ± 11.5 |
| RVC test half, G.711 | Deployed adapter | 5 | 6.11 ± 0.71 | 0.076 ± 0.024 | 0.083 ± 0.031 | 91.3 ± 3.1 | 1.3 ± 0.5 | 15.8 ± 6.5 |
| RVC test half, G.711 | Final adapter | 5 | 8.53 ± 1.91 | 0.129 ± 0.053 | 0.143 ± 0.064 | 85.0 ± 6.3 | 1.1 ± 0.8 | 28.2 ± 12.8 |
| RVC test half, G.711 | SSL-AASIST | 5 | 5.17 ± 1.51 | 0.078 ± 0.021 | 0.098 ± 0.026 | 90.9 ± 2.4 | 0.4 ± 0.4 | 17.5 ± 4.7 |
| Language control (genuine only), G.711 | Deployed adapter | 5 | - | 0.022 ± 0.011 | 0.028 ± 0.014 | 97.6 ± 1.2 | 2.4 ± 1.2 | - |
| Language control (genuine only), G.711 | Final adapter | 5 | - | 0.004 ± 0.006 | 0.007 ± 0.009 | 99.5 ± 0.7 | 0.5 ± 0.7 | - |
| Language control (genuine only), G.711 | SSL-AASIST | 5 | - | 0.002 ± 0.002 | 0.006 ± 0.007 | 99.7 ± 0.3 | 0.3 ± 0.3 | - |
| English (ASVspoof sample), clean | S1 (English only) | 1 | 0.51 | 0.033 | 0.039 | 96.1 | 0.2 | 5.8 |
| English (ASVspoof sample), clean | Deployed adapter (published) | 1 | 2.30 | 0.137 | 0.151 | 84.9 | 0.1 | 22.7 |
| English (ASVspoof sample), clean | Final adapter | 1 | 13.44 | 0.454 | 0.484 | 50.8 | 0.2 | 73.7 |
| English (ASVspoof sample), G.711 | S1 (English only) | 1 | 6.50 | 0.110 | 0.136 | 83.9 | 0.7 | 23.9 |
| English (ASVspoof sample), G.711 | Deployed adapter (published) | 1 | 24.20 | 0.264 | 0.260 | 70.8 | 7.4 | 40.0 |
| English (ASVspoof sample), G.711 | Final adapter | 1 | 25.95 | 0.483 | 0.511 | 48.3 | 0.9 | 77.0 |

The deployed and final adapters and SSL-AASIST are the five-seed runs of
`w13_runs_v1.md`; the English rows are one training each (the published deployed
checkpoint, the final recipe's seed 1234, and S1), as in `english_final_v1.md`.

## Reading

**1. At the fixed rule, the final systems almost never accuse a genuine speaker; their
errors are misses.** False alarms stay at or below 1.1% on every Hindi-English set for
the final adapter (0.6% at most for SSL-AASIST). The misses on the unseen tools are large: IndicF5 over
G.711 is missed 73.7% of the time by the final adapter and 61.7% by SSL-AASIST, and
Chatterbox 61.7% by SSL-AASIST, although SSL-AASIST's EERs there are 15.79% and
12.86%. The EER therefore overstates what the deployed rule delivers on a tool the model
has not seen: the rule sits at a low-false-alarm point of the curve, and fakes from an
unseen tool score higher P(genuine) than the dev fakes the operating point was set on.
The same pattern shows on CM04 Tortoise (final adapter EER 17.73%, miss 62.9% at the
rule) and in the voice check (33% of IndicF5 and 20% of Chatterbox clips caught,
`docs/voice_check.md`).

**2. The deployed adapter fails the other way on the real codec.** Over AMR-NB it calls
30.5% of the new speakers' genuine clips fake (66.0% on CM04's genuine speakers), the
clip-level counterpart of the real-call result that the mobile codec is its largest
step (`real_calls_v1.md`). The final recipe, trained with AMR-NB audio, brings this to
0.5-0.6%.

**3. Scores are confident, so calibration error follows the errors.** On the seen tool
the ECE is 0.004 for SSL-AASIST and 0.012 for the final adapter. On the unseen
tools it is 0.27-0.35 for both final systems: wrong answers come with P(genuine) near 1,
not near 0.5. A score near 1 does not mean "certainly genuine" on audio from a new
generator, so the voice check and the live warning should present a verdict and its
measured weak rows rather than a confidence percentage.

**4. English.** S1 keeps the best English curve (DET panel c). Both adapters lie well
above it over G.711, and the final adapter's English errors are almost all misses
(77.0% at the rule, 0.9% false alarms), the reverse of the deployed adapter's (40.0%
misses, 7.4% false alarms). This is the motivation for routing English calls to S1 (plan
task A3).

**What this does not license.** Lowering the rule on these test sets would raise the
catch rate on IndicF5 and Chatterbox, but that is tuning on the test set, which this
project does not do. An operating point that holds for unseen tools has to be chosen on
development data that contains a held-out tool (for example, leave-one-tool-out fakes in
the dev set); that is future work, and any such dev set must keep IndicF5, Chatterbox
and Tortoise unseen.

## Limits

- One fixed rule (P(genuine) < 0.5, clip level, one 4 s crop per clip). The live system
  and the voice check use windowed scores with their own dev-chosen thresholds; their
  behaviour is measured in `live_final_recipe_v1.md`, `real_calls_v1.md` and
  `docs/voice_check.md`.
- English rows are one training each; the adapter Chatterbox score exists for one seed
  only and is not in the table.
- The DET figure shows seed 1234 of each system on the Kaggle sets; every seed is in
  the JSON.
