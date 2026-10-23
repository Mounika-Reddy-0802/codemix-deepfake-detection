# The week-13 Kaggle runs, seed for seed (v1)

EER (%) mean ± sd over training seeds (false alarms: % of genuine clips called fake).
Difference: first minus second over the seeds both have; negative favours the first.
p paired: exact sign-flip test over paired seeds (minimum 0.0625 with five seeds);
p MWU: exact Mann-Whitney test, unpaired (minimum 0.008 with five each). Built by
`src/reporting/w13_runs.py`.

## Runs 7 and 8: SSL-AASIST vs the adapter, both on the final-recipe data

| Set | SSL-AASIST | adapter | Difference | SSL-AASIST better | p paired | p MWU |
|---|---|---|---|---|---|---|
| Original test pool (XTTS) | 0.25 ± 0.12 (n=5) | 1.13 ± 0.18 (n=5) | -0.88 | 5/5 | 0.062 | 0.008 |
| CM04 Tortoise | 1.75 ± 0.71 (n=5) | 17.73 ± 2.27 (n=5) | -15.97 | 5/5 | 0.062 | 0.008 |
| RVC test half | 5.17 ± 1.51 (n=5) | 8.53 ± 1.91 (n=5) | -3.36 | 5/5 | 0.062 | 0.024 |
| New speakers, XTTS | 0.02 ± 0.03 (n=5) | 0.94 ± 0.34 (n=5) | -0.92 | 5/5 | 0.062 | 0.008 |
| New speakers, Tortoise | 0.73 ± 0.38 (n=5) | 8.81 ± 1.43 (n=5) | -8.09 | 5/5 | 0.062 | 0.008 |
| IndicF5 | 15.79 ± 2.61 (n=5) | 21.68 ± 3.74 (n=5) | -5.90 | 4/5 | 0.125 | 0.056 |
| False alarms, language control | 0.29 ± 0.33 (n=5) | 0.51 ± 0.67 (n=5) | -0.22 | 4/5 | 0.312 | 0.802 |
| New speakers, XTTS, AMR-NB | 0.05 ± 0.07 (n=5) | 0.69 ± 0.24 (n=5) | -0.64 | 5/5 | 0.062 | 0.008 |
| New speakers, Tortoise, AMR-NB | 0.51 ± 0.21 (n=5) | 6.04 ± 1.17 (n=5) | -5.53 | 5/5 | 0.062 | 0.008 |
| IndicF5, AMR-NB | 21.75 ± 5.58 (n=5) | 21.32 ± 2.27 (n=5) | +0.43 | 2/5 | 0.938 | 0.690 |
| CM04, AMR-NB | 1.36 ± 0.31 (n=5) | 6.90 ± 2.06 (n=5) | -5.54 | 5/5 | 0.062 | 0.008 |

## Chatterbox (CM09), SSL-AASIST only

12.86 ± 3.83 (n=5); by seed 7: 13.52, 42: 18.28, 99: 10.46, 1234: 8.17, 2025: 13.86. The adapter was scored on
seed 1234 only (31.98), so there is no paired test.
