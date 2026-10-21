# The week-13 Kaggle runs, seed for seed (v1)

EER (%) mean ± sd over training seeds (false alarms: % of genuine clips called fake).
Difference: first minus second over the seeds both have; negative favours the first.
p paired: exact sign-flip test over paired seeds (minimum 0.0625 with five seeds);
p MWU: exact Mann-Whitney test, unpaired (minimum 0.008 with five each). Built by
`src/reporting/w13_runs.py`.

## Run 1: extended recipe (248 speakers) vs deployed (21)

| Set | extended | deployed | Difference | extended better | p paired | p MWU |
|---|---|---|---|---|---|---|
| Original test pool (XTTS) | 1.12 ± 0.31 (n=5) | 3.42 ± 0.44 (n=5) | -2.30 | 5/5 | 0.062 | 0.008 |
| CM04 Tortoise | 24.48 ± 3.48 (n=5) | 33.46 ± 2.63 (n=5) | -8.98 | 5/5 | 0.062 | 0.008 |
| RVC test half | 6.49 ± 1.88 (n=5) | 5.78 ± 0.67 (n=5) | +0.71 | 2/5 | 0.500 | 0.460 |
| New speakers, XTTS | 0.91 ± 0.33 (n=5) | 10.75 ± 0.94 (n=5) | -9.84 | 5/5 | 0.062 | 0.008 |
| New speakers, Tortoise | 13.59 ± 3.39 (n=5) | 29.82 ± 1.73 (n=5) | -16.23 | 5/5 | 0.062 | 0.008 |
| IndicF5 | 27.25 ± 0.68 (n=5) | 35.40 ± 1.64 (n=5) | -8.14 | 5/5 | 0.062 | 0.008 |
| False alarms, language control | 0.47 ± 0.38 (n=5) | 3.40 ± 1.60 (n=5) | -2.93 | 5/5 | 0.062 | 0.008 |
