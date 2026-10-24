# The week-13 Kaggle runs, seed for seed (v1)

EER (%) mean ± sd over training seeds (false alarms: % of genuine clips called fake).
Difference: first minus second over the seeds both have; negative favours the first.
p paired: exact sign-flip test over paired seeds (minimum 0.0625 with five seeds);
p MWU: exact Mann-Whitney test, unpaired (minimum 0.008 with five each). Built by
`src/reporting/w13_runs.py`.

## AASIST zero-shot (published ASVspoof 2019 LA weights)

| Set | EER (%) |
|---|---|
| Original test pool (XTTS) | 42.91 |
| CM04 Tortoise | 52.41 |
| RVC test half | 46.69 |
| New speakers, XTTS | 46.28 |
| New speakers, Tortoise | 54.05 |
| IndicF5 | 54.72 |
| False alarms, language control | 82.33 |

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

## Run 2: AASIST vs wav2vec2 S2, same training data and seeds

| Set | AASIST | S2 | Difference | AASIST better | p paired | p MWU |
|---|---|---|---|---|---|---|
| Original test pool (XTTS) | 4.02 ± 1.95 (n=4) | 3.42 ± 0.44 (n=5) | +0.76 | 2/4 | 0.500 | 1.000 |
| CM04 Tortoise | 29.53 ± 6.73 (n=4) | 33.46 ± 2.63 (n=5) | -4.06 | 3/4 | 0.500 | 0.286 |
| RVC test half | 6.70 ± 0.67 (n=4) | 5.78 ± 0.67 (n=5) | +0.65 | 0/4 | 0.250 | 0.087 |
| New speakers, XTTS | 14.09 ± 1.14 (n=4) | 10.75 ± 0.94 (n=5) | +3.05 | 0/4 | 0.125 | 0.016 |
| New speakers, Tortoise | 26.91 ± 1.12 (n=4) | 29.82 ± 1.73 (n=5) | -2.71 | 4/4 | 0.125 | 0.032 |
| IndicF5 | 27.14 ± 1.18 (n=4) | 35.40 ± 1.64 (n=5) | -7.80 | 4/4 | 0.125 | 0.016 |
| False alarms, language control | 19.03 ± 18.93 (n=4) | 3.40 ± 1.60 (n=5) | +16.30 | 0/4 | 0.125 | 0.016 |

## Run 2: AASIST vs wav2vec2 S2, both on the extended set

| Set | AASIST | S2 ext | Difference | AASIST better | p paired | p MWU |
|---|---|---|---|---|---|---|
| Original test pool (XTTS) | 1.96 ± 0.85 (n=3) | 1.12 ± 0.31 (n=5) | +0.77 | 0/3 | 0.500 | 0.107 |
| CM04 Tortoise | 16.72 ± 0.52 (n=3) | 24.48 ± 3.48 (n=5) | -9.67 | 3/3 | 0.250 | 0.036 |
| RVC test half | 13.46 ± 0.37 (n=3) | 6.49 ± 1.88 (n=5) | +7.49 | 0/3 | 0.250 | 0.036 |
| New speakers, XTTS | 1.81 ± 0.27 (n=3) | 0.91 ± 0.33 (n=5) | +0.85 | 0/3 | 0.250 | 0.036 |
| New speakers, Tortoise | 9.58 ± 1.55 (n=3) | 13.59 ± 3.39 (n=5) | -5.28 | 3/3 | 0.250 | 0.143 |
| IndicF5 | 26.40 ± 1.57 (n=3) | 27.25 ± 0.68 (n=5) | -1.24 | 2/3 | 0.500 | 0.446 |
| False alarms, language control | 2.25 ± 0.81 (n=3) | 0.47 ± 0.38 (n=5) | +1.96 | 0/3 | 0.250 | 0.036 |

## Run 3: G.711 + AMR-NB training vs deployed (G.711)

| Set | two-codec | deployed | Difference | two-codec better | p paired | p MWU |
|---|---|---|---|---|---|---|
| Original test pool (XTTS) | 3.14 ± 1.18 (n=3) | 3.09 ± 0.61 (n=3) | +0.05 | 1/3 | 0.750 | 1.000 |
| CM04 Tortoise | 31.08 ± 7.42 (n=3) | 34.56 ± 2.98 (n=3) | -3.48 | 1/3 | 1.000 | 1.000 |
| RVC test half | 6.62 ± 1.29 (n=3) | 5.11 ± 0.81 (n=3) | +1.51 | 1/3 | 0.500 | 0.200 |
| New speakers, XTTS | 10.47 ± 0.67 (n=3) | 10.13 ± 0.83 (n=3) | +0.34 | 0/3 | 0.250 | 0.400 |
| New speakers, Tortoise | 27.93 ± 3.40 (n=3) | 32.14 ± 1.09 (n=3) | -4.21 | 2/3 | 0.500 | 0.300 |
| IndicF5 | 27.10 ± 1.12 (n=3) | 31.24 ± 2.30 (n=3) | -4.13 | 3/3 | 0.250 | 0.100 |
| False alarms, language control | 3.02 ± 0.62 (n=3) | 3.31 ± 1.50 (n=3) | -0.28 | 1/3 | 1.000 | 1.000 |
| New speakers, XTTS, AMR-NB | 7.59 ± 0.55 (n=3) | 22.37 ± 0.54 (n=3) | -14.78 | 3/3 | 0.250 | 0.100 |
| New speakers, Tortoise, AMR-NB | 19.52 ± 1.88 (n=3) | 36.48 ± 0.88 (n=3) | -16.96 | 3/3 | 0.250 | 0.100 |
| IndicF5, AMR-NB | 28.01 ± 1.88 (n=3) | 46.13 ± 0.90 (n=3) | -18.13 | 3/3 | 0.250 | 0.100 |
| CM04, AMR-NB | 14.92 ± 5.79 (n=3) | 36.76 ± 2.79 (n=3) | -21.84 | 3/3 | 0.250 | 0.100 |
