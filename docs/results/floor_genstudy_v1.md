# Shortcut floors for the generator study (v1)

Each system's floor (eight low-level statistics) is fitted on that system's own training
manifest and scored on the study's test clips without the `gen_train` speakers. Per seed,
the detector-minus-floor difference has a 95% interval from a bootstrap over whole
speakers; 'beats floor' counts the seeds whose whole interval is below zero. EER (%) are
means over seeds. Built by `src/training/floor_genstudy.py`.

| System | Test set | Seeds | Detector EER | Floor EER | Difference | Beats floor | Worse |
|---|---|---|---|---|---|---|---|
| control | cm09 | 5 | 24.32 | 42.34 | -18.02 | 5/5 | 0/5 |
| control | cm07 | 5 | 20.44 | 39.76 | -19.33 | 5/5 | 0/5 |
| control | new_tortoise | 5 | 7.16 | 38.41 | -31.25 | 5/5 | 0/5 |
| control | cm07_amr | 5 | 19.40 | 40.95 | -21.56 | 5/5 | 0/5 |
| control | new_tortoise_amr | 5 | 4.05 | 52.58 | -48.53 | 5/5 | 0/5 |
| fold_a | cm09 | 5 | 15.75 | 42.34 | -26.58 | 5/5 | 0/5 |
| fold_a | cm07 | 5 | 4.19 | 39.76 | -35.57 | 5/5 | 0/5 |
| fold_a | new_tortoise | 5 | 4.13 | 39.26 | -35.13 | 5/5 | 0/5 |
| fold_a | cm07_amr | 5 | 8.23 | 40.60 | -32.37 | 5/5 | 0/5 |
| fold_a | new_tortoise_amr | 5 | 4.89 | 53.32 | -48.43 | 5/5 | 0/5 |
| fold_b | cm09 | 5 | 7.80 | 42.34 | -34.53 | 5/5 | 0/5 |
| fold_b | cm07 | 5 | 12.43 | 39.41 | -26.99 | 5/5 | 0/5 |
| fold_b | new_tortoise | 5 | 5.00 | 38.65 | -33.65 | 5/5 | 0/5 |
| fold_b | cm07_amr | 5 | 14.31 | 40.95 | -26.65 | 5/5 | 0/5 |
| fold_b | new_tortoise_amr | 5 | 3.53 | 52.65 | -49.12 | 5/5 | 0/5 |
| fold_c | cm09 | 5 | 7.04 | 41.85 | -34.81 | 5/5 | 0/5 |
| fold_c | cm07 | 5 | 4.10 | 39.41 | -35.32 | 5/5 | 0/5 |
| fold_c | new_tortoise | 5 | 4.91 | 39.43 | -34.52 | 5/5 | 0/5 |
| fold_c | cm07_amr | 5 | 7.41 | 40.60 | -33.20 | 5/5 | 0/5 |
| fold_c | new_tortoise_amr | 5 | 4.01 | 53.32 | -49.31 | 5/5 | 0/5 |

The floor is above chance on new_tortoise_amr, so beating it there is a weak bar.
