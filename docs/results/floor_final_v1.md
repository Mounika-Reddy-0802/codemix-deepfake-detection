# Shortcut floors for the final-recipe systems (v1)

One floor (eight low-level statistics, `src.training.floor_compare`) fitted on
`codemix_adapt_train_final.csv`, the training data of both systems, and scored on exactly the
clips each detector was scored on. Per seed, the detector-minus-floor difference has a
95% interval from a bootstrap over whole speakers; 'beats floor' counts the seeds whose
whole interval is below zero. EER (%) are means over seeds. Built by
`src/training/floor_final.py`.

| Detector | Test set | Seeds | Detector EER | Floor EER | Difference | Beats floor | Worse |
|---|---|---|---|---|---|---|---|
| adapter | eval_pool | 5 | 1.13 | 18.26 | -17.12 | 5/5 | 0/5 |
| SSL-AASIST | eval_pool | 5 | 0.25 | 18.26 | -18.00 | 5/5 | 0/5 |
| adapter | cm04 | 5 | 17.73 | 54.59 | -36.86 | 5/5 | 0/5 |
| SSL-AASIST | cm04 | 5 | 1.75 | 54.59 | -52.84 | 5/5 | 0/5 |
| adapter | rvc_test | 5 | 8.53 | 43.78 | -35.25 | 5/5 | 0/5 |
| SSL-AASIST | rvc_test | 5 | 5.17 | 43.78 | -38.61 | 5/5 | 0/5 |
| adapter | new_xtts | 5 | 0.94 | 35.37 | -34.43 | 5/5 | 0/5 |
| SSL-AASIST | new_xtts | 5 | 0.02 | 35.37 | -35.35 | 5/5 | 0/5 |
| adapter | new_tortoise | 5 | 8.81 | 36.51 | -27.70 | 5/5 | 0/5 |
| SSL-AASIST | new_tortoise | 5 | 0.73 | 36.51 | -35.79 | 5/5 | 0/5 |
| adapter | cm07 | 5 | 21.68 | 38.82 | -17.14 | 5/5 | 0/5 |
| SSL-AASIST | cm07 | 5 | 15.79 | 38.82 | -23.04 | 5/5 | 0/5 |
| adapter | new_xtts_amr | 5 | 0.69 | 18.08 | -17.39 | 5/5 | 0/5 |
| SSL-AASIST | new_xtts_amr | 5 | 0.05 | 18.08 | -18.03 | 5/5 | 0/5 |
| adapter | new_tortoise_amr | 5 | 6.04 | 54.94 | -48.90 | 5/5 | 0/5 |
| SSL-AASIST | new_tortoise_amr | 5 | 0.51 | 54.94 | -54.42 | 5/5 | 0/5 |
| adapter | cm04_amr | 5 | 6.90 | 71.39 | -64.49 | 5/5 | 0/5 |
| SSL-AASIST | cm04_amr | 5 | 1.36 | 71.39 | -70.03 | 5/5 | 0/5 |
| adapter | cm07_amr | 5 | 21.32 | 37.81 | -16.49 | 5/5 | 0/5 |
| SSL-AASIST | cm07_amr | 5 | 21.75 | 37.81 | -16.06 | 5/5 | 0/5 |
| adapter | cm09 | 1 | 31.98 | 39.45 | -7.48 | 1/1 | 0/1 |
| SSL-AASIST | cm09 | 5 | 12.86 | 39.45 | -26.59 | 5/5 | 0/5 |

## Reading

The floor is above chance on cm04, cm04_amr, new_tortoise_amr: fitted on the training
recordings, the eight statistics rank genuine and fake the wrong way round there,
because those recordings differ from the training ones. Beating such a floor is a
weak bar, so these rows show that the detector does not lean on these cues, not
that it beats an informative baseline.
