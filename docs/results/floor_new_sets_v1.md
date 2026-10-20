# Detector against shortcut floor on the week-13 sets (v1)

Same procedure as `shortcut_floor_v2.md` (`src/training/floor_compare.py`): each
system's floor is fitted on its own training manifest and scored on the same clips
as the detector; the difference (detector - floor, EER points) has a 95% interval
from a bootstrap over whole speakers. Built by `src/training/floor_new_sets.py`.

| System | Test set | Detector EER | Floor EER | Difference [95% CI] | Verdict |
|---|---|---|---|---|---|
| S2 LoRA, XTTS, clean | new_xtts | 7.24 | 16.79 | -9.54 [-14.04, -4.86] | beats floor |
| S2 LoRA, XTTS, clean | new_tortoise | 13.51 | 40.54 | -27.03 [-33.33, -21.35] | beats floor |
| S2 LoRA, XTTS, clean | cm07 | 37.05 | 34.24 | +2.81 [-3.08, +9.38] | inconclusive |
| S2 LoRA, XTTS, channel | new_xtts | 7.39 | 35.20 | -27.81 [-31.60, -23.89] | beats floor |
| S2 LoRA, XTTS, channel | new_tortoise | 29.74 | 50.45 | -20.71 [-26.41, -17.04] | beats floor |
| S2 LoRA, XTTS, channel | cm07 | 43.92 | 33.99 | +9.94 [+3.33, +16.54] | worse than floor |
| S2 LoRA, XTTS+RVC, clean | new_xtts | 10.22 | 21.30 | -11.08 [-16.43, -5.92] | beats floor |
| S2 LoRA, XTTS+RVC, clean | new_tortoise | 26.13 | 45.11 | -18.98 [-25.14, -13.69] | beats floor |
| S2 LoRA, XTTS+RVC, clean | cm07 | 34.41 | 34.07 | +0.34 [-5.70, +7.38] | inconclusive |
| S2 LoRA, XTTS+RVC, channel (deployed) | new_xtts | 9.43 | 38.03 | -28.61 [-33.05, -24.45] | beats floor |
| S2 LoRA, XTTS+RVC, channel (deployed) | new_tortoise | 29.37 | 58.12 | -28.75 [-33.90, -23.81] | beats floor |
| S2 LoRA, XTTS+RVC, channel (deployed) | cm07 | 30.59 | 35.94 | -5.35 [-11.18, +1.52] | inconclusive |
