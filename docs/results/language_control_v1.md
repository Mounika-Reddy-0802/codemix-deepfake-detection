# Code-mixing or recording domain? Language-control false alarms (v1)

Genuine clips of 183 speakers who were never trained or tested on; every speaker
contributes all-English, all-Hindi and mixed clips from the same lecture recordings.
A false alarm is a genuine clip scored below P(bonafide) = 0.5. Intervals are 95%
speaker-bootstrap intervals. Built by `src/reporting/language_control.py` from
`experiments/results/language_control_v1/`.

## False-alarm rate by language condition

| Model | all English | all Hindi | mixed |
|---|---|---|---|
| S1 (English only), clean audio | 84.3% [79.2, 89.2] | 85.2% [81.2, 88.7] | 78.1% [72.4, 83.6] |
| S1 (English only), phone line | 43.1% [37.4, 49.1] | 53.2% [47.6, 59.0] | 41.2% [35.2, 47.5] |
| S2 trained clean, clean audio | 5.2% [2.3, 8.5] | 2.9% [0.9, 5.7] | 2.3% [0.5, 4.5] |
| deployed S2, phone line | 4.3% [2.1, 7.0] | 4.1% [2.0, 6.7] | 2.3% [1.0, 3.8] |

## Within-speaker difference to all-English (percentage points)

| Model | mixed - English | Hindi - English |
|---|---|---|
| S1 (English only), clean audio | -9.5 [-15.6, -3.5] (n=183) | -2.5 [-8.4, +3.2] (n=183) |
| S1 (English only), phone line | -5.2 [-13.4, +2.9] (n=183) | +3.5 [-4.2, +11.0] (n=183) |
| S2 trained clean, clean audio | -3.3 [-6.3, -0.6] (n=183) | -2.5 [-6.0, +0.7] (n=183) |
| deployed S2, phone line | -2.6 [-5.8, +0.5] (n=183) | -1.3 [-4.9, +2.2] (n=183) |

## Reading

If code-mixing drove the false alarms, the mixed column would stand out and the
mixed - English difference would be clearly positive. An interval that contains
zero, or a negative difference, means the model treats mixed speech no worse than
the same speaker's single-language speech: the rejections come from the
recording domain, not from the language mix.
