# False alarms by Code-Mixing Index on the evaluation pool (v1)

Genuine clips of the 15 evaluation-pool speakers, grouped by the CMI of their
transcript (0 = one language, 50 = an even mix). A false alarm is P(bonafide) < 0.5;
intervals are 95% speaker-bootstrap intervals. Built by
`src/reporting/cmi_bands.py`.

| Model | CMI 0-10 | CMI 10-25 | CMI 25-40 | CMI 40-50 | Spearman(CMI, score) |
|---|---|---|---|---|---|
| S1 (English only), clean audio | 94.5% [88.3, 98.5] (n=476) | 95.2% [91.1, 98.0] (n=524) | 95.1% [90.6, 98.0] (n=365) | 93.5% [85.3, 98.4] (n=201) | -0.011 |
| S1 (English only), phone line | 54.4% [41.8, 66.5] (n=476) | 42.9% [33.0, 53.6] (n=524) | 38.6% [27.9, 49.4] (n=365) | 40.8% [26.3, 52.9] (n=201) | +0.133 |
| deployed S2, phone line | 1.3% [0.4, 2.1] (n=476) | 0.8% [0.2, 1.4] (n=524) | 0.5% [0.0, 1.5] (n=365) | 1.0% [0.0, 3.0] (n=201) | +0.096 |

A detector that confused code-mixing with spoofing would show false alarms
rising across the bands and a negative correlation between CMI and its
P(bonafide). Flat bands and a correlation near zero point to the recording
domain instead (see `language_control_v1.md`).
