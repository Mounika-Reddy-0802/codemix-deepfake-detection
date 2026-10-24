# Final recipe vs deployed adapter (week 13, run 4)

Final recipe: LoRA adapter trained on the 248-speaker set over both telephone codecs
(simulated G.711 and real AMR-NB, `configs/train_lora_final.yaml`). Deployed: the
21-speaker G.711 adapter. Five seeds each (1234, 2025, 7, 42, 99), same seeds paired.
EER (%) mean +/- sd; control_fa = genuine clips called fake (%). Raw output:
`experiments/results/final_v1/`.

| Test set | Final | Deployed | Final better in |
|---|---|---|---|
| Original test pool | 1.13 +/- 0.18 | 3.50 +/- 0.75 | 5/5 |
| CM04 (Tortoise) | 17.73 +/- 2.27 | 36.18 +/- 5.48 | 5/5 |
| RVC test half | 8.53 +/- 1.91 | 6.11 +/- 0.71 | 0/5 |
| New speakers, XTTS | 0.94 +/- 0.34 | 12.00 +/- 1.90 | 5/5 |
| New speakers, Tortoise | 8.81 +/- 1.43 | 30.77 +/- 2.11 | 5/5 |
| CM07 (IndicF5) | 21.68 +/- 3.74 | 34.90 +/- 1.43 | 5/5 |
| Language control false alarms | 0.51 +/- 0.67 | 2.43 +/- 1.21 | 4/5 |
| New XTTS, AMR-NB | 0.69 +/- 0.24 | 23.17 +/- 0.85 | 5/5 |
| New Tortoise, AMR-NB | 6.04 +/- 1.17 | 36.28 +/- 2.29 | 5/5 |
| CM07, AMR-NB | 21.32 +/- 2.27 | 46.47 +/- 1.17 | 5/5 |
| CM04, AMR-NB | 6.90 +/- 2.06 | 37.93 +/- 3.83 | 5/5 |

The final recipe is better on every set except RVC, where it is worse in all five
seeds (8.5% against 6.1%). With five paired seeds the smallest possible two-sided
sign-flip p is 0.0625, so each single set is indicative; the direction is the same on
ten of eleven sets. On the real AMR-NB codec the deployed adapter collapses (23-47%)
while the final recipe stays at the level of its G.711 results.
