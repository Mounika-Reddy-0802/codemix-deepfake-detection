# The larger encoder on the final recipe (week 13, run 7)

SSL-AASIST (XLS-R 300M front end, `src/baselines/ssl_aasist_model.py`) fine-tuned on exactly
the data of the final recipe: `data/manifests/codemix_adapt_train_final.csv`, 16,036 clips,
248 speakers, both telephone codecs (simulated G.711 and real AMR-NB). Whole network,
RawBoost, 3 epochs, learning rate 1e-5, batch 8, best dev epoch kept; seeds 1234, 2025, 7.
All three models finished with no failed step. Raw output: `experiments/results/final_ssl_v1/`.
EER (%) mean +/- sd; the adapter column is the five-seed final recipe of
`docs/results/final_recipe_v1.md` (CM09: seed 1234 only, 31.98).

| Test set | SSL-AASIST (3 seeds) | Adapter, final recipe |
|---|---|---|
| Original test pool | 0.27 +/- 0.14 | 1.13 +/- 0.18 |
| New speakers, XTTS | 0.04 +/- 0.04 | 0.94 +/- 0.34 |
| New speakers, Tortoise | 0.48 +/- 0.04 | 8.81 +/- 1.43 |
| CM04, Tortoise | 1.33 +/- 0.38 | 17.73 +/- 2.27 |
| CM07, IndicF5 | 16.54 +/- 3.34 | 21.68 +/- 3.74 |
| RVC test half | 5.11 +/- 0.67 | 8.53 +/- 1.91 |
| Genuine called fake | 0.41 +/- 0.40 | 0.51 +/- 0.67 |
| AMR-NB, new speakers XTTS | 0.08 +/- 0.07 | 0.69 +/- 0.24 |
| AMR-NB, new speakers Tortoise | 0.55 +/- 0.29 | 6.04 +/- 1.17 |
| AMR-NB, CM04 | 1.31 +/- 0.41 | 6.90 +/- 2.06 |
| AMR-NB, CM07 (IndicF5) | 21.55 +/- 7.42 | 21.32 +/- 2.27 |
| CM09 (Chatterbox), G.711 | 11.85 +/- 3.20 | 31.98 (one seed) |

Lower on ten of twelve sets; level on false alarms and on IndicF5 over AMR-NB (21.6 against
21.3). IndicF5 is still the hardest tool for it (16.5% over G.711, 21.6% over AMR-NB, with
a wide seed spread over AMR-NB). Not measured: the shortcut floor for either system with the
final data, the real-time cost of a 300M encoder on a live call, the spontaneous HiACC set.
