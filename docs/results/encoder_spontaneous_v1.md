# The larger encoder on spontaneous speech, and its speed (week 14, run 9)

SSL-AASIST trained on the final-recipe data (248 speakers, both telephone codecs) with seeds
1234 and 2025, the published recipe (RawBoost, whole network, 3 epochs, learning rate 1e-5,
batch 8, best dev epoch kept), scored on every week-13 set plus HiACC, and timed. Notebook:
`notebooks/kaggle_w14_encoder_spontaneous.ipynb`; raw output:
`experiments/results/encoder_spontaneous_v1/`. Every step ran without error.

## Spontaneous speech (HiACC, 3,318 genuine adult clips, 24 speakers, G.711 line)

Share called fake at log-odds 0 (P(genuine) < 0.5, the adapter's operating rule), with a
95% interval from 1,000 resamples of whole speakers:

| Model | Called fake | Speakers above 10% | Dev EER |
|---|---:|---:|---:|
| SSL-AASIST, seed 1234 | 5.8% [3.2, 9.1] | 5 of 24 | 0.00% |
| SSL-AASIST, seed 2025 | 46.7% [37.6, 55.7] | 23 of 24 | 0.00% |
| Adapter, final recipe, seed 1234 (`hiacc_final_v1.md`) | 43.9% [36.5, 51.2] | 22 of 24 | - |

**Reading.** The larger encoder *can* learn a notion of genuine speech that covers
spontaneous talk: one training calls only 5.8% of it fake, the first model in the project
under 10%. It does not do so reliably: a second training of the same recipe calls 46.7%
fake, and both reach 0% dev EER, so nothing in our model selection would have chosen the good
one. Spontaneous speech still needs spontaneous genuine data in training (or at least in the
dev set used to pick a model).

## Speed per 4 s window (Kaggle, median of 20 after 3 warm-up calls)

| Encoder | Parameters | T4 GPU | CPU, 2 threads |
|---|---:|---:|---:|
| SSL-AASIST (XLS-R 300M + AASIST) | 315.9 M | 68.8 ms | 1,136.7 ms |
| wav2vec2-base (the adapter's encoder) | 94.4 M | 25.2 ms | 398.8 ms |

The live path scores a window every 2 s, so the larger encoder keeps up even on two CPU
threads (57% of the budget) and has ample room on a GPU. Its cost on a call is about three
times the adapter's, not a reason to rule it out.

## The rest of the sets (EER %, two seeds; for reference against runs 7 and 8)

| Set | Seed 1234 | Seed 2025 |
|---|---:|---:|
| New speakers, Tortoise, G.711 / AMR-NB | 0.91 / 1.30 | 0.37 / 0.41 |
| CM04 Tortoise, G.711 / AMR-NB | 1.74 / 0.88 | 0.68 / 0.27 |
| IndicF5, G.711 / AMR-NB | 25.83 / 31.77 | 14.86 / 21.84 |
| Chatterbox | 12.58 | 12.41 |
| RVC test half | 6.30 | 5.01 |

These two models are new trainings: the same seed on the GPU does not give the same model
(seed 1234 here reaches 25.8% on IndicF5, 12.7% in run 7), which is one more reason every
claim in the paper rests on several seeds.
