# Spontaneous speech in training (plan task A2.5–A2.6, decision D1 part b)

**Status: hypotheses written on 1 November 2026, before any run was launched.** Results are
added below them when the runs finish; nothing above the results section changes after launch.

## Why

Every model in the project rejects genuine spontaneous speech far more often than read speech:
the final adapter calls 43.9% of HiACC's genuine clips and 37.5% of IndicVoices `iv_test`'s
fake, against about 1% of read lectures (`hiacc_final_v1.md`, `indicvoices_data_v1.md`). All
the code-mixed training data is read speech (MUCS lectures). Team decision D1 (P-038, 1
November) lets a speaker-disjoint slice of IndicVoices Hindi into Stage-3 training to test
whether spontaneous training speech removes the failure.

## What is trained

The final recipe unchanged except for its data (`configs/train_lora_spont.yaml`,
`src/data/spont_recipe.py`): the final recipe's training set plus the 150 `iv_train` speakers
(genuine extempore Hindi and XTTS-v2 clones of the same speakers, G.711 and AMR-NB; 31,842
clips, 398 speakers), and the final dev set plus the 20 `iv_dev` speakers (1,598 clips, 22
speakers). `config_guard` checks both manifests before anything trains.

| Run | Model | Seeds | Notebook |
|---|---|---|---|
| B-adapter | wav2vec2-base + LoRA, as the final recipe | 1234, 2025, 7 | `notebooks/kaggle_w15_spont_adapter.ipynb` |
| B-SSL | SSL-AASIST, as week 14 | 1234, 2025 | `notebooks/kaggle_w15_spont_ssl_s{1234,2025}.ipynb` |

## The comparison (frozen models, same seeds, same scorers)

| Measure | Frozen adapter (seeds 1234 / 2025 / 7) | Frozen SSL-AASIST (seeds 1234 / 2025) |
|---|---|---|
| HiACC genuine called fake, one 4 s crop, fixed rule | 43.9% (seed 1234 only; the other two checkpoints were not kept) | 5.76% / 46.68%, mean 26.22% |
| CM04 Tortoise EER, G.711 | 17.48 / 16.60 / 15.71, mean 16.60 | 1.74 / 0.68, mean 1.21 |
| CM07 IndicF5 EER, G.711 | 25.66 / 18.09 / 23.19, mean 22.31 | 25.83 / 14.86, mean 20.35 |
| CM09 Chatterbox EER, G.711 | not scored by the adapter notebook | 12.58 / 12.41, mean 12.50 |

Sources: `experiments/results/final_v1/summary.txt` (adapter EERs), `hiacc_final_v1.md`
(adapter HiACC), `experiments/results/encoder_spontaneous_v1/summary.txt` (SSL-AASIST). The
fixed rule is P(genuine) < 0.5 for the adapter and log-odds < 0 for SSL-AASIST.

## Hypotheses (fixed before launch)

**H1, the purpose (HiACC: spontaneous speech from a corpus never trained on).** Genuine HiACC
clips called fake at the fixed rule fall clearly:

- adapter: mean over the three seeds at most **33.9%** (10 points below 43.9%), and seed 1234
  below its frozen 43.9%;
- SSL-AASIST: mean over the two seeds at most **16.2%** (10 points below 26.2%), and neither
  seed more than 2 points above its own frozen seed (at most 7.8% and 48.7%).

**H2, the cost (unseen generators).** Paired by seed, the mean increase in G.711 EER is at most
**+2.0 points** on CM04 (Tortoise) and on CM07 (IndicF5), and for SSL-AASIST also on CM09
(Chatterbox).

**Verdict per model family:** *helps* if H1 and H2 both hold; *trade-off* if H1 holds and H2
fails; *does not help* if H1 fails. The verdict needs every planned seed; with fewer it is
reported as incomplete.

**Reported, not judged:** `iv_test` (now the same corpus as training, speaker-disjoint) EER and
genuine called fake, G.711 and AMR-NB; the evaluation pool, RVC test half, new speakers, the
language control (read genuine speech) and every AMR-NB set.

## What does not change, whatever the result

The frozen final adapter, the SSL-AASIST checkpoints and the voice check stay as they are
before the viva; this run is an extra experiment reported beside them. If its results are not
in by 5 November it becomes future work.

## Compute (1 November, before launch)

The project's Kaggle account has a GPU quota of 6 h a week (3.6 h used on 1 November; it
resets on 7 November). Measured run times: the final recipe's adapter took 43 min a seed
(`final_v1/runs.json`) and week 14's SSL-AASIST 2.9–3.3 h a seed
(`encoder_spontaneous_v1/runs.json`), both on half this data. So:

- B-adapter seed 1234 runs first, in the 2.4 h left (the notebook starts a seed only with
  100 min of quota to spare; `RUN_SEEDS`, `GPU_HOURS`). It is the seed with a paired HiACC
  measure (43.9%).
- Seeds 2025 and 7 need a second session (about 3 h): another team member's account, or this
  one after 7 November.
- B-SSL needs about 6 h a seed, a whole week's quota. It is not launched; it runs only if a
  team account can take it in time, otherwise it is future work. The hypotheses above do not
  change.

## Results

Not yet run.
