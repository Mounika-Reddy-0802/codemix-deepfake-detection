# Spontaneous speech in training (plan task A2.5–A2.6, decision D1 part b)

**Status: hypotheses written on 1 November 2026, before any run was launched.** Results are
added below them when the runs finish; nothing above the results section changes after launch
except the session log in the compute table. **Adapter: all three seeds in, verdict
*trade-off*** (2 November). SSL-AASIST: running.

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

## Compute and launch (1 November)

The account's weekly GPU quota is 30 h (3.6 h used at launch) and Kaggle runs at most two
GPU sessions at once. A first reading of the quota was wrong (6 h: the API's summary printed
the hours past whole days); for a few minutes the plan was cut to fit it, which is why the
adapter's seeds run in two sessions. The seeds are independent runs of the same notebook on
the same commit, so the split changes nothing in the results; each session's `summary.json`
records its `run_seeds`. Launched from the laptop (`kaggle kernels push`, T4):

| Session | Seeds | Started | Finished | Commit |
|---|---|---|---|---|
| `w15-spont-adapter` | 1234 | 1 Nov 23:07 | 2 Nov 01:20 | `2dc1cda` |
| `w15-spont-ssl-s1234` | SSL-AASIST 1234 | 1 Nov 23:12 | | |
| `w15-spont-adapter-b` | 2025, 7 | 2 Nov 01:22 | 2 Nov 04:28 | `c13e7a8` |
| `w15-spont-ssl-s2025` | SSL-AASIST 2025 | when a slot frees | | |

Between `2dc1cda` and `c13e7a8` the branch gained result files and a `--scores-in` option in
`false_alarm` that the notebook does not use; training and scoring code is identical, so the
three adapter seeds are one experiment. At 04:28 the laptop's queue script started
`w15-spont-adapter-b` a second time (it still listed the session after being restarted). That
duplicate run is not part of the plan and does not enter any verdict.

## Results

### Adapter, all three seeds: verdict **trade-off**

RUN HEALTH clean in both sessions. Frozen values are the final recipe on the same sets and
seeds (`final_v1`); its HiACC measure exists for seed 1234 only, because the checkpoints of
seeds 2025 and 7 were not kept. The verdict is `src/reporting/spontaneous_training.py`, the
rule above applied to the runs (`experiments/results/spontaneous_training_v1/verdict_adapter.json`).

| Measure | Frozen, seeds 1234 / 2025 / 7 | Spontaneous recipe, seeds 1234 / 2025 / 7 | Paired mean change |
|---|---|---|---:|
| **H1** HiACC genuine called fake (3,318 clips, 24 speakers) | 43.85 / – / – | 1.05 / 0.87 / 1.72, mean **1.22** | −42.8 (seed 1234) |
| **H2** CM04 Tortoise EER, G.711 | 17.48 / 16.60 / 15.71 | 19.55 / 27.52 / 27.45 | **+8.24** |
| **H2** CM07 IndicF5 EER, G.711 | 25.66 / 18.09 / 23.19 | 29.96 / 37.72 / 31.52 | **+10.75** |
| CM04 / CM07 EER over AMR-NB | 7.21 / 4.59 / 5.27 and 21.49 / 18.86 / 22.09 | 18.59 / 24.90 / 25.37 and 37.62 / 33.89 / 28.79 | +17.26 / +12.62 |
| New speakers' Tortoise EER, G.711 | 9.00 / 8.55 / 6.79 | 15.59 / 13.96 / 14.85 | +6.68 |
| RVC test half EER | 10.34 / 7.59 / 9.21 | 7.58 / 19.87 / 8.89 | +3.06 |
| Evaluation pool EER (XTTS-v2) | 1.34 / 1.16 / 1.26 | 1.08 / 1.34 / 0.76 | −0.19 |
| New speakers' XTTS-v2 EER, G.711 | 1.53 / 0.86 / 0.67 | 0.84 / 0.86 / 0.29 | −0.36 |
| Language control, read genuine called fake | 0.16 / 1.63 / 0.62 | 0.08 / 1.16 / 0.16 | −0.34 |
| `iv_test` genuine called fake (same corpus as training) | 37.5 / – / – | 0.68 / 2.25 / 1.95 | |
| `iv_test` EER, G.711 / AMR-NB | 14.2 / – (seed 1234) | 1.48 / 1.48 / 1.48 and 0.63 / 0.49 / 0.63 | |

H1 holds on every condition: mean 1.22% (allowed at most 33.9%), seed 1234 1.05% (below 43.85%).
H2 fails on both sets: +8.24 and +10.75 points against +2.0 allowed, and every seed rises on
both. By the rule fixed before launch the adapter's verdict is **trade-off**. The three
identical `iv_test` EERs are a coincidence of set size, not one model three times: each seed's
scores differ (correlation 0.96–0.98, different thresholds), and each crosses with 15 of 1,024
genuine clips and 9 of 598 clones wrong, mostly different clips (5 and 1 shared by all three).

**Diagnostics** (added after the first result; they are not part of the hypotheses and choose
nothing).

*Threshold-free separation of spontaneous genuine speech.* EER of the 4,342 genuine spontaneous
clips (HiACC and `iv_test`) against each tool's clones, same scorer
(`src/reporting/spontaneous_separation.py`, `experiments/results/spontaneous_training_v1/separation_*.json`):

| Clones | Frozen, seed 1234 | Spontaneous recipe, seeds 1234 / 2025 / 7 |
|---|---:|---|
| XTTS-v2 of the `iv_test` speakers (598) | 13.38 | 1.84 / 1.01 / 1.34 |
| RVC test half (317) | 38.80 | 20.23 / 11.33 / 8.57 |
| Tortoise, new speakers (222) | 51.80 | 28.39 / 12.26 / 21.17 |
| Tortoise, CM04 (458) | 57.63 | 42.90 / 18.77 / 28.34 |
| IndicF5, CM07 (526) | 72.82 | 51.98 / 38.21 / 37.10 |

*At the fixed rule the model calls more of everything genuine.* Clones passing as genuine at
P(genuine) ≥ 0.5: Tortoise CM04 66.2% frozen (seed 1234) against 84.5 / 73.8 / 85.2%; IndicF5
84.0% against 97.5 / 91.3 / 90.7%; RVC 36.9% against 73.8 / 47.9 / 25.6%; new speakers' XTTS-v2
7.8% against 3.3 / 0.5 / 0.3%.

### Reading

1. **The purpose is met.** HiACC is a corpus no model trained on, so the fall from 43.9% to
   about 1% of its genuine clips called fake is not a same-corpus effect. Every seed also
   separates spontaneous genuine speech from every tool's clones better than the frozen model
   did, which no operating point can do (`operating_points_spont_v1.md`).
2. **The cost is real and above what was allowed.** The unseen-generator benchmarks lose 8–11
   points of G.711 EER and 13–17 over AMR-NB. Their genuine side is read speech (MUCS
   lectures), so the new model separates spontaneous genuine speech from clones better and
   read genuine speech from clones worse.
3. **The frozen model's Tortoise number depends on the genuine speech being read.** Seed 1234
   frozen: 17.5% EER against read genuine speech (CM04), 57.6% against spontaneous genuine
   speech, worse than chance. The new seeds: 19.5–27.5% against read and 18.8–42.9% against
   spontaneous. Worse on the benchmark, less dependent on speaking style.
4. **Not tested here, two explanations fit.** (a) Every genuine training clip from IndicVoices
   is spontaneous and every clone is fluent TTS, so part of what the model learns may be
   delivery rather than synthesis. (b) The added clones are all XTTS-v2, which makes the model
   more XTTS-specific: XTTS-v2 sets improve while Tortoise and IndicF5 worsen. Telling them
   apart needs clones that speak spontaneously, or the same added speakers cloned by other
   tools.
5. **What does not change.** As agreed before launch, the frozen adapter stays the project's
   model; this run is reported beside it. A recipe that keeps both results is future work: more
   generator variety in the added clones, and clones made from spontaneous speech.

### How to reproduce the adapter numbers

The two sessions' outputs are in `experiments/results/spontaneous_training_v1/` (`adapter_s1234/`,
`adapter_s2025_7/`; `frozen_s1234/` holds the frozen seed's clone scores). On a laptop, no GPU:

```bash
O=experiments/results/spontaneous_training_v1
python -m src.reporting.spontaneous_training \
    --adapter-runs $O/adapter_s1234/runs.json $O/adapter_s2025_7/runs.json \
    --out $O/verdict_adapter.json
D=$O/adapter_s2025_7; S=2025   # seed 1234: D=$O/adapter_s1234; S=1234
python -m src.reporting.spontaneous_separation \
    --genuine $D/spont_s${S}__hiacc_scores.csv $D/spont_s${S}__iv_test_scores.csv \
    --clones xtts_v2=$D/spont_s${S}__iv_test_scores.csv tortoise_cm04=$D/spont_s${S}__cm04_scores.csv \
    tortoise_new=$D/spont_s${S}__new_speakers_scores.csv@tortoise \
    indicf5=$D/spont_s${S}__cm07_scores.csv rvc=$D/spont_s${S}__rvc_test_scores.csv \
    --out $O/separation_spont_adapter_s${S}.json
```

The frozen row uses `hiacc_false_alarm_v1/final_s1234_channel_scores.csv` and
`indicvoices_iv_test_v1/final_s1234_clones_g711_scores.csv` as genuine and as XTTS-v2 clones, and
`frozen_s1234/final_s1234__*_scores.csv` for the other tools.
