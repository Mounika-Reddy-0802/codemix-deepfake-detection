# IndicVoices Hindi: inventory, speaker pools and clone jobs (v1)

Plan task A2 (spontaneous genuine speech in training). Every Hindi-English training set
so far is read lecture speech, and the final recipe calls 43.9% of genuine spontaneous
clips fake (`hiacc_final_v1.md`). IndicVoices (AI4Bharat, CC BY 4.0, gated; arXiv
2403.01926) holds unscripted Hindi from thousands of speakers. This document records how
its Hindi part was inventoried and how speakers were chosen, all from metadata, before
any audio was downloaded or any model trained.

**Status of the training use.** Team decision D1 (P-038, 1 November 2026, an amendment like
P-025 and P-028) admits the `iv_train` pool in Stage-3 training and the `iv_dev` pool in the
dev set; `src/training/config_guard.py` enforces both and refuses every other IndicVoices
speaker. The `iv_test` pool stays evaluation-only, a second spontaneous test set beside HiACC.
Everything below up to the spontaneous recipe's manifests was done before D1 was taken.

## Inventory (1 November 2026)

`src/data/indicvoices_inventory.py` read the metadata columns of all 83 Hindi parquet
shards over HTTP range requests, never the audio column (about 8 s per shard).
Aggregate figures: `experiments/results/indicvoices_inventory_hindi.json`.

| | |
|---|---|
| Speakers | 2,356 (1,186 female, 1,167 male, 3 other) |
| Clips | 450,690 |
| Hours | 828.2: extempore 574.5, conversation 181.1, read 72.6 |
| Areas | 1,426 rural, 930 urban speakers |
| States | Uttar Pradesh 957, Madhya Pradesh 720, Bihar 384, Rajasthan 293, Delhi 2 |
| Age groups | 18-30: 808, 30-45: 677, 45-60: 473, 60+: 398 |
| Extempore per speaker | median 15.9 min; 1,757 speakers with at least 5 min |

## Why the download is limited to 40 row groups

Each speaker's clips are spread over about 18 of the 83 shards (median; up to 38), and
parquet audio can only be read a whole row group at a time (about 1,100 clips and 97 MB
per group). Fetching every clip of 200 speakers would mean reading most of the 50 GB.
So a fixed set of 40 row groups was drawn first (seed 1234; 43,746 clips from 32 shards,
about 3.9 GB), and a speaker is **eligible** with at least 20 extempore clips of 2-15 s
inside those groups: 395 speakers. Frozen in `data/manifests/indicvoices_row_groups.csv`
(`src/data/indicvoices_pools.py`).

## Speaker pools

A seeded draw (1234) of the eligible speakers, test pool first, frozen in
`data/manifests/indicvoices_pools.csv` (sha256 prefix `f8ba8c36ce87`). Speaker ids carry
the prefix `iv_`.

| Pool | Role | Speakers | Female | Rural | Usable clips kept |
|---|---|---:|---:|---:|---:|
| `iv_test` | evaluation only | 30 | 17 | 16 | see below |
| `iv_dev` | development only (dev manifest, never training) | 20 | 12 | 14 | |
| `iv_train` | Stage-3 adaptation (decision D1) | 150 | 64 | 102 | |
| All | | 200 | 93 | 132 | 6,670 |

Each pool spans 4 states and all four age groups. The guard enforces the roles
(`tests/test_config_guard.py`): `iv_train` only in a training manifest, `iv_dev` only in a dev
manifest, `iv_test` in neither, and any other IndicVoices speaker (or a clone of one, whatever
its source column says) nowhere in a training config. Until decision D1 was taken every
IndicVoices speaker was refused.

## Clone jobs

XTTS-v2 (the seen tool), 20 clones per speaker in every pool, built with the project's
existing recipe (`src/data/pool_jobs.py`: the longest reference clip of at least 6 s,
romanised transcripts, the length and digit filters, a coprime stride). Transcripts come
only from the speaker's own pool, so no test text reaches training
(`src/data/indicvoices_jobs.py`, tests in `tests/test_indicvoices_jobs.py`).

| Pool | Jobs | Seeds |
|---|---:|---|
| `iv_train` | 3,000 | 20001-23000 |
| `iv_dev` | 400 | 30001-30400 |
| `iv_test` | 600 | 40001-40600 |

Tables: `data/manifests/indicvoices_{iv_train,iv_dev,iv_test}_xtts_jobs.csv`.

## The download (1 November 2026)

`src/data/indicvoices_download.py` read the audio column of the 40 frozen row groups and
kept 6,670 clips (14.9 h): `iv_train` 4,952 clips, `iv_dev` 694, `iv_test` 1,024. The
manifest equals the files on disk and the clips planned from metadata, clip for clip. Each
clip was rendered through the G.711 line at 20 dB (`src.data.channel_bundle`) and through
AMR-NB at 12.2 kbit/s (`src.data.amr_channel`), the two codecs of the final recipe. The
clip manifests carry transcripts and per-speaker demographics from a gated corpus, so they
are not committed; they rebuild from the tracked row groups and pools.

## A second spontaneous test set, before any training

The 1,024 genuine clips of the 30 `iv_test` speakers, over the G.711 line, scored by the
final adapter (seed 1234) exactly as HiACC was (`src/training/false_alarm.py`: one 4 s crop
per clip, called fake when P(genuine) < 0.5, 1,000 resamples of whole speakers):

| Genuine speech | Clips | Speakers | Called fake |
|---|---:|---:|---:|
| IndicVoices extempore Hindi (`iv_test`) | 1,024 | 30 | **37.5%** [26.1, 49.0] |
| HiACC spontaneous (`hiacc_final_v1.md`) | 3,318 | 24 | 43.9% [36.5, 51.2] |
| Read lectures, original test pool (reference) | 1,566 | 15 | 1.1% |

Difference from read lectures: +36.4 points [+25.2, +48.3]
(`experiments/results/indicvoices_iv_test_v1/final_s1234_clip.json`).

The same 1,024 clips scored by SSL-AASIST on Kaggle (Kaggle Run 2, 1 November; the project's
SSL-AASIST clip scorer, fake when the log-odds of genuine are below 0; the average of the two
checkpoints fake when its mean P(genuine) is below 0.5), each against the same model's read
lectures (`experiments/results/indicvoices_iv_test_v1/ssl_{s1234,s2025,avg}_clip.json`):

| Model | IndicVoices `iv_test` called fake | Speakers above 10% | Read lectures called fake | Difference |
|---|---:|---:|---:|---:|
| Final adapter | 37.5% [26.1, 49.0] | 23 of 30 | 1.1% | +36.4 [+25.2, +48.3] |
| SSL-AASIST seed 1234 | 14.6% [7.1, 23.5] | 9 | 0.4% | +14.3 [+6.8, +23.1] |
| SSL-AASIST seed 2025 | 39.5% [28.5, 50.5] | 25 | 0.6% | +38.9 [+28.0, +49.6] |
| Average of the two | 17.4% [9.0, 26.6] | 11 | 0.3% | +17.1 [+8.8, +26.3] |

Every model rejects spontaneous speech far more often than read speech, and the two SSL-AASIST
checkpoints differ by 25 points on the same clips, as they did on HiACC (6.4% and 30.9%,
`ensemble_v1.md`). **The
spontaneous-speech failure is not a property of HiACC**: unscripted speech from 30 new
speakers of a different corpus is rejected at a similar rate. IndicVoices speech is mostly
Hindi rather than Hindi-English, so speaking style and language mix both differ from the
training data; the result is the size of the problem, not its cause. This test set is
evaluation-only whatever decision D1 is taken, and is the second measure (beside HiACC) of
whether spontaneous training data helps.

## The same speakers through the live path, every model

The 1,024 clips stitched into 94 genuine 60 s calls and replayed through the live path at
each model's own dev-chosen operating point, exactly as HiACC was validated (chooses
nothing; `experiments/results/indicvoices_iv_test_v1/live_path.json`, windows in
`experiments/calibration_{final,ssl,ssl_s2025,ssl_avg}/iv_test_windows.csv`):

| Model | IndicVoices test: calls warned / likely cloned (of 94) | HiACC: warned / likely cloned (of 41) | IndicVoices clips called fake |
|---|---|---|---:|
| Final adapter | 93 / 88 | 41 / 40 | 38.5% |
| SSL-AASIST seed 1234 | 79 / 42 | 11 / 1 | 13.3% |
| SSL-AASIST seed 2025 | 90 / 69 | 31 / 15 | 41.3% |
| Average of the two | 87 / 63 | 23 / 10 | 19.5% |

(Clips called fake: each clip's mean window P(genuine) below 0.5.)

**No model handles spontaneous speech.** SSL-AASIST seed 1234 looked as if it did on HiACC
(11 of 41 calls warned, 6.4% of clips), which was the main argument for the larger encoder;
on IndicVoices it warns 79 of 94 calls and calls 13.3% of the clips fake, twice its HiACC
rate. One spontaneous test set overstated it. With two corpora in hand, every model warns on
most genuine spontaneous callers, and the gap to read speech (1.1% for the final adapter) is
the clearest measured weakness of the project.

## The clones (Kaggle Run 1, 1 November 2026)

`notebooks/kaggle_w15_indicvoices_clones.ipynb` (commit `9d75976`) generated all 4,000 jobs
and screened them with the week-13 quality check; each passing clone was saved clean and over
the G.711 line, and the AMR-NB copies were rendered on the laptop (`src.data.amr_channel`).

| Pool | Generated | Passed the check | Manifests |
|---|---:|---:|---|
| `iv_test` | 600 | 598 | `iv_xtts_test{,_channel20,_amr}.csv` |
| `iv_dev` | 400 | 398 | `iv_xtts_dev{,_channel20,_amr}.csv` |
| `iv_train` | 3,000 | 2,951 | `iv_xtts_train{,_channel20,_amr}.csv` |

## Clones of the test speakers (evaluation only)

The 30 `iv_test` speakers' 1,024 genuine clips and their 598 clones, both over G.711. The final
adapter scored all of them together on the laptop (the project's clip scorer;
`experiments/results/indicvoices_iv_test_v1/final_s1234_clones_g711.json`). For SSL-AASIST the
genuine clips are Kaggle Run 2's scores and the clones were scored on the laptop CPU; 32 genuine
clips scored both ways agree to 0.0001 in log-odds, so the two halves are on the same scale
(`ssl_clones_g711.json`, `ssl_{s1234,s2025,avg}_clones_g711_scores.csv`). Each model at its fixed
clip rule:

| | Final adapter | SSL-AASIST s1234 | SSL-AASIST s2025 | Average |
|---|---:|---:|---:|---:|
| EER | 14.2% [12.6, 16.2] | 3.0% | 2.0% | 2.0% |
| Clones missed | 3.7% | 0.0% | 0.0% | 0.0% |
| Genuine clips called fake | 37.5% | 14.6% | 39.5% | 17.4% |
| *Diagnostic only:* genuine called fake where 95% of clones are caught | 33.3% | 1.2% | 1.1% | 0.6% |

For comparison, XTTS-v2 on the read-speech new speakers through the same chain
(`robustness_v1.md`): adapter EER 0.7% and 5.3% missed, SSL-AASIST 0.0-0.3% and none missed.

**The clones are caught; the genuine speakers are not believed.** Every model catches at least
96% of the clones, as many as on read speech, so a spontaneous speaker's XTTS-v2 clone is no
harder to detect. The errors are all on the genuine side, and the two encoders fail there
differently:

- **The adapter cannot rank spontaneous speech.** Its EER is twenty times the read-speech
  figure, and even an operating point that catches only 95% of the clones would still call a
  third of the genuine clips fake: spontaneous genuine speech and clones overlap in its scores.
- **SSL-AASIST ranks it well, at the wrong place.** Both checkpoints separate genuine
  spontaneous clips from clones at 2-3% EER, and an operating point catching 95% of the clones
  would call about 1% of the genuine clips fake. Their 14.6% and 39.5% false alarms come from
  the fixed rule (log-odds below 0), which sits inside the spontaneous genuine scores. This is
  also where the two checkpoints differ: their ranking is equally good, their placement of
  spontaneous speech is not.

The last row reads a threshold off the test set and is therefore a diagnosis, not a result:
no threshold is chosen here. It suggests that for SSL-AASIST an operating point chosen on
**spontaneous dev speech** (the `iv_dev` pool) might fix most of the false alarms without any
retraining, which a dev set of read lectures cannot do. That is the cheaper half of decision
D1, and it is testable: choose the threshold on `iv_dev` only, then score `iv_test` and HiACC
once. These clones are of the seen tool (XTTS-v2); an unseen generator would not separate this
well.

## The spontaneous recipe's manifests (built, not trained)

`python -m src.data.spont_recipe` (standard columns only, the pool roles checked):

| Manifest | Clips | Speakers | Genuine / fake |
|---|---:|---:|---|
| `codemix_adapt_train_spont.csv` | 31,842 | 398 (248 + 150 `iv_train`) | 19,148 / 12,694 |
| `codemix_adapt_dev_spont_channel20.csv` | 1,598 | 22 (2 + 20 `iv_dev`) | 880 / 718 |

The IndicVoices part is 15,806 of the training clips (4,952 genuine and 2,951 clones, each
over both codecs), about half the set; the dev set gains 1,092 clips to the 506 every operating
point has been chosen on so far. `config_guard` accepts both since decision D1 was recorded.

## Next: decision D1, taken on 1 November (P-038)

1. **Operating points chosen on spontaneous dev speech** (part a): the `iv_dev` pool (694
   genuine clips and 398 clones) added to the dev set, the checkpoints unchanged, then the
   held-out sets scored once: `operating_points_spont_v1.md`.
2. **Retraining on the spontaneous recipe** (part b): the adapter (seeds 1234, 2025, 7) and
   SSL-AASIST (seeds 1234, 2025), hypotheses written before launch:
   `spontaneous_training_v1.md`.
