# Week 15 — Lahari — spontaneous speech as data (decision D1)

Every model in the project calls far more genuine spontaneous speech fake than read speech,
and all our code-mixed training speech is read lectures. This week I built a second
spontaneous corpus, then tested the two fixes the team agreed on 1 November (decision D1,
P-038): choosing the operating points again with spontaneous speech in the dev set, and
training with spontaneous speech. Both were decided, and the second one's hypotheses written,
before any result was seen.

## What I did, and why it is built this way

### A second spontaneous corpus, chosen before any audio was read

HiACC alone could not tell a property of one corpus from a property of spontaneous speech.
IndicVoices (AI4Bharat, CC BY 4.0) holds unscripted Hindi from thousands of speakers.
`src/data/indicvoices_inventory.py` read the metadata of all 83 Hindi shards and never the
audio: 2,356 speakers, 828 hours. A fixed draw of 40 row groups (seed 1234) bounds the
download, 395 speakers have enough extempore clips inside it, and a seeded draw froze three
pools from metadata alone (`src/data/indicvoices_pools.py`, `data/manifests/indicvoices_pools.csv`):
150 `iv_train`, 20 `iv_dev`, 30 `iv_test`. Only then was audio downloaded: 6,670 clips, 14.9 h,
level-normalised (`src/data/indicvoices_download.py`). Each speaker got 20 XTTS-v2 clones made
with the project's existing recipe, with transcripts from the speaker's own pool only, so no
test text reaches training (`src/data/indicvoices_jobs.py`); 3,947 passed screening.

The failure is not HiACC's: the final adapter calls 37.5% of the 30 test speakers' genuine
clips fake, against about 1% of read speech, while their XTTS-v2 clones are still caught
(`docs/results/indicvoices_data_v1.md`).

### The guard, changed only as far as D1 allows

`src/training/config_guard.py` now admits `iv_train` in a training manifest and `iv_dev` in a
dev manifest, keeps `iv_test` out of both, and refuses every other IndicVoices speaker or a
clone of one, whatever its source column says. Hindi rows count as Stage-3 data, like
code-mixed rows, so only a LoRA or `s3_native` run may hold them. Each rule has a test in
`tests/test_config_guard.py`, including the real spontaneous config.

### Part (a): operating points chosen again, no retraining

The unchanged rule (P-031), with the 20 `iv_dev` speakers added to every model's dev set.
`src/reporting/operating_points.py` scores old and new points on the same windows. For
SSL-AASIST seed 1234, the voice check's model, genuine spontaneous files called cloned fall
from 7.6% to 0.7% (HiACC) and from 15.3% to 5.0% (IndicVoices), but Tortoise clones passing
rise from 4% to 32% and IndicF5 and Chatterbox clones from 71-78% to 96%. Threshold-free, the
newest generators overlap spontaneous genuine speech (40-44% EER), so no threshold serves both
(`docs/results/operating_points_spont_v1.md`). The team had agreed beforehand to adopt the new
point for the voice check whatever the test sets showed, because a false alarm on a real
person is the costlier error, and the voice check page says what it lets through.

### Part (b): training with spontaneous speech, hypotheses fixed before launch

The final recipe unchanged except for its data: its training set plus the 150 `iv_train`
speakers (31,842 clips, 398 speakers), its dev set plus `iv_dev` (`configs/train_lora_spont.yaml`,
`src/data/spont_recipe.py`). Before launch I wrote down what would count as helping: HiACC
genuine clips called fake down at least 10 points (H1), and G.711 EER on the unseen
generators up at most 2 points (H2). `src/reporting/spontaneous_training.py` codes that rule
unchanged and gives the verdict.

The adapter, three seeds on Kaggle: **a trade-off**. H1 holds by far: 43.9% of HiACC's genuine
clips called fake becomes 1.2% (mean), and HiACC was never trained on. H2 fails on every seed:
Tortoise EER rises 8.2 points and IndicF5 10.7. The two results are one effect seen from two
sides. `src/reporting/spontaneous_separation.py` measures, threshold-free, how far spontaneous
genuine speech sits from each tool's clones: every seed separates them better than the frozen
model (Tortoise 57.6% to 18.8-42.9% EER), while the benchmarks, whose genuine side is read
speech, get worse. The frozen model's good Tortoise number holds only when the genuine speech
is read; against spontaneous genuine speech it is worse than chance
(`docs/results/spontaneous_training_v1.md`).

SSL-AASIST, two seeds: **also a trade-off**, with a smaller cost and a larger gain. HiACC
genuine clips called fake fall from 26.2% to 1.5% (mean), and spontaneous genuine speech now
sits far from the newest generators' clones: IndicF5 from 49% to 15-20% EER, Chatterbox from
39-57% to 8-17%. H2 fails on Tortoise (+3.7) and Chatterbox (+3.4) and holds on IndicF5
(+1.2); seed 1234 stays within the 2 allowed points everywhere, seed 2025 loses 6-9.

As agreed before launch, the frozen models stay the reported ones. Before the retrained
SSL-AASIST was measured in the voice check, I wrote down and committed what it had to do to
replace the voice check's detector: both seeds, each at its own dev-chosen point, keep genuine
speech (spontaneous 97.3%, read 99.0%) and raise the unseen-tool mean by 10 points. Measured
afterwards (Mounika's branch), both catch far more clones (unseen-tool mean 70-75% against 44%),
but seed 2025 judges 97.6% of read genuine speech real, so the voice check keeps its detector
(P-040).

### English calls: a language gate in front of the detectors

The final recipe misses English clones (25.95% EER over the phone line), while S1, the
English-only detector, rejects Hindi-English speech. `src/inference/routing.py` puts a
spoken-language identifier (SpeechBrain's VoxLingua107 model, on CPU) in front of both.
I wrote three rules without a parameter and the selection criterion down before any
development result was read, chose `english_over_hindi` on development audio, and only then
scored the test sets once. English EER falls to 0.60% clean and 9.80% over G.711; on
Hindi-English speech the gate costs at most 0.4 points. The gate also seems to catch
Tortoise, but that is an accent cue: Tortoise is an English text-to-speech model and its
Hindi sounds English-accented, which gives nothing against a Hindi-native generator. It is
reported as a confound, not a gain (`docs/results/routing_v1.md`).

### DET curves and calibration

Every result document quotes the EER, a point chosen after seeing the labels; a deployed
detector runs at a fixed rule. `src/reporting/det_curves.py` adds, from the stored per-clip
scores of every seed, the full DET curves, the Brier score and calibration error, and the
split of errors at the fixed rule: on the unseen tools almost every error is a miss
(IndicF5: 73.7% of clones missed by the final adapter, 61.7% by SSL-AASIST), and many false
alarms come only from the deployed adapter over AMR-NB (`docs/results/calibration_v1.md`).

### Figure work left from week 14

The channel spectrogram (clean, G.711, AMR-NB), the code that draws the architecture and
flowchart diagrams, panel letters and a one-column version of the overview figure, and the
call-chain figure's first label kept off the axis. The figure tests now skip without
matplotlib, as CI has none.

### What went wrong, and how it was caught

- The first text of P-038 described the operating-point rule wrongly. I corrected it from the
  code, with a note, before any `iv_dev` clip was scored.
- I first read the Kaggle quota as 6 h a week: the API prints the hours past whole days. It is
  30 h; the compute note says so and why the adapter's seeds ran in two sessions.
- Two copies of the laptop's queue script were running, each with its own queue, so one
  session was started a second time. The duplicate is outside every verdict; one fetch-only
  watcher replaced both.
- CI had been failing on this branch since its first push: two tests needed `soundfile` and
  `pyarrow`, which CI does not install, and nobody had read the CI result. They now skip
  there, and the IndicVoices inventory checks for the Hugging Face token before importing
  `pyarrow`, so its instructions no longer need it.
- The separation module said P(genuine) and log-odds score files could be mixed. They cannot:
  genuine scores from one file are ranked against clones from another. It now refuses a mix;
  every number reported was computed from files on one scale.

### Voice check v2 on clean files (8 November)

I measured v2 on the same 3,306 clean recordings I measured the detector before it on
(`any_audio_eval score`, telephone front end, its frozen point 0.073), and wrote
`src/reporting/voice_check_v2_clean.py` to compare the two on the recordings both scored, with
the 31 training speakers left out of both. The clean list has no speaker column, so each file's
speaker comes from the manifests by file name, and a name under two speakers stops the table.
IndicF5 clones caught: 0% to 72.7%; Chatterbox: 0% to 86.1%; RVC 20.1% to 58.4%; Tortoise 71.1%
to 86.8%. Genuine speech is judged as before, except one HiACC recording v2 calls cloned. The
same run gives v2's verdict on every demo clip: scene 6 (IndicF5) is now caught, scene 7's
Chatterbox clip still passes (P 0.25 against 0.073); `docs/viva/voice_check_demo.md` and the
demo folder's `expected.csv` now hold v2's verdicts.

### The generator study on SSL-AASIST (P-044): carries over

Voice check v2 heard both new tools, so its numbers on them say nothing about tools it never
heard. I registered the generator study's question for SSL-AASIST before any training
(`docs/results/ssl_generator_study_v1.md`): control, IndicF5-only and Chatterbox-only systems
on the spontaneous recipe, seeds 1234 and 2025, six Kaggle notebooks pinned to the registration
commit. With two seeds the exact rank test cannot pass, so each seed's difference must also
have a 95% speaker-bootstrap interval wholly below zero.

All six ran clean on 10 November and **both folds are confirmed**. Trained also on IndicF5,
Chatterbox (never heard) falls from 11.53% to 3.29% EER on held-out speakers; trained also on
Chatterbox, IndicF5 (never heard) falls from 20.97% to 9.22%. Both seeds lower, every bootstrap
interval below zero, Tortoise and false alarms inside the guards. The adapter study's
unconfirmed direction (fold A, short only on false alarms) holds here: its false alarms move
0.08 points. Two seeds only, and the size of the gain differs a lot between them (fold B 17.4
and 6.1 points), so the direction is shown and the size is not. For the paper,
`src/reporting/genstudy_both_figure.py` draws both studies side by side in one column.

## What is still open

Since 8 November the voice check runs v2 (P-043), which catches most IndicF5 and Chatterbox
clones; a Chatterbox clone replayed into a phone still passes, and the generator rows cost RVC
11-12 points. A tool SSL-AASIST heard helps it catch the other (P-044, above); whether that
reaches tools beyond these two is untested. Telling apart the two
explanations of the trade-off, a delivery cue learned from spontaneous genuine speech against
fluent TTS, or a model made more XTTS-specific by XTTS-only clones, needs clones that speak
spontaneously, or the added speakers cloned by other tools.
The IndicVoices conversation recordings are untested. Whether the voice check shows an
"uncertain" band between real and cloned is the team's decision.
