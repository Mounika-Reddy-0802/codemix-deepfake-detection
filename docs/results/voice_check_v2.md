# Voice check v2: the voice check's detector trained on the newest tools (v1)

**Status: run on 8 November 2026. Answer: adopted** — both seeds met all five conditions, so
seed 1234 is now the voice check's detector (`configs/voice_check.yaml`). The rule below was
written on 7 November before any training or scoring of this model, so it could not be fitted
to its results. Team decision P-042; plan
task A3 of the review plan. The rule is coded in `src/reporting/voice_check_v2.py` and
committed with this page.

## Why

The voice check (any recording judged real or cloned) runs SSL-AASIST. It keeps genuine
speech safe, but on clean wideband files it catches **none** of the IndicF5 and Chatterbox
clones (0 of 105 and 0 of 130), and over the phone line 12.5% and 4.4%
(`docs/voice_check.md`). Those are the two newest tools in the project and the ones a user is
most likely to meet. The retrained SSL-AASIST of decision D1 part (b) caught more of them
without ever hearing them (35.6-45.4% and 51.6%) but was not adopted (P-040).

v2 trains the voice check's model on those tools directly. In the paper's terms that makes
IndicF5 and Chatterbox **seen** tools for this model only; the paper's unseen-tool numbers come
from detectors that never heard them and do not change.

## What is trained

SSL-AASIST exactly as the spontaneous recipe's runs (RawBoost, whole network, 3 epochs,
learning rate 1e-5, batch 8, best dev epoch kept), seeds 1234 and 2025, on
`data/manifests/codemix_adapt_train_vcv2.csv` (`src/data/voice_check_v2.py`): the spontaneous
recipe plus the generator study's training half (P-041), i.e. the IndicF5 and Chatterbox
clones of the 31 `gen_train` speakers over G.711 (282 and 265) and those speakers' genuine
clips over both codecs (608). The dev set is the spontaneous recipe's, unchanged. Tortoise is
in no training set, so it stays unseen. The guard checks the same manifests through
`configs/train_lora_vcv2.yaml`, as `train_lora_spont.yaml` does for the spontaneous recipe.

## Adoption rule (fixed now)

Each seed at its own operating point, chosen on dev only by the unchanged rule (P-031), then
measured with `src/reporting/any_audio_accuracy.py` exactly as the current detector: one 4 s
head crop per clip on the G.711 sets, HiACC on the same 1,000 clips, the real calls by their
mean window P(genuine). **Every quantity, for the reference and the candidates alike, leaves
out the 31 `gen_train` speakers.** Reference: the current voice check (frozen seed 1234 at
0.082).

Seed 1234 replaces the current detector only if **both** seeds meet all five:

1. spontaneous genuine (HiACC) judged real at least 97.3%;
2. read genuine judged real at least 99.0% on the new speakers and on the language control,
   and at least 11 of 12 genuine live calls and 5 of 6 genuine replays;
3. IndicF5 and Chatterbox clones (held-out speakers) caught **at least 50% each** (the
   reference: 12.5% and 4.4% over the line);
4. the tools v2 never heard (Tortoise on the new speakers, Tortoise CM04, RVC): none more than
   10 points below the reference;
5. XTTS-v2 clones caught at least 98.0%.

Conditions 1, 2 and 5 are P-040's, unchanged. Otherwise the current detector stays, and the
result is reported either way. Clean-file and field-recording accuracy of v2 are reported, not
judged. Seed 1234 because it is the seed the voice check and every deployed model use;
requiring both seeds keeps the choice from resting on the better seed.

## Run

Two Kaggle notebooks, one seed each (`notebooks/kaggle_w15_vcv2_ssl_s{1234,2025}.ipynb`),
pinned to the commit that registered this rule; about 6 GPU hours each, started when the
generator study frees Kaggle's two sessions. Then the dev-only operating points, the real
calls, and:

```
python -m src.reporting.voice_check_v2 --out experiments/results/voice_check_v2.json
```

**Correction before any result (7 November, 09:42, before training started).** An independent
read of the code found that excluding the `gen_train` speakers crashed on the reference's HiACC
file, which has no speaker column. The exclusion is now applied to the four sets that hold any of
the 61 cloned speakers (new speakers, CM04, IndicF5, Chatterbox); HiACC, the language control,
RVC and the calls hold none of them. The conditions and thresholds are unchanged. The guard gate
`configs/train_lora_vcv2.yaml` is a LoRA config only so that `config_guard` checks the same
manifests, as `train_lora_spont.yaml` does for the spontaneous recipe; the model itself is
SSL-AASIST fine-tuned whole.

## Results

**Answer: adopted.** `python -m src.reporting.voice_check_v2` on 8 November
(`experiments/results/voice_check_v2.json`): both seeds meet all five conditions, so seed 1234
replaces the voice check's detector.

**Training and operating points (dev only).** Both Kaggle runs finished with every step clean.
Best dev EER 0.25% (seed 1234) and 0.44% (seed 2025), both at epoch 2 of 3. The unchanged rule
(P-031: the most sensitive point with at most 5% of genuine 60 s dev calls alerted) chose 0.073
and 0.591; at those points no genuine dev call reached "likely cloned" and every fake dev call
did (seed 2025 put 4.8% of genuine dev calls on "caution", within the rule's 5%).

**The five conditions.** The 31 `gen_train` speakers are left out of every number, the
reference's included, so the reference reads 12.3% and 4.2% here (12.5% and 4.4% with all
speakers).

| Condition | Bar | Current detector | v2 seed 1234 | v2 seed 2025 | Met |
|---|---|---|---|---|---|
| 1. Spontaneous genuine (HiACC) judged real | ≥ 97.3% | 99.3% | 99.8% | 99.7% | both |
| 2. Read genuine, new speakers / language control | ≥ 99.0% | 99.9% / 99.9% | 100.0% / 100.0% | 99.9% / 100.0% | both |
| 2. Genuine live calls / genuine replays | ≥ 11/12, ≥ 5/6 | 12/12, 6/6 | 12/12, 6/6 | 12/12, 6/6 | both |
| 3. IndicF5 / Chatterbox clones caught | ≥ 50% each | 12.3% / 4.2% | 89.3% / 93.8% | 90.2% / 93.5% | both |
| 4. Tortoise, new speakers | ≥ 70.0% | 80.0% | 94.1% | 98.5% | both |
| 4. Tortoise, CM04 | ≥ 73.7% | 83.7% | 91.9% | 98.2% | both |
| 4. RVC | ≥ 35.4% | 45.4% | 73.5% | 77.6% | both |
| 5. XTTS-v2 clones caught | ≥ 98.0% | 99.7% | 100.0% | 100.0% | both |

All on the simulated G.711 line, one 4 s head crop per clip, each model at its own point; the
calls are judged by their mean window P(genuine). Every condition, every row and the AMR-NB and
replay rows the rule does not judge are in `experiments/results/voice_check_v2_table.md`
(`python -m src.reporting.voice_check_v2_table`), which also shows the spontaneous recipe alone
(D1 part b, P-040) on the same clips.

**Correction to the plan above.** The plan calls RVC a tool v2 never heard. That is wrong: every
recipe here, the current detector's included, trains on RVC conversions from CM02's
speaker-held-out half (714 rows, P-025); the RVC test half shares no speaker with them. Condition
4 applied its 10-point limit to RVC exactly as written, and it was met, so the decision does not
change. Tortoise is the only tool here that no model heard.

### What it means

- **The two newest tools are now caught on speakers v2 never heard:** IndicF5 89-90% and
  Chatterbox 93-94% over the line, from 12% and 4%. They are seen tools for this model, so this
  is not a claim about unseen generators; the paper's unseen-tool numbers come from detectors
  that never heard them and do not change.
- **What the generator rows added, against the spontaneous recipe alone on the same clips:**
  IndicF5 34-43% to 89-90%, Chatterbox 52-53% to 93.5-93.8%, AMR-NB IndicF5 33-34% to 55-68%.
  Tortoise moved little (92.6/95.6% to 94.1/98.5% on new speakers, 92.8/95.9% to 91.9/98.2% on
  CM04).
- **The cost is RVC:** 85.8% and 89.0% with the recipe alone, 73.5% and 77.6% with v2, 11-12
  points lower in both seeds. It is still far above the current detector (45.4%), which is
  why condition 4 holds, but the generator rows traded RVC for IndicF5 and Chatterbox.
- **Genuine speech is not hurt:** read 99.9-100%, spontaneous 99.7-99.8%, all 12 live calls and
  all 6 genuine replays judged real, in both seeds.

### Reported, not judged

- **Replayed clones.** A Chatterbox clone played into a phone is still missed by every model,
  v2 included (0 of 6): v2 learned Chatterbox through the simulated line, not through a speaker
  and a microphone. XTTS-v2 replays: seed 1234 4 of 6 (current detector 5 of 6), seed 2025 6 of 6.
- **AMR-NB.** IndicF5 clones through the real mobile codec: 68.0% and 54.9% (current detector
  1.6%); genuine read speech 99.8-100%.
- **Clean wideband files** (the page's own scoring path, telephone front end, no phone line;
  `experiments/results/voice_check_v2_clean.md`, `python -m src.reporting.voice_check_v2_clean`).
  On the recordings both detectors scored, without the 31 training speakers: IndicF5 clones
  caught 72.7% of 55 (detector before: 0%), Chatterbox 86.1% of 72 (0%), RVC 58.4% (20.1%),
  Tortoise 86.8% (71.1%), XTTS-v2 100.0% (97.3%). Genuine speech is judged as before (76.4%
  against 76.5% of 1,455; the rest have under 2 s of speech, which the page declines to judge),
  and v2 calls one genuine recording cloned, a HiACC clip, where the detector before called none.
  Threshold-free, IndicF5 against genuine speech goes from 24.82% to 1.77% EER.
- **Uncertainty** (`experiments/results/voice_check_v2_intervals.md`,
  `python -m src.reporting.voice_check_v2_intervals`; 95% speaker-bootstrap intervals). IndicF5
  clones caught 89.3% [83.2, 94.2] and 90.2% [83.5, 95.5] against 12.3% [7.1, 18.1]; Chatterbox
  93.8% [87.5, 98.8] and 93.5% [87.1, 98.5] against 4.2% [1.6, 7.3]; the intervals do not
  overlap. RVC rests on 6 speakers (73.5% [62.1, 84.4] and 77.6% [65.5, 86.4] against 45.4%
  [29.0, 62.8]) and Tortoise on CM04 on 7. Twelve of 12 genuine live calls is consistent with a
  true rate as low as 73.5% (exact interval) and 6 of 6 replays with 54.1%: the calls show no
  failure, not a measured rate.

### Limits

- Two seeds decided the adoption; their operating points differ a lot (0.073 and 0.591)
  because their score scales differ, and each was chosen on dev only. Two more seeds (7 and 42,
  below) give the spread of the EERs but have no operating point.
- IndicF5 and Chatterbox are measured on held-out speakers of the same two tools; a new
  generator has not been tested.
- The 30 real calls are the team's own recordings, a small sample.
- Field recordings (people using the page) are still to come (review plan A4).

## Seeds 7 and 42 (registered 10 November 2026, before training)

Two seeds are a thin base for numbers that go into the paper. Voice check v2 is therefore
trained twice more, with seeds 7 and 42 (the project's next two seeds), from the same notebook
and the same pinned commit (`cb33e57`); only the seed changes
(`notebooks/kaggle_w15_vcv2_ssl_s{7,42}.ipynb`, team decision P-045). The two trainings run on
team members' Kaggle accounts beside the last generator-study runs.

These runs are **reported, not judged**:

- The adoption (P-043) is not reopened, whatever they show. The voice check keeps seed 1234 at
  0.073.
- No seed is dropped or replaced. All four are reported, each seed's value with the mean, the
  standard deviation and the range (`src/reporting/voice_check_v2_seeds.py`, written with this
  section).
- Every EER is on speakers outside `gen_train`, computed by the same code as the adoption:
  IndicF5 and Chatterbox clones, Tortoise (new speakers and CM04), RVC and XTTS-v2, on the G.711
  line and, where the sets exist, the AMR-NB codec. The genuine clips called fake (language
  control, HiACC, IndicVoices) are the notebook's own rates.
- If the two checkpoints are later scored on the development calls, the five conditions at each
  seed's dev-only point are reported as a check of whether the rule would have held for them.
  That cannot change the adoption either.

```
python -m src.reporting.voice_check_v2_seeds --results <the four run folders> \
    --out experiments/results/voice_check_v2_seeds.md
```

**Results (10 November).** Both trainings finished with every step clean, from the pinned
commit (`experiments/results/voice_check_v2_seeds.md`). EER (%) on speakers outside
`gen_train`; genuine clips called fake (%) at the notebook's fixed score.

| Measure | s1234 | s2025 | s7 | s42 | Mean ± SD | Range |
|---|---|---|---|---|---|---|
| IndicF5 clones, G.711 | 2.53 | 2.53 | 5.26 | 2.53 | 3.21 ± 1.37 | 2.53-5.26 |
| IndicF5 clones, AMR-NB | 3.22 | 8.98 | 6.94 | 4.07 | 5.80 ± 2.65 | 3.22-8.98 |
| Chatterbox clones, G.711 | 1.49 | 1.15 | 2.31 | 0.82 | 1.44 ± 0.64 | 0.82-2.31 |
| Tortoise, new speakers, G.711 | 0.03 | 0.17 | 1.62 | 0.14 | 0.49 ± 0.76 | 0.03-1.62 |
| Tortoise, CM04, G.711 | 0.57 | 0.43 | 1.36 | 0.50 | 0.71 ± 0.43 | 0.43-1.36 |
| RVC, held-out half, G.711 | 2.59 | 5.98 | 5.33 | 2.91 | 4.20 ± 1.70 | 2.59-5.98 |
| XTTS-v2, new speakers, G.711 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 ± 0.00 | 0.00-0.00 |
| Language-control genuine called fake | 0.08 | 0.00 | 0.23 | 0.08 | 0.10 ± 0.10 | 0.00-0.23 |
| HiACC spontaneous genuine called fake | 0.30 | 0.24 | 0.90 | 0.24 | 0.42 ± 0.32 | 0.24-0.90 |
| IndicVoices spontaneous genuine called fake | 0.29 | 0.39 | 0.88 | 1.07 | 0.66 ± 0.38 | 0.29-1.07 |

The AMR-NB rows for Tortoise and the XTTS-v2 evaluation pool are in the full table.

- **The two new seeds agree with the first two.** Over the G.711 line every clone EER is under
  6% in every seed, and at most about 1% of genuine speech is called fake. The weakest number
  in any seed is IndicF5 through AMR-NB (9.0%, seed 2025).
- **Seed 7 is the weakest training** on most rows: IndicF5 5.3%, Tortoise on new speakers 1.6%,
  HiACC genuine called fake 0.9%. It is reported as it came out; nothing is dropped.
- **The voice check's seed sits at the favourable end.** Seed 1234 was named before any
  training. Of the eight clone rows other than XTTS-v2 (where every seed is at or near 0), it
  has the lowest EER, alone or tied, on five: IndicF5 on both codecs, Tortoise new speakers on
  both codecs, and RVC. It is never the highest. The numbers the page reports at 0.073 are
  therefore on the good side of what this recipe gives; for the paper, the four-seed mean and
  range are the fair figure.
- **The adoption is not reopened**, as registered. The two new checkpoints were not scored on
  the development calls (only their result files were downloaded), so the five conditions at
  their own operating points are not reported.
