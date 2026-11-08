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
- **Clean wideband files** (the page's own scoring path, no phone line): being measured with
  `python -m src.inference.any_audio_eval`; added here when the run ends.

### Limits

- Two seeds; their operating points differ a lot (0.073 and 0.591) because their score scales
  differ, and each was chosen on dev only.
- IndicF5 and Chatterbox are measured on held-out speakers of the same two tools; a new
  generator has not been tested.
- The 30 real calls are the team's own recordings, a small sample.
- Field recordings (people using the page) are still to come (review plan A4).
