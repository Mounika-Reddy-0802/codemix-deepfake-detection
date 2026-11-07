# Voice check v2: the voice check's detector trained on the newest tools (v1)

**Status: registered, not run.** Written on 7 November 2026 before any training or scoring of
this model, so the adoption rule cannot be fitted to its results. Team decision P-042; plan
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

**Correction before any result (7 November, 09:50, before training started).** An independent
read of the code found that excluding the `gen_train` speakers crashed on the reference's HiACC
file, which has no speaker column. The exclusion is now applied to the four sets that hold any of
the 61 cloned speakers (new speakers, CM04, IndicF5, Chatterbox); HiACC, the language control,
RVC and the calls hold none of them. The conditions and thresholds are unchanged. The guard gate
`configs/train_lora_vcv2.yaml` is a LoRA config only so that `config_guard` checks the same
manifests, as `train_lora_spont.yaml` does for the spontaneous recipe; the model itself is
SSL-AASIST fine-tuned whole.

## Results

Not run yet.
