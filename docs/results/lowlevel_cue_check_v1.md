# W5-T4 — the low-level-cue gate, and it FAILS

**Run late, and it should have come first.** The plan makes this a precondition:
*"must be near chance, else fix the pipeline before trusting anything."* It was
skipped, and S1, the gap matrix and S2 were all measured before anyone checked.
This is the result of finally running it.

## What the check is

AffectDF's Appendix G: fit a logistic regression on eight cheap signal statistics —
RMS mean, RMS std, peak amplitude, clipping ratio, DC offset, zero-crossing rate,
spectral rolloff, HF energy ratio — and see whether they alone separate bonafide from
spoof. On a sound corpus they cannot: **AffectDF reaches 53.16% EER**, i.e. chance.

If they *can*, the corpus has a shortcut. A detector scoring well on it may have
learned recording provenance rather than anything about synthetic speech, and every
model number measured on that data is suspect.

Fitted on the adaptation pool, scored on the speaker-disjoint eval pool. Regression
hand-rolled in numpy (`src/data/lowlevel_cue.py`) so CI exercises it.

## Result

| Condition | Low-level-cue EER | AUC | Verdict |
|---|---|---|---|
| **Clean** (the bundle every headline number uses) | **1.39%** | 0.9990 | ❌ FAIL |
| Clean + RMS-normalised to −23 dBFS | 5.17% | 0.9764 | ❌ FAIL |
| **Channel-matched** (G.711 @ 20 dB) | **9.25%** | 0.9674 | ❌ FAIL |
| *AffectDF's own corpus, for reference* | *53.16%* | — | *pass* |

Eight numbers per clip separate our classes at **1.39% EER**. Chance is 50%.

## What this does to each result we have

Set the low-level baseline beside the model it is supposed to validate:

| | Model EER | Low-level EER | Model's margin |
|---|---|---|---|
| S2 LoRA, clean | **1.34%** | **1.39%** | **none** |
| S2 LoRA, channel-matched | **3.89%** | 9.25% | 2.4× better |
| S1 baseline, clean | 53.71% | 1.39% | far worse than the shortcut |

**The clean 1.34% is not defensible.** A logistic regression on eight statistics
matches it. Nothing in that number requires the model to have learned anything about
speech, and it must not be reported as a gap-closure result until this is fixed.

**The channel-matched 3.89% partially survives.** There the model beats the
low-level baseline by 2.4×, so it is using something those statistics do not capture.
It still sits on a corpus that fails the gate, so it is weakened, not clean.

**S1's failure is now more interesting, not less.** The unadapted baseline scores
53.71% on data where a trivial regression gets 1.39%. It did not merely fail to
detect Hinglish deepfakes — it failed to exploit a shortcut sitting in plain sight,
which is consistent with `gap_matrix_v1.md`'s finding that it collapses both classes
toward "spoof" indiscriminately.

## Root cause

Per-class feature means over the 3,966 eval clips:

| Feature | bonafide (MUCS) | spoof (XTTS) | Distribution overlap |
|---|---|---|---|
| **peak_amplitude** | 0.9397 | **0.9968** | 0.56 |
| **zero_crossing_rate** | 0.1656 | **0.0922** | 0.38 |
| **rms_std** | 0.1498 | 0.0981 | 0.34 |
| rms_mean | 0.1437 | 0.0963 | 0.47 |
| hf_energy_ratio | 0.0072 | 0.0035 | 0.98 |

No single feature is decisive — the overlaps run 0.34 to 0.98 — but combined they
are. Two things drive it:

**1. A real pipeline bug: the bundles were never level-normalised.**
`src/data/preprocess.py:98` applies `rms_normalize(x, -23.0)`, and
`configs/data/channel_sim.yaml` specifies `target_dbfs: -23.0`. But
`portable_bundle.build` — which produced `lora_bundle` and `colab_bundle`, the audio
behind *every* number this project has reported — goes straight from `load_wav` to
`save_wav` and applies no normalisation at all. XTTS output arrives effectively
peak-normalised (0.9968, tightly clustered); raw MUCS spans do not (0.9397). That
alone is a label.

Fixing it helps and is clearly correct, but it is **not sufficient**: normalising to
−23 dBFS moves the check only from 1.39% to 5.17%.

**2. The classes come from genuinely different recording domains.** The residual
separation lives in zero-crossing rate (MUCS 1.8× higher) and RMS variability —
NPTEL lecture audio against a vocoder, which is the exact confound
`gap_matrix_v1.md` already flags. This is not a normalisation bug; it is the corpus
construction. Channel simulation compresses it (1.39% → 9.25%) because band-limiting
and noise destroy much of the evidence, which is precisely the argument for the
channel-matched protocol being the primary one.

## After the fix

`portable_bundle.build` now RMS-normalises to −23 dBFS, matching `preprocess.py`
and `channel_sim.yaml`. Bundles rebuilt, gate re-run, S2 retrained from scratch on
the normalised audio. Level separation collapsed from **3.06 dB to 0.08 dB**.

| | Shortcut EER | S2 EER | S2's margin |
|---|---|---|---|
| Clean, before the fix | 1.39% | 1.34% | **none** |
| **Clean, after the fix** | **5.17%** | **1.34%** | **3.9×** |
| Channel-matched | 9.25% | 3.89% | 2.4× |

Two things follow, and they point in opposite directions.

**S2 was not using the loudness artifact.** Retrained on normalised audio it scores
**1.34%** — identical. The *original* checkpoint, trained on unnormalised audio,
scores **1.16%** on the normalised eval set, slightly better than on the data it was
trained for. Removing the cue the regression leaned on hardest (peak amplitude was
its largest coefficient at −4.48) costs the model nothing. Whatever S2 learned, it
was not level.

**The gate still fails.** 5.17% is not chance. The dominant coefficient is now
zero-crossing rate (+4.66), and MUCS runs 1.8× higher on it than XTTS. That is
lecture audio against a vocoder — corpus construction, not a bug, and no gain change
touches it.

So the clean number is no longer indistinguishable from a shortcut: it went from a
0× margin to 3.9×. It is defensible with the caveat stated, rather than
indefensible. It is still measured on a corpus that a trivial classifier can beat
50/50 odds on by a wide margin.

## What has to happen

1. ~~Normalise in `portable_bundle.build`.~~ **Done** — see "After the fix".
   Necessary, and confirmed not sufficient.
2. **Make channel-matched the primary protocol**, not a secondary column. It is the
   condition where the model demonstrably beats the shortcut, and it is the
   deployment condition.
3. **Do not quote the clean 1.34%** in the paper, a review, or the report until the
   gate passes. `gap_closure_v1.md` now carries this warning.
4. **Consider RawBoost / augmentation on the training data**, which the plan already
   lists as the mitigation for shortcut learning (§6, risk 3).
5. **Re-examine `gap_matrix_v1.md`.** Its 44.65% code-mixed column is measured on
   `colab_bundle`, built by the same unnormalised path. The S1 numbers are less
   affected — S1 does not appear to use the shortcut — but the column should be
   re-derived once bundles are rebuilt.

## Reproduce

```powershell
$env:DATA_ROOT = "D:\dfdata"
.venv\Scripts\python.exe -m src.data.lowlevel_cue `
  --train data\manifests\codemix_adapt_train.csv --train-root "D:\dfdata\lora_bundle" `
  --test  data\manifests\codemix_eval.csv        --test-root  "D:\dfdata\lora_bundle" `
  --out experiments\results\lowlevel_cue_check.json
```
