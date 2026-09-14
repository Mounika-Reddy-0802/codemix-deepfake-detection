# Level-normalising the bundle, and what the shortcut gate says afterwards

**W10 retrain prep, owner M.** `lowlevel_cue_check_v1.md` showed eight cheap
per-clip statistics separate our real and fake clips at **1.39% EER**, with peak
amplitude the strongest cue, so the clean LoRA's 1.34% was not quotable. The first
bundles were never level-normalised. This fixes that, re-runs the gate, and
prepares the four adapters the results freeze needs.

## What changed

- `src/data/portable_bundle.py` now RMS-normalises every clip to **−23 dBFS** before
  writing it, so a rebuilt bundle cannot reintroduce the level cue.
- `src/data/normalise_bundle.py` normalises an existing bundle into a new root,
  keeping the same relative paths, so every manifest works unchanged under a new
  `--data-root`. It counts clips where the peak guard stopped the gain short of
  target (the clip would otherwise clip):

| Set | Clips | Peak-guard trips |
|---|---:|---:|
| Bundle, real | 2,626 | 8 (0.3%) |
| Bundle, fake (XTTS) | 4,000 | 91 (2.3%) |
| Attacks, real (RVC sources) | 1,345 | 3 (0.2%) |
| Attacks, fake (RVC + Tortoise) | 1,862 | 2 (0.1%) |

The G.711 @ 20 dB channel condition was re-rendered from the normalised root.

## The gate after normalisation

| Condition | Before | After normalisation | Verdict |
|---|---:|---:|---|
| Clean | 1.39% | **5.17%** (AUC 0.9764) | ❌ FAIL |
| Channel-matched (G.711 @ 20 dB) | 9.25% | **10.01%** (AUC 0.9568) | ❌ FAIL |

Chance is 50%. Normalisation did what it is for: peak amplitude falls from the
dominant cue (coefficient −4.48) to a secondary one (−2.57), and the clean gate
moves 3.7× away from a shortcut. **It does not pass.** The top clean cue is now
`zero_crossing_rate` (+4.66), which is not a level: it is the recording domain.
The real side is NPTEL lecture audio with room noise and fricative energy, and
the fake side is vocoder output that does not reproduce it. No gain change can
remove that.

The channel render compresses what is left. Band-limiting to telephone bandwidth
discards most of the high-frequency evidence, and the coefficients flatten (the
largest is peak amplitude at −2.28, next `hf_energy_ratio` +1.41).

## How the results must be read

1. **The channel-matched condition carries the headline.** It is the deployment
   condition (a phone call) and the one with the highest shortcut floor.
2. **Every model number is quoted beside its floor, not beside 50%.** A clean
   adapter at 1–2% is not evidence of learning when the floor is 5.17%. A channel
   adapter is evidence only by how far it gets below 10.01%.
3. **The unseen attacks are the strongest evidence.** CM04's floor is 22.21% and
   CM02's is 22.3% (P-022, P-025), so a low EER there cannot come from the cues
   measured here.

## What runs next

`notebooks/kaggle_w10_retrain_normalised.ipynb` trains four adapters on the
normalised data. All four are identical to the original LoRA config except for
their manifests:

| Adapter | Training data | Condition |
|---|---|---|
| `lora_norm_clean` | XTTS | clean |
| `lora_norm_channel` | XTTS | G.711 @ 20 dB |
| `lora_norm_rvc_clean` | XTTS + RVC training half | clean |
| `lora_norm_rvc_channel` | XTTS + RVC training half | G.711 @ 20 dB |

The RVC half comes from `src/data/rvc_holdout.py`. The 25 speakers CM02 touches are
split into two halves, and a conversion is kept only if its target and source
speakers are in the same half. Training gets 357 conversions and their 343 real
sources (13 speakers). Testing gets 317 conversions and 302 sources (12 speakers).
No speaker appears on both sides (`check_disjoint`). This is what lets RVC detection
stay measurable after the adapter has seen RVC (P-026).

Each adapter is scored in its own condition on the eval pool, on CM04 (Tortoise,
never trained on) and on the RVC test half. Kaggle input:
`saikrishnareddy9/codemix-bundle-normalised`, which has a `clean/` root and a
`channel/` root.

Sources: `experiments/results/bundle_normalisation_stats.json`,
`attack_normalisation_stats.json`, `lowlevel_cue_check_normalised_bundle.json`,
`lowlevel_cue_check_normalised_channel20.json`.
