# False alarms on spontaneous speech from new speakers (HiACC adult) v1

Answers external review comment 3: *evaluate performance on additional unseen speakers
to assess generalisation across different voices and speaking styles.*

Code: `src/training/false_alarm.py` (tests: `tests/test_false_alarm.py`).
Results: `experiments/results/hiacc_false_alarm_v1/deployed_s2_channel.json` and
`deployed_s2_channel_scores.csv`.

## What was measured

Every code-mixed test set before this was MUCS lecture audio: read, prepared speech.
HiACC's adult partition is different on both counts: 24 speakers no model has heard,
speaking spontaneously. It holds genuine speech only, so it cannot give an EER; it
gives the number a deployment meets first with a new caller, the share of genuine
speech the detector calls fake.

- **Audio.** All 3,318 adult clips (3.2 h, 24 speakers), never the child partition.
  Built with the project's own tools, exactly like the test bundles: level-normalised
  to -23 dBFS (`src.data.portable_bundle`), then rendered through the G.711 chain at
  20 dB (`src.data.channel_bundle`; self-check: energy above 4 kHz 0.437% -> 0.0000%,
  measured SNR 18.7 dB).
- **Model.** The deployed S2 (XTTS + RVC, channel-matched), scored on the CPU.
- **Operating point.** The deployed one: a clip is called fake when P(genuine) < 0.5.
- **Reference.** The same model's genuine clips from the evaluation pool (MUCS read
  lectures, 15 speakers), so the effect is measured against something.
- **Interval.** 1,000 resamples of whole speakers (a speaker's clips are not
  independent).

## Result

| Genuine speech | Clips | Speakers | Called fake |
|---|---:|---:|---:|
| MUCS read lectures (evaluation pool) | 1,566 | 15 | **0.89%** [0.54, 1.29] |
| **HiACC spontaneous speech** | 3,318 | 24 | **54.37%** [44.43, 64.15] |
| Difference | | | **+53.48 points** [+43.51, +63.17] |

- **Every speaker is affected.** Per-speaker false alarms run from 13.9% to 89.5%
  (median 54.9%); all 24 speakers are above 10%. The median P(genuine) of a HiACC clip
  is 0.25.
- **Clip length does not explain it.** HiACC clips are short (median 3.0 s, 66% under
  the 4 s window), but longer clips are rejected *more*: 27.8% under 2 s, 60.0% at
  2-4 s, 72.9% at 4-6 s, 67.3% over 6 s. Short MUCS clips stay at 1.2% (250 clips
  under 4 s).

## What it means

The deployed detector learned what genuine *lecture recordings* sound like. Genuine
speech that sounds different -- other speakers, another recording set-up, spontaneous
delivery -- is rejected as fake more often than not. This is the failure the paper
already reports for S1 (English-trained, rejecting 94.8% of genuine Hindi-English),
now measured for our own adapted model: adaptation moved the detector's notion of
genuine speech to MUCS, not to Hindi-English speech in general.

For deployment this is the more damaging error, since every falsely flagged genuine
caller is a false accusation. It is a limitation the paper must state, alongside the
unseen-tool result.

## Limitations of this test

- **Style and recording are confounded.** HiACC differs from MUCS in speaking style
  *and* in recording conditions; this test cannot say which drives the false alarms.
- **Genuine speech only.** HiACC has no spoofs, so this is a false-alarm rate, not an
  EER; it says nothing about how well clones of these speakers would be caught.
- **One model.** Only the deployed S2 was scored. Whether the augmentation or the
  layer mix (`docs/external_review_response.md`, comment 2) lowers these false alarms
  is the obvious next test, once those models are trained.
- **No fix by training on HiACC.** HiACC is evaluation-only under the project's
  dataset rules, so it cannot be added to adaptation data; more varied genuine speech
  for adaptation would have to come from elsewhere.

## Reproduce

```
python -m src.data.portable_bundle --manifest data/manifests/hiacc_adult_eval_source.csv --out-dir C:/dfdata/hiacc_norm --manifest-out data/manifests/hiacc_adult_eval.csv
python -m src.data.channel_bundle --manifest data/manifests/hiacc_adult_eval.csv --out-dir C:/dfdata/hiacc_norm_ch20 --manifest-out data/manifests/hiacc_adult_eval_channel20.csv --data-root C:/dfdata/hiacc_norm
python -m src.training.false_alarm --checkpoint <lora_norm_rvc_channel best.pt> --manifest data/manifests/hiacc_adult_eval_channel20.csv --data-root C:/dfdata/hiacc_norm_ch20 --reference-scores results/w10/lora_norm_rvc_channel__eval_pool_scores.csv --device cpu --scores-out experiments/results/hiacc_false_alarm_v1/deployed_s2_channel_scores.csv --out experiments/results/hiacc_false_alarm_v1/deployed_s2_channel.json
```
About 15 minutes on the laptop CPU.
