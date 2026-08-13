# Channel-simulation verification — verdict (W3-T2)

The channel-matched protocol is a load-bearing claim: every evaluation clip is
passed through an 8 kHz telephony chain so the detector is tested in the
condition it is deployed in. This records both halves of the check.

## 1. Objective measurement — 20/20 pass

`python -m src.data.channel_qa` over the 20 rendered pairs
(`docs/qa/channel_sim_measurements.csv`):

| Measure | Result | Reading |
|---|---|---|
| Band-limited | **20 / 20** | narrowband stage genuinely happened |
| Energy above 4 kHz | **0.6% clean → 0.0001% channel** | the decisive test |
| Measured SNR | **17.6 dB** median (target 20) | noise stage active, close to target |
| Correlation with original | **0.9914** | same utterance, not damaged |
| Clipping / truncation | none | no artefacts from the chain itself |
| Failures | none | |

Both corpora are genuinely 16 kHz, so the near-total loss of energy above 4 kHz
is the downsample working rather than an already-narrowband source. A chain that
skipped the downsample and merely attenuated would have failed here — that case
is pinned by `tests/test_channel_qa.py`.

The 99%-energy bandwidth figures (3227 Hz clean, 2766 Hz channel) look close
together and are **not** the discriminator: speech energy concentrates below
4 kHz regardless, so that quantile barely moves. Energy above the cutoff is what
separates the two.

## 2. Listening test — passed

All three members listened to the 20 pairs at
`<DATA_ROOT>/processed/listening_test/` on 13 Aug 2026.

**Verdict: the channel output clearly sounds like a phone call, and speech stays
fully intelligible in both the Hindi and English halves, including across
code-switch points.** No blocking artefacts.

Recorded in `docs/qa/channel_sim_listening_sheet.csv`: telephony 4.0/5,
intelligibility 4.0/5, 60/60 rows rated.

> **Scope of the scores.** The team agreed one score per rater covering the set,
> rather than scoring each of the 20 clips independently. The sheet therefore
> supports the pass/fail claim above but should not be used to analyse
> per-clip variance. If a later section needs per-clip ratings, the set must be
> re-rated clip by clip.

## 3. Settings this verdict applies to

```yaml
codec: g711        # G.711 mu-law, 8 kHz narrowband
snr_db: 20.0
seed: 1234
```

Changing the codec or SNR invalidates this verdict — re-run both halves. In
particular **AMR-NB has not been verified**: `ffmpeg` is not installed on the
machine that ran this, so `channel_sim` silently falls back to G.711. Before any
AMR-NB condition is reported, install ffmpeg and repeat this check.

## 4. Conclusion

The channel-matched protocol is verified for G.711 at 20 dB SNR, by measurement
and by ear. Evaluation audio may be generated under these settings.
