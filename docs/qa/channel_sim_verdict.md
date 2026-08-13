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

## 3. AMR-NB — objective measurement 20/20 pass (13 Aug 2026)

`ffmpeg` 9.0 is now installed (`libopencore_amrnb`), so the AMR-NB branch of
`channel_sim` runs for real instead of falling back to G.711. Rendered at
`<DATA_ROOT>/processed/listening_test_amr/`; measurements in
`docs/qa/channel_sim_measurements_amr.csv`.

| Measure | G.711 | AMR-NB @ 7.40k | Reading |
|---|---|---|---|
| Band-limited | 20 / 20 | **20 / 20** | both genuinely narrowband |
| Max energy above 4 kHz | 0.0001% | **0.0000%** | AMR-NB is the harder low-pass |
| Measured SNR (aligned) | 17.62 dB | **5.57 dB** | AMR-NB is far lossier — the point of having it |
| Correlation (aligned) | 0.9914 | **0.8848** | parametric codec, waveform not preserved |
| Codec delay | 0 samples | **76–80 samples ≈ 4.8 ms** | encoder frame + lookahead |
| Failures | none | **none** | |

**AMR-NB is a genuinely harsher condition than G.711**, not a cosmetic variant:
12 dB less SNR and visibly lower waveform correlation at the same nominal 20 dB
noise setting. That makes it worth reporting as a separate column rather than
folding into one "telephony" number.

> **The first AMR-NB run failed 15 of 20 pairs, and the chain was not at fault.**
> AMR-NB delays its output by ~4.8 ms; `channel_qa` compared the pair at sample
> offset 0, which G.711 happens to satisfy exactly. A 4.8 ms shift drives
> correlation to ~0 and SNR negative, so a working codec measured as broken.
> `estimate_lag`/`align` now find the codec delay by cross-correlation and measure
> on the aligned overlap. G.711 re-measures identically (lag 0 on all 20 pairs,
> SNR 17.62, correlation 0.9914), so the fix costs nothing where there is no
> delay. Pinned by `tests/test_channel_qa.py`, including a test that alignment
> never rescues genuinely unrelated audio.

**Still outstanding: nobody has listened to the AMR-NB pairs.** The objective half
passes; the perceptual half needs the same three-rater pass G.711 got. Sheet ready
at `docs/qa/channel_sim_listening_sheet_amr.csv`.

## 4. Settings this verdict applies to

```yaml
codec: g711        # verified by measurement AND by ear
snr_db: 20.0
seed: 1234
```

```yaml
codec: amr_nb      # verified by measurement only -- listening pass outstanding
snr_db: 20.0
seed: 1234
```

Changing the SNR invalidates both verdicts — re-run the halves that apply.

## 5. Conclusion

The channel-matched protocol is verified for **G.711 at 20 dB SNR by measurement
and by ear**, and for **AMR-NB at 20 dB SNR by measurement only**. Evaluation
audio may be generated under the G.711 settings now; an AMR-NB column may be
reported once the listening pass is done.
