# W5-T4 — the shortcut gate asked of CM02: RVC leaves a spectral tell, not a level one

`lowlevel_cue_check_v1.md` ran AffectDF's Appendix-G gate on the corpus behind every
headline number and it **failed at 1.39% EER**, where chance is 50%. Two causes were
identified there: a real bundle bug (`portable_bundle.build` never level-normalises)
and a genuine recording-domain gap (NPTEL lecture audio against a vocoder).

This is the same gate asked of CM02, and CM02 can be asked a **much sharper version
of the question**.

## Why this design is stricter than the original check

The XTTS check compares MUCS bonafide against XTTS spoof: different utterances,
different speakers, only loosely matched. A separation there can be the generator,
or the corpus, or the level bug, and the check cannot tell you which.

Every RVC job records the exact source segment it converted, so here:

- **bonafide** = the 1,345 unique MUCS source segments the jobs read;
- **spoof** = the 1,500 RVC conversions **of those same segments**.

Same recordings, same words, same room, same microphone. The only thing that differs
between the classes is the conversion itself. Anything the gate finds is therefore
attributable to RVC and to nothing else.

**Fit and score are disjoint by source speaker** (13/12 of the 25 source speakers,
alternating over sorted ids so the split is reproducible). The identity that matters
for a low-level check is the recording channel, and the channel comes from the
source, not the target voice. Target voices are deliberately *mixed* across the
split — if each of the 12 voice models carried its own gain, mixing them makes the
shortcut easier to find, so a pass here would be conservative.

## Result

| Condition | Low-level-cue EER | AUC | Verdict |
|---|---:|---:|---|
| **raw** (as recorded / as generated) | **22.42%** | 0.8537 | ❌ FAIL |
| **normalised** (16 kHz, −23 dBFS both classes) | **22.28%** | 0.8599 | ❌ FAIL |
| *XTTS, clean (`lowlevel_cue_check_v1.md`)* | *1.39%* | *0.9990* | *FAIL* |
| *XTTS, clean + normalised* | *5.17%* | *0.9764* | *FAIL* |
| *AffectDF's own corpus* | *53.16%* | — | *pass (chance)* |

95% CI on the raw EER is [20.02%, 24.91%]; on the normalised, [19.72%, 24.40%].
1,369 scored clips (638 bonafide, 731 spoof), 0 files missing.

**This is a FAIL against the plan's near-chance requirement, and it is also the best
result this gate has returned on any of our data — by 16×.** Both statements are
true and neither should be dropped when this is quoted.

## The two findings that matter

### 1. Normalisation changes almost nothing (22.42% → 22.28%)

This is the opposite of the XTTS result, where normalising moved the gate from 1.39%
to 5.17% — a 3.7× improvement that identified the level bug as a real contributor.

Here normalisation buys **0.14 percentage points**. The per-class means show why:

| Feature | bonafide (MUCS source) | spoof (RVC) | ratio | after normalising |
|---|---:|---:|---:|---|
| rms_mean | 0.1625 | 0.1073 | 0.66 | **1.01 — equalised** |
| rms_std | 0.1677 | 0.1092 | 0.65 | **0.99 — equalised** |
| peak_amplitude | 0.9717 | 0.9198 | 0.95 | 1.35 — *flips* |
| **zero_crossing_rate** | **0.1324** | **0.0798** | **0.60** | **0.60 — unchanged** |
| **hf_energy_ratio** | **0.0047** | **0.0023** | **0.50** | **0.50 — unchanged** |
| spectral_rolloff_hz | 642.2 | 678.9 | 1.06 | 1.06 — unchanged |

Level normalisation does exactly what it should: RMS mean and standard deviation
equalise to within 1%. And the gate barely moves, because **the tell was never in the
levels**. The two features that carry it are scale-invariant, so normalisation cannot
touch them.

### 2. The tell is spectral: RVC's decoder loses high-frequency content

The fitted coefficients name the same feature in both conditions:

| Condition | Top three coefficients |
|---|---|
| raw | `zero_crossing_rate` **+2.648**, `dc_offset` −1.856, `clipping_ratio` +1.358 |
| normalised | `zero_crossing_rate` **+2.719**, `peak_amplitude` −1.140, `rms_std` +0.483 |

Zero-crossing rate dominates both, and it gets *stronger* once levels are equalised
and the level features stop competing with it. Converted clips have **40% lower
zero-crossing rate and half the high-frequency energy ratio** of the very same source
recordings.

That is a resynthesis signature. RVC's neural decoder reconstructs the waveform from
pitch and content features, and it does not reproduce the source's high-frequency
detail — the room noise, the breath, the fricative energy that a real microphone
captured. The conversion keeps the prosody (P-021: 96.4% f0-IQR retention) and loses
the top of the spectrum.

Note the raw condition's second and third features, `dc_offset` and `clipping_ratio`,
are near-degenerate: DC offset is −0.0022 against 0.0000 and the clipping ratio is
0.0001 against 0.0000. Those are artefacts of the generation path writing clean
zero-mean float audio, not properties of voice conversion, and they vanish under
normalisation — which is part of why the normalised run is the more honest of the two.

## What this means for the project

**It is evidence for P-021's argument, from a second direction.** P-021 retired
P-019's claim that XTTS flattens prosody, and replaced it with a cleaner statement:
the two generators deviate from real speech in *opposite* directions. This gate says
the same thing about spectral content — XTTS's corpus-level separation is dominated
by level and domain artefacts, CM02's residual is a bandwidth loss intrinsic to
conversion. **The two attack families are distinguishable from real speech by
different cheap statistics**, which is exactly why a detector trained only on CM01
has not seen CM02's behaviour.

**It does not clear the clean 1.34%.** That number is measured on the XTTS bundle and
this check does not touch it. `lowlevel_cue_check_v1.md`'s conclusion stands
unchanged: do not quote the clean gap-closure figure.

**It does say the CM02 corpus is the sounder of the two.** A detector scoring on CM02
has a 22.3% shortcut baseline to beat rather than a 1.39% one, so a good CM02 number
would mean considerably more than a good CM01 number does.

## What would move it further

1. **Band-limit before the gate.** Channel simulation moved the XTTS check from 1.39%
   to 9.25% by destroying high-frequency evidence. Since CM02's tell *is* high-frequency,
   the channel-matched condition should compress it much harder — worth measuring,
   and it is the deployment condition anyway.
2. **Do not treat 22.3% as a pass by comparison.** It is 27.7 points from chance. The
   honest framing is "much better than CM01, still not clean", not "acceptable".
3. **Report the shortcut baseline beside every CM02 model number**, the way
   `lowlevel_cue_check_v1.md` forced for CM01. A CM02 EER above ~22% is not evidence
   of detection.

## Reproduce

```bash
export DATA_ROOT=/c/dfdata
python -m src.data.rvc_gate \
  --clips  <dir holding the 1,500 rvc_*.wav> \
  --data-root "$DATA_ROOT"
```

Writes `experiments/results/lowlevel_cue_check_cm02_raw.json` and
`..._normalised.json`. The clips come from the private Kaggle archive
`saikrishnareddy9/rvc-cm02-generation-archive`; the job table
(`data/manifests/rvc_generation_jobs.csv`) is in the repo, so the source side is
cut from MUCS on the fly and nothing but the conversions needs fetching.
