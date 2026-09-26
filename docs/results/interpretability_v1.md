# Interpretability v1 — what the deployed detector responds to

**Owner M, Week 11.** The detector reads a raw waveform and has no hand-written
features, so "what is it classifying on?" can only be answered by changing one
property of the audio at a time and watching the score. Three probes, all in
`src/inference/interpretability.py` behind `configs/interpretability.yaml`:

```
python -m src.inference.interpretability --config configs/interpretability.yaml
```

Model: `lora_norm_rvc_channel` (the deployed S2 adapter). Clips: the five demo
calls, held-out eval-pool audio at G.711 @ 20 dB. Scoring mirrors the live path:
4 s windows every 2 s, silence gate at −50 dBFS, each window normalised to
−23 dBFS, mean over the first 8 speech windows. Numbers below are
`experiments/results/interpretability.json`.

> **Read P-033 first.** An earlier exploratory version of this table reported that
> time-stretching a clone defeated the detector. That was a bug in the scratch
> script, not a property of the model. The numbers here are the ones to quote.

## 1. Perturbations

Mean P(genuine) over 8 windows. The verdict threshold is 0.5.

| Perturbation | genuine | RVC source | XTTS clone | RVC clone | Tortoise |
|---|---:|---:|---:|---:|---:|
| original | 0.999 | 1.000 | 0.000 | 0.002 | 0.991 |
| volume × 0.1 | 0.999 | 1.000 | 0.000 | 0.002 | 0.992 |
| extra μ-law pass | 0.995 | 0.999 | 0.000 | 0.001 | 0.878 |
| white noise, 20 dB SNR | 0.977 | 0.991 | 0.001 | 0.056 | 0.975 |
| white noise, 10 dB SNR | 0.882 | 0.957 | 0.001 | 0.281 | 0.971 |
| STFT analyse and resynthesise | 0.999 | 1.000 | 0.000 | 0.002 | 0.992 |
| phase vocoder, rate 1.0 | 0.999 | 1.000 | 0.000 | 0.001 | 0.989 |
| **phase scrambled** | **0.000** | **0.002** | 0.000 | 0.045 | 0.502 |
| time reversed | 0.511 | 0.144 | 0.000 | 0.000 | 0.517 |
| shuffle 500 ms chunks | 1.000 | 0.999 | 0.000 | 0.000 | 0.998 |
| **shuffle 100 ms chunks** | **0.999** | **0.997** | **0.000** | **0.000** | 0.858 |
| shuffle 25 ms chunks | 0.525 | 0.936 | 0.000 | 0.000 | 0.994 |
| **shuffle 5 ms chunks** | **0.000** | **0.000** | 0.000 | 0.000 | 0.000 |
| keep only 0–1 kHz | 0.964 | 0.994 | 0.000 | 0.563 | 0.876 |
| keep only 1–2 kHz | 0.066 | 0.029 | 0.126 | 0.107 | 0.178 |
| keep only 2–4 kHz | 0.710 | 0.701 | 0.161 | 0.320 | 0.594 |
| remove 0.3–1 kHz | 0.999 | 0.998 | 0.452 | 0.121 | 0.999 |
| remove 2–4 kHz | 0.999 | 0.998 | 0.017 | 0.115 | 0.998 |
| slower × 1.4 | 0.484 | 0.939 | 0.101 | 0.014 | 0.843 |
| faster × 0.7 | 0.177 | 0.751 | 0.000 | 0.000 | 0.715 |
| pitch +3 semitones | 0.294 | 0.485 | 0.000 | 0.000 | 0.618 |
| pitch −3 semitones | 0.879 | 0.994 | 0.177 | 0.114 | 0.857 |

**What this says.**

1. **Not language, words or prosody.** Shuffling 100 ms chunks destroys word order,
   grammar, code-mixing and sentence melody. The genuine clip stays at 0.999 and the
   clones stay at 0.000. The verdict is unchanged by what is being said.
2. **The cue is short.** It survives 100 ms shuffling, weakens at 25 ms and is gone
   at 5 ms, where every clip reads as fake. At this speaker's pitch one vocal-cord
   cycle is about 10 ms, so 5 ms chunks cut inside individual pulses.
3. **The cue is in the phase, not the spectrogram.** Phase scrambling keeps the
   magnitude spectrogram exactly and randomises phase: genuine audio drops from
   0.999 to 0.000. A plain STFT round trip, which preserves phase, changes nothing.
4. **Robust to the deployment nuisances.** Level, an extra telephone codec pass and
   noise down to 10 dB SNR leave the verdicts intact.
5. **Band evidence.** Most of the "real" evidence and the XTTS tell sit below 1 kHz;
   removing 0.3–1 kHz lets the XTTS clone recover to 0.452. Removing 2–4 kHz moves
   the RVC clone from 0.002 to 0.115, matching P-022's finding that RVC loses
   high-frequency detail.
6. **Rhythm and pitch changes do not create false negatives.** Both clones stay
   below the threshold under every stretch and shift tested. They do lower the
   genuine score (0.484 slower, 0.177 faster, 0.294 pitch up), so these are a
   false-alarm risk on real speech, not an evasion route.

## 2. Where the decision forms

Held-out AUC between genuine and cloned windows, along the class-mean direction
fitted on the other 60%. 40 genuine and 40 cloned windows from four demo clips.

| Point in the encoder | AUC |
|---|---:|
| convolution output, 25 ms view, no context | 0.901 |
| transformer layer 1 | 0.794 |
| transformer layer 2 | 0.865 |
| transformer layer 3 | 1.000 |
| transformer layers 4–12 | 0.996–1.000 |

The convolutional stack, which never sees more than 25 ms and has no notion of a
word, already separates the classes at 0.901. Separation is complete by layer 3.
In wav2vec 2.0 the linguistic content appears in the middle and upper layers, so
the decision is essentially finished before any of that exists.

**Caveat:** 80 windows from four clips, so single-layer values carry sampling
noise. The shape (high at the convolution output, saturating early) is the claim,
not the individual numbers.

## 3. Language

Real and fake English (ASVspoof 2019 LA eval) through the Hinglish-adapted model.
Clips shorter than one 4 s window are skipped, which is why the scored counts are
well below the 200 drawn per class.

| English audio | clips scored | mean score | verdict correct |
|---|---:|---:|---:|
| real speakers | 53 | 1.000 | 100% |
| synthetic attacks | 33 | 0.155 | 84.8% |

A different language, a different accent and no Hindi at all, and real speech is
still read as real. The RVC pair makes the same point inside one language: the RVC
clone keeps its source speaker's exact words, timing and code-mixing, and the two
score 1.000 and 0.002.

## 4. S1 against S2 on identical audio

`configs/interpretability_s1.yaml` and `configs/interpretability_s2_en.yaml` run the
same probes, on the same five clips, for the English-only S1 and the deployed S2.
The English clips are 40 s "calls" built by concatenating ASVspoof eval clips from
one real speaker and from attacks A10 (neural TTS) and A17 (voice conversion).
S3 is not included: its checkpoints live in the private Kaggle dataset and are not
on the demo machine.

Mean P(genuine); the two English attack columns are single clips, so read them as
examples, not as error rates.

| Probe | S1 en real | S1 en A10 | S1 en A17 | S1 cm real | S1 cm clone | S2 en real | S2 en A10 | S2 en A17 | S2 cm real | S2 cm clone |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| original | 0.999 | 0.377 | 0.001 | **0.436** | **0.703** | 1.000 | 0.835 | 0.000 | 0.999 | 0.000 |
| shuffle 500 ms | 0.634 | 0.001 | 0.001 | 0.110 | 0.048 | 0.964 | 0.000 | 0.000 | 1.000 | 0.000 |
| shuffle 100 ms | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.000 | 0.000 | 0.000 | **1.000** | 0.000 |
| shuffle 25 ms | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.000 | 0.000 | 0.000 | 0.634 | 0.000 |
| phase scrambled | 0.001 | 0.001 | 0.001 | 0.001 | 0.009 | 0.000 | 0.000 | 0.000 | 0.004 | 0.000 |
| keep only 0-1 kHz | 0.080 | 0.001 | 0.001 | 0.001 | 0.001 | 0.373 | 0.000 | 0.000 | **0.964** | 0.000 |
| remove 2-4 kHz | 0.999 | 0.109 | 0.001 | 0.055 | 0.074 | 1.000 | 0.676 | 0.000 | 0.999 | 0.017 |

Held-out layer AUC on the English clips is the same shape for both: 0.95 at the
convolutional output, 0.93 and 0.97 at layers 1 and 2, saturating at 1.00 from
layer 3. Whatever each model uses, it is computed in the first three layers.

**1. The architecture does not change; the reference for "real" does.** S1 reads its
own domain correctly (0.999 real, A17 caught) and inverts on code-mixed audio: the
real Hinglish speaker scores 0.436 and the XTTS clone 0.703, so the clone looks more
genuine than the human. That is the below-0.5 AUC of the gap table in one pair of
clips.

**2. The two models key on different bands.** Keeping only 0-1 kHz, S1 loses its real
speaker (0.999 to 0.080) while S2 keeps the code-mixed one (0.999 to 0.964). S1's
evidence for genuine English sits largely above 1 kHz; S2's evidence for genuine
phone-line Hinglish is in the pitch-harmonic band that survives a narrowband codec.
That is what channel-matched adaptation moved.

**3. Correction to the "100 ms shuffle changes nothing" claim.** It holds for the
deployment condition, which is what section 1 measured: S2 on code-mixed phone-line
audio keeps the genuine verdict at 1.000 through a 100 ms shuffle and 0.634 at
25 ms. It does *not* hold on clean studio English, where both models drop the
genuine verdict to about 0.000 once chunks are 100 ms, though both survive 500 ms.
Clean studio audio has no codec or room noise to hide the discontinuities a shuffle
introduces, so the chopping itself reads as an artefact. The language conclusion
does not rest on this probe alone: an RVC conversion keeps its source speaker's
exact words and slang and scores 0.002 against 1.000, and real English scores 1.000
through the Hinglish-adapted model.

**4. Neither model is reliable on every attack family.** On these single clips S1
catches A17 (0.001) and is weak on A10 (0.377, on the correct side of the threshold
but barely); S2 catches A17 (0.000) and misses A10 (0.835). Per-attack behaviour
belongs in the error analysis, not in a claim about what the models "use".

## Reproduce

```
python -m src.inference.interpretability --config configs/interpretability.yaml
python -m src.inference.interpretability --config configs/interpretability_s1.yaml
python -m src.inference.interpretability --config configs/interpretability_s2_en.yaml
pytest tests/test_interpretability.py
```

The English probe clips are built by concatenating ASVspoof eval clips for one
speaker and one attack into 40 s calls; S1 and S2 must see byte-identical audio for
the comparison in section 4 to mean anything.

Three of those tests exist because of P-033: they fail if any perturbation mutates
the audio it is given.
