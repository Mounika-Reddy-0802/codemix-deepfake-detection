# XTTS pilot — listening verdict (W3-T5)

The team rated the generated clips and reported these aggregate scores:

| Axis | Score (1–5) |
|---|---|
| Sounds human | **1.5** |
| Said the right words | **2** |
| Code-switch natural | **1** |

**Verdict: the XTTS-v2 clips do not meet a usable quality bar.** The weakest axis
is the one the project exists to measure — code-switch naturalness at 1/5.

## This agrees with the objective measurement

Ears and instruments reached the same conclusion independently. Intra-utterance
pitch range (f0 IQR):

| | f0 IQR |
|---|---|
| Real spontaneous speech (HiACC) | 42.2 Hz |
| Real read speech (MUCS, our reference voices) | 41.1 Hz |
| **XTTS-v2 output, every configuration tried** | **25–29 Hz** |

Five configurations were tested — longest reference, most-expressive reference,
temperature 0.95, expressive + hot, and five references totalling 49–115 s. None
closed the gap; raising temperature made it worse. XTTS invents prosody from text
and regresses to a flat contour, and the reference supplies timbre, not
intonation. Recorded as **P-019**.

## What this does and does not decide

**Decided — the quality bar.** The XTTS family is a *legitimate but easy* attack.
Detector numbers measured against it will be optimistic, and the f0 gap plus these
scores belong in the datasheet rather than being discovered at review.

**Decided — RVC is necessary, not optional.** Voice conversion starts from a real
human recording and swaps only timbre, so the pitch contour is genuine and this
flattening cannot occur. The second attack family is what makes the eval
meaningful; it stopped being a nice-to-have the moment code-switch naturalness
came back at 1/5.

**Not decided — romanised vs Devanagari.** The blind A/B compares the two arms
per clip per rater; three aggregate numbers cannot separate them. Given 1/5 on the
axis that matters, the likelihood is that both arms are weak rather than one
winning, so the script choice rests on P-014's readability argument rather than on
measured audio quality. If the per-clip sheet is filled in later,
`src.data.pilot_rating.summarise_ab` scores it against the withheld key.

## Consequence for the 4,000-clip corpus

The Week-4 XTTS corpus (3,998 usable clips, 6.91 h) stands as generated. It is a
real attack family and worth keeping — but it must be **reported as an easy one**,
and it should not be the only spoof source the detector is evaluated against.
