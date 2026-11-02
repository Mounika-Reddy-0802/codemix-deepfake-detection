# English routing with a language gate (v1)

Plan task A3. Over the G.711 line the final adapter reaches 25.95% EER on English and S1,
the English-only detector, 6.50% (`english_final_v1.md`). A deployment that also takes
English calls can put a language gate in front of the detectors: English speech goes to
S1, everything else to the Hindi-English detector. This document measures whether that
helps, and what it costs on Hindi-English speech.

Code: `src/inference/routing.py` (tests in `tests/test_routing.py`).

## The gate

A spoken-language identifier, SpeechBrain's VoxLingua107 ECAPA model (107 languages),
run on CPU (about 0.65 s per 4 s clip on this laptop with four threads, 680 MB of memory).
It reads exactly the audio the detectors score: the same loader and the same first 4 s
of each clip.

Three candidate rules, none with a parameter to tune, were written down before the
development results were read:

- `top_english`: S1 when the most likely language is English (the first rule);
- `english_over_hindi`: S1 when P(English) > P(Hindi);
- `not_indian`: the Hindi-English detector when the most likely language is one of the
  languages of India and its neighbours, S1 for everything else.

The selection criterion was fixed with them: the highest worst-case routing accuracy over
the development sets, a tie going to the earlier, simpler rule.

## Rule selection on development audio only (1 November 2026)

No test output existed when the rule was chosen (`experiments/results/routing_v1_dev.json`).

| Rule | English dev, clean (400) | English dev, G.711 (400) | Hindi-English dev, G.711 (506) | Worst case |
|---|---:|---:|---:|---:|
| top_english | 64.25 | 39.75 | 100.00 | 39.75 |
| **english_over_hindi** | 98.25 | 90.75 | 96.64 | **90.75** |
| not_indian | 98.50 | 93.75 | 90.71 | 90.71 |

Percent of clips routed to the right detector. English dev: a fixed balanced sample
(seed 1234) of the ASVspoof 2019 LA development partition, clean and through the
project's G.711 line at 20 dB. Hindi-English dev: the 506 development clips every
operating point in the project is chosen on.

**Chosen: `english_over_hindi`.** The first rule would have failed English over the
phone line: the identifier often hears the ASVspoof speakers' English as Welsh, Latin or
Norwegian (79 of 400 clean clips as Welsh alone), and the line makes it worse. Comparing
English with Hindi directly avoids that. The cost is that 3.36% of the Hindi-English dev
clips are sent to S1, which rejects Hindi-English speech.

## Test results

Each test set is a fixed sample (seed 1234, 1,000 clips, balanced genuine and fake; the
whole set when smaller), and the three systems are scored on exactly the same clips with
the scores every result document already uses (S1's English scores from
`english_channel_v1`, the final adapter's from the week-13 runs, seed 1234; S1 scored here
on the Hindi-English samples). Report: `experiments/results/routing_v1.json`.

Rule: `english_over_hindi`. EER (%), and false alarms / misses (%) at the fixed rule (fake when
P(genuine) < 0.5). Hindi-English sets over G.711.

| Test sample | Clips | Sent to S1 | EER: S1 | EER: final | EER: routed | False alarm: S1 / final / routed | Miss: S1 / final / routed |
|---|---:|---:|---:|---:|---:|---|---|
| English (ASVspoof sample), clean | 1000 | 99.4% | 0.60 | 13.50 | **0.60** | 0.2 / 0.2 / 0.2 | 5.8 / 71.8 / 6.0 |
| English (ASVspoof sample), G.711 | 1000 | 90.4% | 6.80 | 25.00 | **9.80** | 0.6 / 0.4 / 0.4 | 23.4 / 75.6 / 28.0 |
| Hindi-English original test pool (XTTS-v2) | 1000 | 2.2% | 60.40 | 1.20 | **1.60** | 46.6 / 0.8 / 0.2 | 74.6 / 2.2 / 5.4 |
| Hindi-English new speakers (XTTS-v2, Tortoise) | 1000 | 4.6% | 46.60 | 2.20 | **1.40** | 43.4 / 0.2 / 0.0 | 50.8 / 10.4 / 10.6 |
| CM04 Tortoise | 958 | 27.9% | 44.36 | 17.64 | **10.23** | 46.6 / 0.8 / 0.2 | 43.4 / 66.2 / 51.5 |
| IndicF5 | 1000 | 0.0% | 50.40 | 25.80 | **25.80** | 48.8 / 0.4 / 0.4 | 52.0 / 84.4 / 84.4 |
| RVC test half | 619 | 0.3% | 55.90 | 10.34 | **10.34** | 43.4 / 1.3 / 1.0 | 66.6 / 36.9 / 37.2 |
| Language control (genuine only) | 1000 | 0.4% | - | - | **-** | 44.6 / 0.2 / 0.3 | - / - / - |

## Reading

**English is recovered.** The gate sends 99.4% of clean and 90.4% of G.711 English to S1,
and the routed system's English EER is 0.60% clean (the final adapter alone: 13.50%) and
9.80% over the phone line (25.00% alone; S1 alone 6.80%, the gap being the 9.6% of G.711
English the gate keeps). Misses on English fakes fall from 71.8% to 6.0% (clean) and from
75.6% to 28.0% (G.711).

**Hindi-English costs almost nothing.** The gate sends 0-4.6% of the Hindi-English clips to
S1. On the original test pool the routed EER is 1.60% against 1.20%; on IndicF5 and RVC the
EER does not change (false alarms and misses move by at most 0.3 points). False alarms on genuine Hindi-English speech stay at
0.0-1.0%.

**English spoken by Hindi-English speakers stays with the right detector.** The
language-control set (genuine English sentences read by MUCS speakers) was the expected
hard case, because S1 calls 44.6% of it fake. The gate hears that English as closer to
Hindi and sends only 0.4% of it to S1, so routing raises its false alarms from 0.2% to
0.3% only.

**The Tortoise "gain" is a confound, not detection.** On CM04 the routed EER (10.23%) is
lower than either detector alone (17.64% and 44.36%), which routing alone cannot explain.
The gate sends 57.6% of the Tortoise clones to S1 but 0.6% of the genuine clips: Tortoise
is an English text-to-speech model, and its Hindi comes out English-accented. P(English)
by itself separates Tortoise from genuine speech at 7.2% EER on CM04 (23.2% on the new
speakers). That is an accent cue specific to this tool, not evidence of synthesis, and it
gives nothing against a Hindi-native generator (IndicF5: 0% routed). It is reported, not
counted as an improvement.

**What this licenses.** A language gate in front of the detectors fixes the English
weakness of the final recipe (`english_final_v1.md`) at a cost of at most 0.4 points of EER
on Hindi-English speech, on these samples. It does not improve detection of any unseen
tool. The gate adds about 0.65 s of CPU per 4 s window on this laptop (680 MB), so in the
live system it would run once per call on the first speech, not on every window.

## Limits

- One training of each detector (S1, the final adapter seed 1234); 1,000-clip samples.
- The English sets are ASVspoof (VCTK speakers): British and other non-Indian accents. An
  English caller with an Indian accent is the language-control case above, which the gate
  routes to the Hindi-English detector.
- The gate's own errors on the English dev audio (1.75-9.25%) cap what routing can
  recover over the phone line.
