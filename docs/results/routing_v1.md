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

