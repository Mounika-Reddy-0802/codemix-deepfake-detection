# Results on the week-13 sets: new speakers and a second unseen tool (v1)

EER (%) with a 95% speaker-bootstrap interval. New speakers: 60 speakers never
used before, 2,400 genuine clips. IndicF5: 54 test speakers, 651 all-Hindi genuine
clips. XTTS is the tool the S2 systems adapt on; Tortoise and IndicF5 are unseen by
every system. Built by `src/reporting/new_speakers.py` from
`experiments/results/ext_scoring_v1`.

| Model | Audio | new XTTS (CM05) | new Tortoise (CM06) | IndicF5 (CM07) |
|---|---|---|---|---|
| S1 (English only) | clean | 44.4 [40.5, 49.5] | 47.3 [43.2, 51.6] | 49.8 [45.4, 54.5] |
| S1 (English only) | phone line | 44.8 [40.8, 48.8] | 52.7 [48.7, 57.8] | 51.3 [46.1, 56.5] |
| S2 LoRA, XTTS, trained clean | clean | 7.2 [4.3, 10.0] | 13.5 [10.4, 17.5] | 37.0 [33.9, 41.4] |
| S2 LoRA, XTTS, trained clean | phone line | 25.2 [21.7, 29.5] | 29.7 [25.4, 33.6] | 49.8 [45.3, 53.5] |
| S2 LoRA, XTTS, trained phone line | clean | 24.8 [20.0, 29.5] | 31.1 [27.4, 35.7] | 47.5 [42.7, 52.8] |
| S2 LoRA, XTTS, trained phone line | phone line | 7.4 [5.5, 9.5] | 29.7 [25.6, 33.2] | 43.9 [39.9, 47.9] |
| S2 LoRA, XTTS+RVC, trained clean | clean | 10.2 [5.9, 14.1] | 26.1 [21.4, 30.0] | 34.4 [30.1, 38.6] |
| S2 LoRA, XTTS+RVC, trained clean | phone line | 27.4 [23.1, 31.3] | 33.8 [30.0, 37.9] | 44.9 [41.2, 49.2] |
| S2 LoRA, XTTS+RVC, trained phone line (deployed) | clean | 22.3 [17.1, 27.2] | 30.7 [26.5, 35.5] | 42.0 [37.7, 46.5] |
| S2 LoRA, XTTS+RVC, trained phone line (deployed) | phone line | 9.4 [6.6, 12.5] | 29.4 [25.4, 33.3] | 30.6 [26.7, 34.9] |

Per-speaker spread (each speaker's EER on their own clips):

| Model | Audio | Attack | Speakers | Median | Min | Max | Above 20% |
|---|---|---|---|---|---|---|---|
| S1 (English only) | clean | new XTTS (CM05) | 60 | 42.9 | 2.9 | 97.1 | 51 |
| S1 (English only) | clean | new Tortoise (CM06) | 60 | 50.0 | 2.5 | 100.0 | 59 |
| S1 (English only) | clean | IndicF5 (CM07) | 54 | 50.0 | 0.0 | 100.0 | 51 |
| S1 (English only) | phone line | new XTTS (CM05) | 60 | 42.9 | 10.0 | 90.0 | 50 |
| S1 (English only) | phone line | new Tortoise (CM06) | 60 | 50.0 | 1.2 | 100.0 | 59 |
| S1 (English only) | phone line | IndicF5 (CM07) | 54 | 50.0 | 10.0 | 86.9 | 50 |
| S2 LoRA, XTTS, trained clean | clean | new XTTS (CM05) | 60 | 0.0 | 0.0 | 31.8 | 1 |
| S2 LoRA, XTTS, trained clean | clean | new Tortoise (CM06) | 60 | 1.2 | 0.0 | 32.9 | 14 |
| S2 LoRA, XTTS, trained clean | clean | IndicF5 (CM07) | 54 | 37.4 | 0.0 | 65.8 | 43 |
| S2 LoRA, XTTS, trained clean | phone line | new XTTS (CM05) | 60 | 17.1 | 0.0 | 60.0 | 22 |
| S2 LoRA, XTTS, trained clean | phone line | new Tortoise (CM06) | 60 | 25.0 | 0.0 | 62.1 | 41 |
| S2 LoRA, XTTS, trained clean | phone line | IndicF5 (CM07) | 54 | 48.6 | 5.0 | 82.9 | 50 |
| S2 LoRA, XTTS, trained phone line | clean | new XTTS (CM05) | 60 | 13.2 | 0.0 | 70.0 | 23 |
| S2 LoRA, XTTS, trained phone line | clean | new Tortoise (CM06) | 60 | 25.0 | 0.0 | 96.2 | 45 |
| S2 LoRA, XTTS, trained phone line | clean | IndicF5 (CM07) | 54 | 50.0 | 0.0 | 100.0 | 47 |
| S2 LoRA, XTTS, trained phone line | phone line | new XTTS (CM05) | 60 | 2.9 | 0.0 | 20.0 | 0 |
| S2 LoRA, XTTS, trained phone line | phone line | new Tortoise (CM06) | 60 | 25.0 | 0.0 | 71.2 | 41 |
| S2 LoRA, XTTS, trained phone line | phone line | IndicF5 (CM07) | 54 | 40.0 | 0.0 | 78.9 | 48 |
| S2 LoRA, XTTS+RVC, trained clean | clean | new XTTS (CM05) | 60 | 0.0 | 0.0 | 32.9 | 2 |
| S2 LoRA, XTTS+RVC, trained clean | clean | new Tortoise (CM06) | 60 | 18.8 | 0.0 | 68.8 | 27 |
| S2 LoRA, XTTS+RVC, trained clean | clean | IndicF5 (CM07) | 54 | 27.5 | 0.0 | 100.0 | 32 |
| S2 LoRA, XTTS+RVC, trained clean | phone line | new XTTS (CM05) | 60 | 17.1 | 0.0 | 77.1 | 23 |
| S2 LoRA, XTTS+RVC, trained clean | phone line | new Tortoise (CM06) | 60 | 25.0 | 0.0 | 73.8 | 46 |
| S2 LoRA, XTTS+RVC, trained clean | phone line | IndicF5 (CM07) | 54 | 41.9 | 0.0 | 80.0 | 47 |
| S2 LoRA, XTTS+RVC, trained phone line (deployed) | clean | new XTTS (CM05) | 60 | 10.0 | 0.0 | 62.9 | 15 |
| S2 LoRA, XTTS+RVC, trained phone line (deployed) | clean | new Tortoise (CM06) | 60 | 25.0 | 0.0 | 97.5 | 40 |
| S2 LoRA, XTTS+RVC, trained phone line (deployed) | clean | IndicF5 (CM07) | 54 | 34.2 | 0.0 | 100.0 | 40 |
| S2 LoRA, XTTS+RVC, trained phone line (deployed) | phone line | new XTTS (CM05) | 60 | 2.9 | 0.0 | 17.1 | 0 |
| S2 LoRA, XTTS+RVC, trained phone line (deployed) | phone line | new Tortoise (CM06) | 60 | 25.0 | 0.0 | 73.8 | 42 |
| S2 LoRA, XTTS+RVC, trained phone line (deployed) | phone line | IndicF5 (CM07) | 54 | 27.1 | 0.0 | 65.2 | 31 |
