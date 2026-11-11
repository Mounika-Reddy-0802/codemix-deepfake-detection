# Voice check v2: headline percentages with 95% intervals

Percent judged correctly at each model's frozen dev-only point, on the simulated G.711 line, the 31 `gen_train` speakers left out of the sets that hold them. Brackets: 95% speaker-bootstrap interval (10,000 resamples, seed 20261108); w: Wilson interval over clips (no speaker column; narrower than a speaker interval); e: exact interval over clips (no error to resample); c: exact interval over calls.

| Measure | ssl_s1234_spont | ssl_vcv2_s1234 | ssl_vcv2_s2025 |
|---|---|---|---|
| Spontaneous genuine (HiACC) judged real | 99.3 [98.6, 99.7] w | 99.8 [99.5, 100.0] | 99.7 [99.2, 100.0] |
| Read genuine, new speakers | 99.9 [99.7, 100.0] | 100.0 [99.8, 100.0] e | 99.9 [99.7, 100.0] |
| Read genuine, language control | 99.9 [99.7, 100.0] | 100.0 [99.7, 100.0] e | 100.0 [99.7, 100.0] e |
| IndicF5 clones caught | 12.3 [7.1, 18.1] | 89.3 [83.2, 94.2] | 90.2 [83.5, 95.5] |
| Chatterbox clones caught | 4.2 [1.6, 7.3] | 93.8 [87.5, 98.8] | 93.5 [87.1, 98.5] |
| Tortoise, new speakers | 80.0 [72.5, 86.9] | 94.1 [88.4, 98.5] | 98.5 [96.2, 100.0] |
| Tortoise, CM04 | 83.7 [76.7, 90.6] | 91.9 [88.0, 95.8] | 98.2 [96.4, 99.6] |
| RVC conversions caught | 45.4 [29.0, 62.8] | 73.5 [62.1, 84.4] | 77.6 [65.5, 86.4] |
| XTTS-v2 clones caught | 99.7 [99.4, 100.0] | 100.0 [99.7, 100.0] e | 100.0 [99.7, 100.0] e |
| Genuine live calls judged real | 100.0 [73.5, 100.0] c | 100.0 [73.5, 100.0] c | 100.0 [73.5, 100.0] c |
| Genuine replays | 100.0 [54.1, 100.0] c | 100.0 [54.1, 100.0] c | 100.0 [54.1, 100.0] c |

Speakers per set (last column): hiacc 24, read_new_speakers 37, indicf5 25, chatterbox 30, tortoise_new 37, tortoise_cm04 7, rvc 6.
