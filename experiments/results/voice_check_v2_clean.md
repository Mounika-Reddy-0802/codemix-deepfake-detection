# Voice check v2 on clean wideband files

Telephone front end, each detector at its frozen point; the 31 `gen_train` speakers left out of both; only recordings both runs scored. Percent judged correctly; recordings with under 2 s of speech count as not judged correctly.

| Clean test recordings | before 8 Nov (0.082) | v2 (0.073) |
|---|---|---|
| Genuine, read and HiACC spontaneous | 76.5 % of 1455 (24 % too short; 0 called cloned) | 76.4 % of 1455 (24 % too short; 1 called cloned) |
| XTTS-v2 clones | 97.3 % of 150 | 100.0 % of 150 |
| Tortoise clones (in no training set) | 71.1 % of 38 | 86.8 % of 38 |
| RVC conversions (held-out speakers) | 20.1 % of 154 (11 % too short) | 58.4 % of 154 (11 % too short) |
| IndicF5 clones (held-out speakers) | 0.0 % of 55 (13 % too short) | 72.7 % of 55 (13 % too short) |
| Chatterbox clones (held-out speakers) | 0.0 % of 72 (3 % too short) | 86.1 % of 72 (3 % too short) |

Per set (dev rows are the development clips):

| Set | Kind | before 8 Nov (0.082) | v2 (0.073) |
|---|---|---|---|
| dev | genuine | 97.3 % of 186 (3 % too short; 0 called cloned) | 96.8 % of 186 (3 % too short; 1 called cloned) |
| rvc_test | rvc | 20.1 % of 154 (11 % too short) | 58.4 % of 154 (11 % too short) |
| hiacc | genuine | 70.8 % of 1000 (29 % too short; 0 called cloned) | 70.7 % of 1000 (29 % too short; 1 called cloned) |
| dev | xtts_v2 | 97.5 % of 320 | 100.0 % of 320 |
| eval_pool | genuine | 95.1 % of 61 (5 % too short; 0 called cloned) | 95.1 % of 61 (5 % too short; 0 called cloned) |
| eval_pool | xtts_v2 | 100.0 % of 82 | 100.0 % of 82 |
| new_speakers | genuine | 85.3 % of 109 (15 % too short; 0 called cloned) | 85.3 % of 109 (15 % too short; 0 called cloned) |
| new_speakers | xtts_v2 | 94.1 % of 68 | 100.0 % of 68 |
| new_speakers | tortoise | 66.7 % of 6 | 83.3 % of 6 |
| cm04 | genuine | 92.6 % of 54 (7 % too short; 0 called cloned) | 92.6 % of 54 (7 % too short; 0 called cloned) |
| cm04 | tortoise | 71.9 % of 32 | 87.5 % of 32 |
| cm07 | genuine | 81.2 % of 85 (19 % too short; 0 called cloned) | 81.2 % of 85 (19 % too short; 0 called cloned) |
| cm07 | indicf5 | 0.0 % of 55 (13 % too short) | 72.7 % of 55 (13 % too short) |
| cm09 | chatterbox | 0.0 % of 72 (3 % too short) | 86.1 % of 72 (3 % too short) |
| rvc_test | genuine | 92.5 % of 146 (8 % too short; 0 called cloned) | 92.5 % of 146 (8 % too short; 0 called cloned) |

Equal error rate per set (threshold-free), %:

| Set | before 8 Nov (0.082) | v2 (0.073) |
|---|---|---|
| dev | 0.00 | 0.00 |
| rvc_test | 9.56 | 2.94 |
| hiacc | - | - |
| eval_pool | 0.00 | 0.00 |
| new_speakers | 0.00 | 0.00 |
| cm04 | 0.00 | 0.00 |
| cm07 | 24.82 | 1.77 |
| cm09 | - | - |
