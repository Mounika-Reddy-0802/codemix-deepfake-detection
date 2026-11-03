# Voice check on audio not measured before

Front end chosen on the development clips only (balanced accuracy): g711 97.4 %, none 93.2 % -> **g711**.

% judged correctly at the frozen threshold (number of recordings in brackets).

| Set | Kind | g711 | none |
|---|---|---|---|
| dev (dev) | genuine | 97.3 (186) | 97.3 (186) |
| eval_pool | genuine | 90.7 (118) | 90.7 (118) |
| new_speakers | genuine | 87.8 (164) | 87.8 (164) |
| cm04 | genuine | 90.4 (114) | 90.4 (114) |
| cm07 | genuine | 81.0 (153) | 81.0 (153) |
| rvc_test | rvc | 20.1 (154) | 15.6 (154) |
| hiacc | genuine | 70.8 (1000) | 70.8 (1000) |
| dev (dev) | xtts_v2 | 97.5 (320) | 89.1 (320) |
| eval_pool | xtts_v2 | 97.8 (182) | 96.7 (182) |
| new_speakers | xtts_v2 | 94.3 (123) | 88.6 (123) |
| new_speakers | tortoise | 53.8 (13) | 30.8 (13) |
| cm04 | tortoise | 69.1 (68) | 64.7 (68) |
| cm07 | indicf5 | 0.0 (134) | 0.0 (134) |
| cm09 | chatterbox | 0.0 (134) | 0.0 (134) |
| rvc_test | genuine | 92.5 (146) | 92.5 (146) |

Equal error rate per set (threshold-free), %:

| Set | g711 | none |
|---|---|---|
| dev | 0.00 | 0.43 |
| eval_pool | 1.02 | 1.02 |
| new_speakers | 0.00 | 0.00 |
| cm04 | 0.00 | 1.71 |
| cm07 | 25.76 | 27.52 |
| rvc_test | 9.56 | 8.09 |
| hiacc | – | – |
| cm09 | – | – |
