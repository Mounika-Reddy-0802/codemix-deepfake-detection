| Condition | ssl_s1234_spont | ssl_spontrain_s1234 | ssl_spontrain_s2025 | ssl_vcv2_s1234 | ssl_vcv2_s2025 |
|---|---|---|---|---|---|
| **Simulated G.711 line** |  |  |  |  |  |
| Genuine, read lectures (60 new speakers) | 99.9 (n=1480) | 99.8 (n=1480) | 97.8 (n=1480) | 100.0 (n=1480) | 99.9 (n=1480) |
| Genuine, 183 language-control speakers | 99.9 (n=1290) | 99.8 (n=1290) | 98.7 (n=1290) | 100.0 (n=1290) | 100.0 (n=1290) |
| Genuine, SPONTANEOUS (HiACC, 24 speakers) | 99.3 (n=1000) | 99.5 (n=1000) | 99.1 (n=1000) | 99.8 (n=1000) | 99.7 (n=1000) |
| XTTS-v2 clones | 99.7 (n=1102) | 100.0 (n=1102) | 100.0 (n=1102) | 100.0 (n=1102) | 100.0 (n=1102) |
| Tortoise clones | 80.0 (n=135) | 92.6 (n=135) | 95.6 (n=135) | 94.1 (n=135) | 98.5 (n=135) |
| Tortoise clones, CM04 | 83.7 (n=221) | 92.8 (n=221) | 95.9 (n=221) | 91.9 (n=221) | 98.2 (n=221) |
| IndicF5 clones | 12.3 (n=244) | 33.6 (n=244) | 43.4 (n=244) | 89.3 (n=244) | 90.2 (n=244) |
| Chatterbox clones | 4.2 (n=260) | 52.3 (n=260) | 52.7 (n=260) | 93.8 (n=260) | 93.5 (n=260) |
| RVC voice conversion | 45.4 (n=317) | 85.8 (n=317) | 89.0 (n=317) | 73.5 (n=317) | 77.6 (n=317) |
| **Real AMR-NB mobile codec** |  |  |  |  |  |
| Genuine, read lectures (60 new speakers) | 99.9 (n=1480) | 99.4 (n=1480) | 98.4 (n=1480) | 100.0 (n=1480) | 99.8 (n=1480) |
| XTTS-v2 clones | 99.0 (n=1102) | 100.0 (n=1102) | 100.0 (n=1102) | 100.0 (n=1102) | 100.0 (n=1102) |
| Tortoise clones | 69.6 (n=135) | 94.1 (n=135) | 97.8 (n=135) | 94.8 (n=135) | 97.8 (n=135) |
| IndicF5 clones | 1.6 (n=244) | 33.2 (n=244) | 34.0 (n=244) | 68.0 (n=244) | 54.9 (n=244) |
| **Real recordings (as captured on the receiving phone)** |  |  |  |  |  |
| Genuine live call (12 calls) | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 |
| Genuine recording replayed into the phone (6) | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 |
| XTTS-v2 clone replayed into the phone (6) | 5/6 | 6/6 | 6/6 | 4/6 | 6/6 |
| Chatterbox clone replayed into the phone (6) | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 |

Percent of clips judged correctly (genuine as real, clones as cloned); the 31 `gen_train` speakers are left out of every column. Operating points, each chosen on dev only: ssl_s1234_spont 0.082, ssl_spontrain_s1234 0.035, ssl_spontrain_s2025 0.515, ssl_vcv2_s1234 0.073, ssl_vcv2_s2025 0.591. Heard in training (other speakers): XTTS-v2 and RVC in every column, IndicF5 and Chatterbox in v2 only; Tortoise in none.
