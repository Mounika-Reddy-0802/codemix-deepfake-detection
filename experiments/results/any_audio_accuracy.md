| Condition | deployed | final | ssl_s1234 | ssl_s1234_spont | ssl_s2025 |
|---|---|---|---|---|---|
| **Simulated G.711 line** | | | | | |
| Genuine, read lectures (60 new speakers) | 92.7 (n=164) | 99.7 (n=2400) | 99.6 (n=2400) | 99.9 (n=2400) | 99.9 (n=2400) |
| Genuine, 183 language-control speakers | - | 99.8 (n=1290) | 99.5 (n=1290) | 99.9 (n=1290) | 99.9 (n=1290) |
| Genuine, SPONTANEOUS (HiACC, 24 speakers) | 48.3 (n=1000) | 47.4 (n=1000) | 92.8 (n=1000) | 99.3 (n=1000) | - |
| XTTS-v2 clones (seen tool) | 77.2 (n=123) | 93.8 (n=1770) | 99.9 (n=1770) | 99.4 (n=1770) | 100.0 (n=1770) |
| Tortoise clones (unseen) | 7.7 (n=13) | 45.5 (n=222) | 94.6 (n=222) | 81.1 (n=222) | 93.7 (n=222) |
| Tortoise clones, CM04 (unseen) | 7.4 (n=68) | 38.9 (n=458) | 92.6 (n=458) | 76.9 (n=458) | 85.6 (n=458) |
| IndicF5 clones (unseen) | 17.9 (n=134) | 17.5 (n=526) | 33.1 (n=526) | 12.5 (n=526) | 29.7 (n=526) |
| Chatterbox clones (unseen) | 9.7 (n=134) | 12.7 (n=134) | 20.0 (n=525) | 4.4 (n=525) | 49.0 (n=525) |
| RVC voice conversion | 83.1 (n=154) | 68.1 (n=317) | 71.3 (n=317) | 45.4 (n=317) | 84.2 (n=317) |
| **Real AMR-NB mobile codec** | | | | | |
| Genuine, read lectures (60 new speakers) | - | 99.9 (n=2400) | 99.9 (n=2400) | 100.0 (n=2400) | 100.0 (n=2400) |
| XTTS-v2 clones | - | 94.2 (n=1770) | 99.8 (n=1770) | 98.5 (n=1770) | 99.9 (n=1770) |
| Tortoise clones | - | 56.8 (n=222) | 91.0 (n=222) | 71.2 (n=222) | 91.4 (n=222) |
| IndicF5 clones | - | 10.1 (n=526) | 11.8 (n=526) | 2.3 (n=526) | 5.7 (n=526) |
| **Real recordings (as captured on the receiving phone)** | | | | | |
| Genuine live call (12 calls) | 4/12 | 11/12 | 12/12 | 12/12 | - |
| Genuine recording replayed into the phone (6) | 0/6 | 6/6 | 6/6 | 6/6 | - |
| XTTS-v2 clone replayed into the phone (6) | 6/6 | 5/6 | 6/6 | 5/6 | - |
| Chatterbox clone replayed into the phone (6) | 6/6 | 0/6 | 0/6 | 0/6 | - |
| Genuine, read into a phone microphone, no network (6) | 5/6 | 6/6 | - | - | - |

Thresholds: deployed 0.500, final 0.687, ssl_s1234 0.580, ssl_s1234_spont 0.082, ssl_s2025 0.334 (each chosen on dev only; ssl_s1234_spont: the dev set includes the 20 iv_dev spontaneous speakers, decision D1 part a).
