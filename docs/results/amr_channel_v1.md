# Clean, simulated G.711 and real AMR-NB (v1)

EER (%) with a 95% speaker-bootstrap interval on the week-13 test sets. G.711 is the
simulated phone line used throughout the paper (8 kHz, mu-law, 20 dB noise); AMR-NB
is the reference 3GPP mobile codec at 12.2 kbit/s, no added noise. Built by
`src/reporting/amr_channel.py`.

| Model | Audio | new XTTS (CM05) | new Tortoise (CM06) | IndicF5 (CM07) |
|---|---|---|---|---|
| S1 (English only) | clean | 44.4 [40.5, 49.5] | 47.3 [43.2, 51.6] | 49.8 [45.4, 54.5] |
| S1 (English only) | G.711 | 44.8 [40.8, 48.8] | 52.7 [48.7, 57.8] | 51.3 [46.1, 56.5] |
| S1 (English only) | AMR-NB | 39.8 [35.4, 44.0] | 47.3 [43.7, 52.6] | 49.8 [46.1, 53.3] |
| S2 trained clean | clean | 10.2 [5.9, 14.1] | 26.1 [21.4, 30.0] | 34.4 [30.1, 38.6] |
| S2 trained clean | G.711 | 27.4 [23.1, 31.3] | 33.8 [30.0, 37.9] | 44.9 [41.2, 49.2] |
| S2 trained clean | AMR-NB | 12.6 [9.8, 15.0] | 35.6 [30.2, 39.8] | 39.8 [34.3, 44.7] |
| Deployed S2 (trained on G.711) | clean | 22.3 [17.1, 27.2] | 30.7 [26.5, 35.5] | 42.0 [37.7, 46.5] |
| Deployed S2 (trained on G.711) | G.711 | 9.4 [6.6, 12.5] | 29.4 [25.4, 33.3] | 30.6 [26.7, 34.9] |
| Deployed S2 (trained on G.711) | AMR-NB | 23.7 [19.6, 27.4] | 37.0 [32.0, 41.8] | 47.0 [41.4, 52.5] |
