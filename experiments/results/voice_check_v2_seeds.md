# Voice check v2 over four seeds

EER (%) on speakers outside `gen_train` (31 left out), and genuine clips called fake (%) at the notebook's fixed score. Seeds 1234 and 2025 decided the adoption (P-043); 7 and 42 were added after it (P-045). Reported, not judged.

| Measure | s1234 | s2025 | s7 | s42 | mean ± SD | range |
|---|---|---|---|---|---|---|
| IndicF5 clones (CM07), G.711 | 2.53 | 2.53 | 5.26 | 2.53 | 3.21 ± 1.37 | 2.53-5.26 |
| IndicF5 clones (CM07), AMR-NB | 3.22 | 8.98 | 6.94 | 4.07 | 5.80 ± 2.65 | 3.22-8.98 |
| Chatterbox clones (CM09), G.711 | 1.49 | 1.15 | 2.31 | 0.82 | 1.44 ± 0.64 | 0.82-2.31 |
| Tortoise, new speakers, G.711 | 0.03 | 0.17 | 1.62 | 0.14 | 0.49 ± 0.76 | 0.03-1.62 |
| Tortoise, new speakers, AMR-NB | 0.07 | 0.14 | 0.78 | 0.10 | 0.27 ± 0.34 | 0.07-0.78 |
| Tortoise, CM04, G.711 | 0.57 | 0.43 | 1.36 | 0.50 | 0.71 ± 0.43 | 0.43-1.36 |
| Tortoise, CM04, AMR-NB | 0.93 | 0.57 | 1.36 | 0.50 | 0.84 ± 0.39 | 0.50-1.36 |
| RVC, held-out half, G.711 | 2.59 | 5.98 | 5.33 | 2.91 | 4.20 ± 1.70 | 2.59-5.98 |
| XTTS-v2, new speakers, G.711 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 ± 0.00 | 0.00-0.00 |
| XTTS-v2, evaluation pool, G.711 | 0.00 | 0.00 | 0.16 | 0.00 | 0.04 ± 0.08 | 0.00-0.16 |
| Language-control genuine called fake | 0.08 | 0.00 | 0.23 | 0.08 | 0.10 ± 0.10 | 0.00-0.23 |
| HiACC spontaneous genuine called fake | 0.30 | 0.24 | 0.90 | 0.24 | 0.42 ± 0.32 | 0.24-0.90 |
| IndicVoices spontaneous genuine called fake | 0.29 | 0.39 | 0.88 | 1.07 | 0.66 ± 0.38 | 0.29-1.07 |
