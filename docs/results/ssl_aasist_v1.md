# SSL-AASIST baseline (week 13, run 5)

SSL-AASIST (Tak et al., Odyssey 2022): AASIST back end on a wav2vec 2.0 XLS-R 300M front
end, ported to Hugging Face (`src/baselines/ssl_aasist_model.py`). Zero-shot = the published
checkpoint (ASVspoof 2019 LA). Fine-tuned = whole network, RawBoost, 5 epochs, lr 1e-5, batch 8,
best dev epoch kept, scored over the simulated G.711 line. "21 speakers" is the deployed
adapter's training set, "248 speakers" the extended set (G.711 only). Seeds: 1234, 2025, 7 on
21 speakers; 1234, 2025 on 248 speakers (the third 248-speaker seed did not fit in the run).
Raw output: `experiments/results/ssl_aasist_v1/`. EER (%) mean +/- sd; false alarms = genuine
clips called fake (%).

| System | New XTTS | New Tortoise | CM04 | CM07 (IndicF5) | RVC | False alarms |
|---|---|---|---|---|---|---|
| SSL-AASIST zero-shot | 22.66 | 32.88 | 29.05 | 53.02 | 44.10 | 73.10 |
| SSL-AASIST, 21 speakers (n=3) | 2.06 +/- 0.99 | 7.80 +/- 1.30 | 4.96 +/- 2.66 | 24.05 +/- 1.77 | 2.64 +/- 1.23 | 3.70 +/- 2.17 |
| LoRA adapter, 21 speakers (n=5) | 10.8 +/- 0.9 | 29.8 +/- 1.7 | 33.5 +/- 2.6 | 35.4 +/- 1.6 | 5.8 +/- 0.7 | 3.4 +/- 1.6 |
| SSL-AASIST, 248 speakers (n=2) | 0.06 +/- 0.08 | 0.50 +/- 0.06 | 1.10 +/- 0.31 | 16.48 +/- 0.96 | 4.85 +/- 0.23 | 0.23 +/- 0.22 |
| LoRA adapter, 248 speakers (n=5) | 0.9 +/- 0.3 | 13.6 +/- 3.4 | 24.5 +/- 3.5 | 27.3 +/- 0.7 | 6.5 +/- 1.9 | 0.5 +/- 0.4 |

The adapter rows are the matched G.711 runs of `docs/results/w13_runs_v1.md`. SSL-AASIST is
lower than the adapter on every set at both training sizes. It has about three times the
parameters of our wav2vec2-base encoder and is trained in full, not through low-rank
adapters. It was not run over the real AMR-NB codec and not on the spontaneous HiACC set,
and it has two or three seeds, so differences are indicative, not tested.
