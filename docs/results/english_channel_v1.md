# English over the phone line v1

Answers part of external review comment 4: *what does channel-matched Hinglish
adaptation cost on English, and what does that mean for a deployment?*

Code: `src/data/stratified_subset.py` (the sample), `src/reporting/english_channel.py`
(the report; tests: `tests/test_stratified_subset.py`, `tests/test_english_channel.py`).
Results: `experiments/results/english_channel_v1/` (`summary.json`, plus one evaluate
JSON and one score file per model and condition).

## What was measured

Until now every English number was on clean audio. The deployed model is meant for
phone calls, so the question is how it treats English *over the phone line*.

- **Sample.** Scoring all 71,237 ASVspoof 2019 LA evaluation clips through the
  phone-line chain on the CPU would take about five hours per model, so the test uses
  a fixed sample: all 13 attacks with 308 clips each and 2,000 genuine clips (6,004
  clips, all 67 speakers), drawn with seed 1234 by `stratified_subset`. The sample
  stands in well for the full set: the deployed model scores 2.30% EER on the clean
  sample and 2.42% on all 71,237 clips.
- **Audio.** Level-normalised to -23 dBFS (`src.data.portable_bundle`), then rendered
  through the same G.711 chain at 20 dB used for the Hinglish test sets
  (`src.data.channel_bundle`; self-check on 20 clips: energy above 4 kHz 0.726% ->
  0.0000%, measured SNR 17.8 dB).
- **Models.** The deployed S2 (XTTS + RVC, channel-matched) and the English-only S1,
  both scored on the CPU with the usual 4 s window.
- **Operating point.** The deployed one: a clip is called fake when P(genuine) < 0.5.
  Nothing was tuned on these clips.
- **Intervals.** EER: 200 resamples of clips, as in every other evaluate result.
  Genuine called fake: 1,000 resamples of whole speakers.

## Result

| Model | Condition | EER (%) | Genuine called fake at 0.5 (%) | Fakes caught at 0.5 (%) |
|---|---|---:|---:|---:|
| Deployed S2 | clean | 2.30 [1.84, 2.85] | 0.05 [0.00, 0.16] | 77.35 |
| Deployed S2 | phone line | **24.20** [23.27, 25.30] | **7.45** [4.27, 11.52] | **59.97** |
| S1 (English only) | clean | pending | pending | pending |
| S1 (English only) | phone line | 6.50 [5.92, 7.19] | 0.65 [0.30, 1.09] | 76.15 |

S1 on the clean sample is still to be scored: the run stopped twice when the laptop
ran out of memory. On all 71,237 clean clips S1 scores 0.85% EER.

EER per attack:

| Attack | Deployed S2, clean | Deployed S2, phone line | S1, phone line |
|---|---:|---:|---:|
| A07 | 0.65 | 37.94 | 3.90 |
| A08 | 0.31 | 32.46 | 12.99 |
| A09 | 0.07 | 14.01 | 3.59 |
| A10 | 12.63 | 47.71 | 10.71 |
| A11 | 0.34 | 30.85 | 4.81 |
| A12 | 0.31 | 26.59 | 4.50 |
| A13 | 0.00 | 3.90 | 0.03 |
| A14 | 0.34 | 19.08 | 7.77 |
| A15 | 4.21 | 33.11 | 12.27 |
| A16 | 0.31 | 15.95 | 1.61 |
| A17 | 2.26 | 8.11 | 2.94 |
| A18 | 6.50 | 17.85 | 9.38 |
| A19 | 0.34 | 3.51 | 1.95 |

## What it means

- **The phone line costs the deployed model most of its English ability.** EER goes
  from 2.30% to 24.20%. At the deployed threshold, one genuine English caller in 13
  is flagged (7.45%), and four English fakes in ten get through.
- **It is not the channel alone.** S1 has never heard phone-line audio, yet on the
  same clips it scores 6.50% and flags 0.65% of genuine callers. The adapted model is
  almost four times worse. Adaptation tied "genuine over the phone" to the MUCS
  Hinglish lectures it was trained on, which is the same pattern the HiACC test found
  for new Hinglish speakers (54% of their genuine speech called fake).
- **Nearly every attack is affected.** On clean audio the deployed model is under 1%
  on 9 of 13 attacks; over the phone line only A13 and A19 stay under 5%. S1 stays
  under 13% on every attack.

For a deployment this means one model cannot be assumed to serve both caller groups.
A service that takes English calls needs either English phone-line audio in the
adaptation data or a separate English path; the paper should say so rather than
quote the clean English number.

## Limitations of this test

- One channel (G.711 at 20 dB). Other codecs and noise levels are not covered.
- A 6,004-clip sample, not the full set; the clean check above suggests it is
  representative, but the phone-line numbers have not been checked against all
  71,237 clips.
- ASVspoof genuine speech is read English from 67 speakers; spontaneous English was
  not tested.

## Reproduce

```
python -m src.data.stratified_subset --manifest data/manifests/asvspoof_eval.csv --out data/manifests/asvspoof_eval_subset6k.csv
python -m src.data.portable_bundle --manifest data/manifests/asvspoof_eval_subset6k.csv --out-dir D:/dfdata/asv_subset_norm --manifest-out data/manifests/asvspoof_eval_subset6k_norm.csv
python -m src.data.channel_bundle --manifest data/manifests/asvspoof_eval_subset6k_norm.csv --out-dir D:/dfdata/asv_subset_norm_ch20 --manifest-out data/manifests/asvspoof_eval_subset6k_channel20.csv --data-root D:/dfdata/asv_subset_norm
python -m src.training.evaluate --checkpoint <model best.pt> --manifest data/manifests/asvspoof_eval_subset6k_channel20.csv --encoder wav2vec2-base --data-root D:/dfdata/asv_subset_norm_ch20 --device cpu --batch-size 16 --num-workers 2 --max-seconds 4.0 --scores-out experiments/results/english_channel_v1/<model>__english_channel_scores.csv --out experiments/results/english_channel_v1/<model>__english_channel.json
python -m src.reporting.english_channel
```
The first command rebuilds the sample byte-for-byte. Scoring takes about 15-25
minutes per model and condition on the laptop CPU.
