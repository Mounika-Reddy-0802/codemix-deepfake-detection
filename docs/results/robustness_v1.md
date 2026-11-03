# Robustness and evasion: post-processing before the phone line (v1)

Plan task A4. The question an examiner asks of any detector: what if the attacker
re-encodes the clone, adds noise, or changes its speed or pitch, and what does a genuine
caller's room do to the false alarms? **This section was written on 1 November 2026,
before any sweep clip was scored**, so the transforms and the way the results are read
are fixed in advance.

**Two plans, one run (record, 2 November).** A second, independent plan for this task was
committed on 2 November at 07:13 (`b1f3921`: 14 transforms on 1,344 clips, two models, a
paired speaker bootstrap and two reading rules), with its own code and Kaggle notebook. It
was never run. By the time it was committed every score below already existed (the final
adapter's on 1 November at 13:25, SSL-AASIST's on 1 November at 19:45), so its reading rules
are not used here: rules written after the scores cannot be told apart from rules chosen to
fit them. The run that was made is this one, 15 transforms on 1,122 clips with three models,
read by the rule above, fixed before its own scoring. The second plan's code, notebook and
clip list were removed so that the project keeps one implementation.

## Design (fixed before scoring)

**Clips.** A fixed sample (seed 1234) of the new-speaker test sets, all clean audio:
300 genuine clips, 300 XTTS-v2 clones (seen tool), all 222 Tortoise clones and 300
IndicF5 clones (both unseen), from 73 test speakers
(`data/manifests/robustness_sweep.csv`, `src/data/robustness.py`).

**Chain.** Each clip's first 6 s (the detectors score the first 4 s; 6 s leaves room for
the 0.95x speed change) -> one transform -> level-normalised to -23 dBFS -> the project's
G.711 line at 20 dB, the same chain as every telephone result. `double_g711` adds a
second line pass instead (a call forwarded over a second hop). Every random choice is
seeded per clip.

**The 15 transforms.**

| Transform | What it stands for |
|---|---|
| `none` | the reference: no post-processing |
| `mp3_32k`, `mp3_64k` | the clone shared as an MP3 file |
| `opus_12k` | a messaging-app voice note |
| `aac_64k` | a handset's call or voice recorder |
| `white_20db`, `white_10db` | background hiss (20 dB is mild, 10 dB is strong) |
| `babble_20db`, `babble_10db` | other people talking nearby (four talkers from the adaptation pool, never a test speaker) |
| `reverb_0.4s` | a reverberant room (0.4 s reverberation time) |
| `speed_0.95`, `speed_1.05` | the clone slowed or sped up by 5% (tempo and pitch together) |
| `pitch_-1`, `pitch_+1` | pitch shifted by one semitone, tempo unchanged |
| `double_g711` | a second telephone hop |

**Measures** (`src/reporting/robustness.py`), per model and transform: the EER of each
clone tool against the genuine clips under the same transform; false alarms (genuine
called fake) and misses per tool at the fixed clip rule, fake when P(genuine) < 0.5; and
each figure's change from `none`.

**How it will be read.** A transform *helps an attacker* if it raises misses or EER on the
clones; it *hurts genuine callers* if it raises false alarms. With 222-300 clips per group
a change of a few points is within sampling noise, so only changes of 5 points or more are
discussed, and every value is reported. Nothing is chosen on these results.

**Models.** The final adapter (seed 1234), scored on this laptop with the project's clip
scorer (`src/training/evaluate.py`); SSL-AASIST (both checkpoints) on Kaggle.

## Results: the final adapter (1 November 2026)

Scores: `experiments/robustness_v1/final_s1234/` (one file per transform, the project's clip
scorer); table: `experiments/results/robustness_v1_final.json`
(`python -m src.reporting.robustness`). The reference row reproduces the published
new-speaker results (Tortoise 8.6% here, 8.81% in `w13_runs_v1.md`; IndicF5 22.3%, 21.68%),
so the sweep's chain is the project's chain.

**Final adapter (seed 1234)** (%; change from no transform in brackets)

| Transform | False alarm | EER xtts_v2 | Miss xtts_v2 | EER tortoise | Miss tortoise | EER indicf5 | Miss indicf5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 0.3 | 0.7 | 5.3 | 8.6 | 56.3 | 22.3 | 84.3 |
| mp3_32k | 0.3 (+0.0) | 0.7 (+0.0) | 6.0 (+0.7) | 9.0 (+0.4) | 56.8 (+0.5) | 21.7 (-0.7) | 84.3 (+0.0) |
| mp3_64k | 0.3 (+0.0) | 0.7 (+0.0) | 5.7 (+0.3) | 8.6 (+0.0) | 57.2 (+0.9) | 22.3 (+0.0) | 84.3 (+0.0) |
| opus_12k | 0.3 (+0.0) | 1.0 (+0.3) | 6.0 (+0.7) | 10.3 (+1.7) | 54.5 (-1.8) | 23.3 (+1.0) | 81.3 (-3.0) |
| aac_64k | 0.3 (+0.0) | 0.3 (-0.3) | 4.7 (-0.7) | 9.0 (+0.4) | 56.8 (+0.5) | 23.0 (+0.7) | 84.3 (+0.0) |
| white_20db | 0.3 (+0.0) | 1.0 (+0.3) | 5.0 (-0.3) | 9.0 (+0.4) | 52.7 (-3.6) | 23.7 (+1.3) | 83.0 (-1.3) |
| white_10db | 1.0 (+0.7) | 2.7 (+2.0) | 6.0 (+0.7) | 10.3 (+1.7) | 41.0 (-15.3) | 30.0 (+7.7) | 75.7 (-8.7) |
| babble_20db | 0.3 (+0.0) | 0.7 (+0.0) | 4.0 (-1.3) | 10.7 (+2.1) | 55.0 (-1.4) | 26.7 (+4.3) | 84.7 (+0.3) |
| babble_10db | 1.7 (+1.3) | 6.3 (+5.7) | 22.0 (+16.7) | 19.0 (+10.3) | 71.2 (+14.9) | 32.0 (+9.7) | 85.7 (+1.3) |
| reverb_0.4s | 25.7 (+25.3) | 15.0 (+14.3) | 9.7 (+4.3) | 23.9 (+15.3) | 23.4 (-32.9) | 43.7 (+21.3) | 64.0 (-20.3) |
| speed_0.95 | 0.3 (+0.0) | 1.0 (+0.3) | 8.3 (+3.0) | 9.0 (+0.4) | 68.5 (+12.2) | 22.7 (+0.3) | 86.7 (+2.3) |
| speed_1.05 | 0.3 (+0.0) | 1.3 (+0.7) | 9.0 (+3.7) | 7.7 (-1.0) | 46.8 (-9.5) | 22.7 (+0.3) | 87.7 (+3.3) |
| pitch_-1 | 24.0 (+23.7) | 14.3 (+13.7) | 8.0 (+2.7) | 29.3 (+20.7) | 33.3 (-23.0) | 35.3 (+13.0) | 50.0 (-34.3) |
| pitch_+1 | 30.0 (+29.7) | 22.3 (+21.7) | 17.7 (+12.3) | 28.4 (+19.7) | 27.0 (-29.3) | 39.3 (+17.0) | 51.7 (-32.7) |
| double_g711 | 0.3 (+0.0) | 1.3 (+0.7) | 3.7 (-1.7) | 10.0 (+1.3) | 50.5 (-5.9) | 24.3 (+2.0) | 81.7 (-2.7) |

## Reading (changes of 5 points or more, as fixed above)

**Re-encoding and a second phone hop do not help an attacker.** MP3 at 32 and 64 kbit/s,
Opus at 12 kbit/s, AAC and a second G.711 hop change every EER by at most 2.0 points and the
false alarms not at all; miss rates move by up to 3 points, and the second hop even lowers
Tortoise's misses by 5.9 points (more caught). The phone line already band-limits and
re-codes the audio, so a lossy file before it hides nothing more. These are the post-processing steps a real attacker is most likely to apply.

**Loud babble helps an attacker.** With other people talking at 10 dB, XTTS-v2 EER rises
from 0.7% to 6.3% and its misses from 5.3% to 22.0%; Tortoise EER rises from 8.6% to 19.0%.
Strong white noise (10 dB) raises IndicF5's EER from 22.3% to 30.0% but lowers the misses
on both unseen tools (Tortoise by 15.3, IndicF5 by 8.7 points): noise pushes every score
towards "fake", genuine included (false alarms +0.7). At 20 dB babble and white noise change
nothing worth discussing (at most +4.3 points).

**Room reverberation hurts genuine callers.** A 0.4 s reverberant room takes false alarms
from 0.3% to 25.7% and raises every tool's EER by 14-21 points. This is the largest effect
in the sweep and the same failure as spontaneous speech: the detector has learned the
recording conditions of its genuine training audio (close-talking read lectures), and a
genuine caller in a different acoustic setting falls outside them.

**Pitch shifting is not an evasion route, but it raises false alarms.** A one-semitone shift
makes the clones easier to catch (Tortoise misses fall from 56.3% to 27-33%, IndicF5 from
84.3% to 50-52%) because the shift adds processing artefacts the detector flags; on genuine
speech the same artefacts raise false alarms by 24-30 points, which matters only if a
genuine caller's audio passes through a voice changer.

**A 5% speed change moves the scores, not the separation.** Tortoise misses rise by 12.2
points at 0.95x and fall by 9.5 at 1.05x while its EER moves by at most 1 point: the
operating point is sensitive to tempo, the ranking is not.

**For deployment.** The detector is robust to how an attacker would usually deliver a clone
(compressed files, extra phone hops), fragile to where a genuine caller might be (a
reverberant room: a quarter of genuine clips called fake), and easier to fool when a clone is
played under loud background talk. Reverberant and noisy genuine speech in training belongs
with spontaneous speech (plan task A2) as the next data to add.

## Results: SSL-AASIST (1 November 2026)

Both checkpoints (seeds 1234 and 2025) scored the same 16,830 rendered clips on Kaggle
(`notebooks/kaggle_w15_scoring_ssl.ipynb`, the project's SSL-AASIST clip scorer, log-odds of
genuine, fake when below 0). The average takes the mean of the two checkpoints' P(genuine)
per clip (`python -m src.inference.ensemble --clips`), fake when below 0.5. Scores:
`experiments/robustness_v1/ssl_{s1234,s2025,avg}/`; tables:
`experiments/results/robustness_v1_ssl_{s1234,s2025,avg}.json`. The reference rows agree with
the published five-seed results on the new speakers (IndicF5 over G.711: 15.7% and 14.0% here,
15.79 ± 2.61 in `calibration_v1.md`).

**SSL-AASIST seed 1234** (%; change from no transform in brackets)

| Transform | False alarm | EER xtts_v2 | Miss xtts_v2 | EER tortoise | Miss tortoise | EER indicf5 | Miss indicf5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 0.7 | 0.3 | 0.0 | 1.0 | 5.0 | 15.7 | 69.7 |
| mp3_32k | 0.7 (+0.0) | 0.3 (+0.0) | 0.0 (+0.0) | 1.0 (+0.0) | 5.0 (+0.0) | 15.7 (+0.0) | 67.7 (-2.0) |
| mp3_64k | 0.7 (+0.0) | 0.3 (+0.0) | 0.0 (+0.0) | 1.0 (+0.0) | 5.4 (+0.5) | 15.7 (+0.0) | 68.3 (-1.3) |
| opus_12k | 0.7 (+0.0) | 0.3 (+0.0) | 0.0 (+0.0) | 1.7 (+0.8) | 3.2 (-1.8) | 12.7 (-3.0) | 60.3 (-9.3) |
| aac_64k | 0.7 (+0.0) | 0.3 (+0.0) | 0.0 (+0.0) | 1.0 (+0.0) | 5.4 (+0.5) | 16.0 (+0.3) | 68.7 (-1.0) |
| white_20db | 0.3 (-0.3) | 0.3 (+0.0) | 0.0 (+0.0) | 1.0 (+0.0) | 5.9 (+0.9) | 14.7 (-1.0) | 67.3 (-2.3) |
| white_10db | 0.3 (-0.3) | 0.3 (+0.0) | 0.3 (+0.3) | 2.7 (+1.7) | 12.2 (+7.2) | 15.7 (+0.0) | 69.7 (+0.0) |
| babble_20db | 1.7 (+1.0) | 0.7 (+0.3) | 0.0 (+0.0) | 1.7 (+0.8) | 6.8 (+1.8) | 16.7 (+1.0) | 75.3 (+5.7) |
| babble_10db | 3.3 (+2.7) | 4.0 (+3.7) | 4.3 (+4.3) | 10.9 (+10.0) | 34.2 (+29.3) | 26.0 (+10.3) | 84.3 (+14.7) |
| reverb_0.4s | 35.7 (+35.0) | 13.3 (+13.0) | 5.7 (+5.7) | 26.1 (+25.1) | 17.6 (+12.6) | 43.7 (+28.0) | 58.7 (-11.0) |
| speed_0.95 | 0.3 (-0.3) | 0.3 (+0.0) | 0.0 (+0.0) | 1.3 (+0.4) | 8.6 (+3.6) | 13.3 (-2.3) | 67.3 (-2.3) |
| speed_1.05 | 0.3 (-0.3) | 0.3 (+0.0) | 0.0 (+0.0) | 0.6 (-0.4) | 4.1 (-0.9) | 16.0 (+0.3) | 73.7 (+4.0) |
| pitch_-1 | 99.0 (+98.3) | 9.7 (+9.3) | 0.0 (+0.0) | 37.4 (+36.4) | 2.7 (-2.3) | 38.3 (+22.7) | 1.0 (-68.7) |
| pitch_+1 | 98.7 (+98.0) | 16.3 (+16.0) | 0.0 (+0.0) | 45.0 (+44.1) | 1.4 (-3.6) | 50.3 (+34.7) | 1.7 (-68.0) |
| double_g711 | 0.3 (-0.3) | 0.3 (+0.0) | 0.0 (+0.0) | 1.0 (+0.0) | 6.3 (+1.4) | 14.0 (-1.7) | 66.7 (-3.0) |

**SSL-AASIST seed 2025** (%; change from no transform in brackets)

| Transform | False alarm | EER xtts_v2 | Miss xtts_v2 | EER tortoise | Miss tortoise | EER indicf5 | Miss indicf5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 0.7 | 0.0 | 0.0 | 0.6 | 1.8 | 14.0 | 61.3 |
| mp3_32k | 0.3 (-0.3) | 0.0 (+0.0) | 0.0 (+0.0) | 1.0 (+0.4) | 2.3 (+0.5) | 14.7 (+0.7) | 63.3 (+2.0) |
| mp3_64k | 0.7 (+0.0) | 0.0 (+0.0) | 0.0 (+0.0) | 0.6 (+0.0) | 2.3 (+0.5) | 13.7 (-0.3) | 61.0 (-0.3) |
| opus_12k | 1.0 (+0.3) | 0.0 (+0.0) | 0.0 (+0.0) | 1.0 (+0.4) | 0.5 (-1.4) | 14.7 (+0.7) | 55.7 (-5.7) |
| aac_64k | 0.7 (+0.0) | 0.0 (+0.0) | 0.0 (+0.0) | 1.0 (+0.4) | 1.8 (+0.0) | 13.3 (-0.7) | 61.3 (+0.0) |
| white_20db | 0.7 (+0.0) | 0.0 (+0.0) | 0.0 (+0.0) | 0.6 (+0.0) | 1.8 (+0.0) | 14.3 (+0.3) | 61.0 (-0.3) |
| white_10db | 1.3 (+0.7) | 0.0 (+0.0) | 0.0 (+0.0) | 1.3 (+0.8) | 4.1 (+2.3) | 15.0 (+1.0) | 60.3 (-1.0) |
| babble_20db | 1.3 (+0.7) | 0.0 (+0.0) | 0.0 (+0.0) | 1.3 (+0.8) | 2.3 (+0.5) | 17.0 (+3.0) | 63.7 (+2.3) |
| babble_10db | 5.7 (+5.0) | 1.0 (+1.0) | 0.0 (+0.0) | 6.3 (+5.8) | 6.8 (+5.0) | 20.0 (+6.0) | 68.0 (+6.7) |
| reverb_0.4s | 11.0 (+10.3) | 8.7 (+8.7) | 7.7 (+7.7) | 19.9 (+19.4) | 37.4 (+35.6) | 44.3 (+30.3) | 85.0 (+23.7) |
| speed_0.95 | 1.0 (+0.3) | 0.0 (+0.0) | 0.0 (+0.0) | 1.3 (+0.8) | 4.5 (+2.7) | 13.3 (-0.7) | 65.0 (+3.7) |
| speed_1.05 | 0.3 (-0.3) | 0.0 (+0.0) | 0.0 (+0.0) | 0.6 (+0.0) | 1.8 (+0.0) | 17.7 (+3.7) | 67.0 (+5.7) |
| pitch_-1 | 89.7 (+89.0) | 7.3 (+7.3) | 0.0 (+0.0) | 35.6 (+35.1) | 2.7 (+0.9) | 37.0 (+23.0) | 4.0 (-57.3) |
| pitch_+1 | 92.7 (+92.0) | 13.3 (+13.3) | 0.0 (+0.0) | 42.7 (+42.2) | 4.1 (+2.3) | 47.7 (+33.7) | 8.7 (-52.7) |
| double_g711 | 0.7 (+0.0) | 0.0 (+0.0) | 0.0 (+0.0) | 0.6 (+0.0) | 1.8 (+0.0) | 15.3 (+1.3) | 61.0 (-0.3) |

**Average of the two SSL-AASIST checkpoints** (%; change from no transform in brackets)

| Transform | False alarm | EER xtts_v2 | Miss xtts_v2 | EER tortoise | Miss tortoise | EER indicf5 | Miss indicf5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 0.3 | 0.0 | 0.0 | 1.0 | 3.2 | 13.0 | 70.0 |
| mp3_32k | 0.3 (+0.0) | 0.0 (+0.0) | 0.0 (+0.0) | 1.0 (+0.0) | 4.1 (+0.9) | 13.7 (+0.7) | 69.3 (-0.7) |
| mp3_64k | 0.3 (+0.0) | 0.0 (+0.0) | 0.0 (+0.0) | 1.0 (+0.0) | 3.2 (+0.0) | 12.7 (-0.3) | 70.0 (+0.0) |
| opus_12k | 0.7 (+0.3) | 0.0 (+0.0) | 0.0 (+0.0) | 1.0 (+0.0) | 2.7 (-0.5) | 13.7 (+0.7) | 61.0 (-9.0) |
| aac_64k | 0.3 (+0.0) | 0.0 (+0.0) | 0.0 (+0.0) | 1.0 (+0.0) | 3.6 (+0.5) | 12.7 (-0.3) | 69.0 (-1.0) |
| white_20db | 0.3 (+0.0) | 0.0 (+0.0) | 0.0 (+0.0) | 0.6 (-0.4) | 3.6 (+0.5) | 13.3 (+0.3) | 69.3 (-0.7) |
| white_10db | 0.3 (+0.0) | 0.3 (+0.3) | 0.0 (+0.0) | 1.3 (+0.4) | 8.1 (+5.0) | 14.7 (+1.7) | 68.7 (-1.3) |
| babble_20db | 1.7 (+1.3) | 0.0 (+0.0) | 0.0 (+0.0) | 1.7 (+0.8) | 3.6 (+0.5) | 16.0 (+3.0) | 74.3 (+4.3) |
| babble_10db | 3.0 (+2.7) | 2.7 (+2.7) | 1.3 (+1.3) | 7.3 (+6.3) | 25.2 (+22.1) | 20.0 (+7.0) | 82.0 (+12.0) |
| reverb_0.4s | 17.3 (+17.0) | 9.0 (+9.0) | 5.0 (+5.0) | 21.6 (+20.7) | 23.0 (+19.8) | 45.0 (+32.0) | 73.0 (+3.0) |
| speed_0.95 | 0.7 (+0.3) | 0.0 (+0.0) | 0.0 (+0.0) | 1.0 (+0.0) | 6.3 (+3.2) | 11.7 (-1.3) | 70.3 (+0.3) |
| speed_1.05 | 0.3 (+0.0) | 0.0 (+0.0) | 0.0 (+0.0) | 0.4 (-0.6) | 3.6 (+0.5) | 15.3 (+2.3) | 73.7 (+3.7) |
| pitch_-1 | 97.0 (+96.7) | 7.7 (+7.7) | 0.0 (+0.0) | 35.1 (+34.1) | 2.7 (-0.5) | 37.0 (+24.0) | 1.7 (-68.3) |
| pitch_+1 | 97.3 (+97.0) | 12.0 (+12.0) | 0.0 (+0.0) | 41.8 (+40.8) | 1.8 (-1.4) | 47.3 (+34.3) | 3.3 (-66.7) |
| double_g711 | 0.3 (+0.0) | 0.0 (+0.0) | 0.0 (+0.0) | 0.6 (-0.4) | 3.6 (+0.5) | 13.0 (+0.0) | 68.3 (-1.7) |

## Reading: SSL-AASIST (changes of 5 points or more, as fixed above)

**The same pattern as the adapter, from a much stronger start.** With no transform the larger
encoder separates Tortoise at 0.6-1.0% EER (adapter 8.6%) and IndicF5 at 13.0-15.7% (adapter
22.3%), and its EER is lower than the adapter's for every tool under every transform except
two: pitch shifting, and the reverberant room, where all four versions fail on the unseen tools
alike (Tortoise and IndicF5 EER 20-45%).

**Re-encoding and a second phone hop do not help an attacker.** MP3, AAC and a second G.711
hop move no EER or false-alarm rate by more than 1.7 points for any of the three. Opus at
12 kbit/s makes IndicF5 clones *easier* to catch (misses -9.3, -5.7 and -9.0 points).

**Loud babble helps an attacker, less than against the adapter.** At 10 dB, seed 1234's
Tortoise EER rises from 1.0% to 10.9% and its misses from 5.0% to 34.2%, IndicF5's EER by 10.3
and misses by 14.7 points. Seed 2025 moves less (Tortoise EER +5.8, IndicF5 EER +6.0, false
alarms +5.0). XTTS-v2 clones under babble are still caught (misses at most 4.3%; the adapter
22.0%).

**Room reverberation hurts genuine callers; how much depends on the training run.** A 0.4 s
room raises false alarms from 0.7% to 35.7% for seed 1234, to 11.0% for seed 2025 and to 17.3%
for the average (the adapter 25.7%), and every tool's EER by 9-32 points. The two checkpoints
differ here as they do on spontaneous speech (`ensemble_v1.md`), and in the same direction:
seed 2025 is worse on spontaneous callers and better in a reverberant room, seed 1234 the
reverse.

**Pitch shifting flags almost every genuine caller.** One semitone up or down makes SSL-AASIST
call 89.7-99.0% of genuine clips fake (the adapter 24-30%) and catch nearly every clone (misses
0-8.7%): it treats the shift itself as manipulation. Pitch shifting is therefore no way to evade
it, but a genuine caller whose audio passes through a voice changer would always be flagged.

**For deployment.** The larger encoder keeps its advantage under the attacks an attacker is
likely to use (compression, extra phone hops, mild noise). Its weaknesses are the same three
as the adapter's, all on the genuine side or in loud background talk: reverberant rooms, voice
changers, and 10 dB babble. Two checkpoints of the same recipe differ by more than 20 points of
false alarms in a reverberant room, so a single training run is not enough to report this.
