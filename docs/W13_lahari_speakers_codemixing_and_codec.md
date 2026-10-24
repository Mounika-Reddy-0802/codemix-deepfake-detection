# Week 13 — Lahari — new speaker pools, is it the code-mixing, and a real codec

The paper said the gap it measures mixes language with recording domain and that the
design could not separate them. This week I built the test that does, carved the new
speaker pools the extension needed, and checked the simulated telephone line against a
real mobile codec.

## What I did, and why it is built this way

### Three new speaker pools, frozen before any audio was used

The study used 50 of MUCS's 520 speakers. `src/data/speaker_pools_ext.py` carves the
unused speakers with enough clips into three pools, committed and hashed before any of
their audio was scored or cloned: 60 new test speakers, 183 language-control speakers and
227 extra adaptation speakers. `config_guard` treats the first two as evaluation-only, so
no configuration can train on them. `src/data/extension_sets.py` builds their genuine
clip lists and the cloning job tables, including all-Hindi references and sentences for
IndicF5, which reads Devanagari.

### Is it the code-mixing? The language-controlled test

MUCS writes Hindi in Devanagari and English in Latin script, so every word's language can
be read from its script (`src/data/codemixing.py`). For each control speaker I took all
their all-English clips and as many all-Hindi and mixed (CMI at least 20) clips from the
same recordings: 1,290 clips, 183 speakers. Speaker, recording and channel are fixed; only
the language changes.

`src/reporting/language_control.py` reports false alarms per condition and the paired,
within-speaker difference (`docs/results/language_control_v1.md`). S1 rejects 84.3% of
genuine English, 85.2% of Hindi and 78.1% of mixed speech on clean audio; within speakers
mixed is rejected 9.5 points *less* often than English [-15.6, -3.5]. Over the phone line
the difference contains zero. The adapted models stay at 2-5% in every condition.

`src/reporting/cmi_bands.py` asks the same question on the evaluation pool clip by clip
(`docs/results/cmi_bands_v1.md`): S1 rejects 93.5-95.2% in every CMI band and CMI does not
predict its score. The gap follows the recording, not the language mix.

### A real mobile codec

The paper's phone line is a simulated G.711 chain. Mobile calls use AMR-NB.
`src/data/amr_channel.py` passes a manifest through the reference AMR-NB encoder in ffmpeg
(12.2 kbit/s, no added noise) and back, and `src/reporting/amr_channel.py` compares clean,
G.711 and AMR-NB (`docs/results/amr_channel_v1.md`). The deployed model loses most of its
channel gain on the real codec: new-speaker XTTS 9.4% to 23.7%, Tortoise 29.4% to 37.0%,
IndicF5 30.6% to 47.0%. The clean-trained adapter holds better (12.6% on XTTS), because
AMR-NB keeps the voice band and adds no noise. Channel training fitted one chain, not
telephone audio in general.

### How natural are the clones?

`src/reporting/naturalness.py` runs UTMOS on 200 clean clips per set
(`docs/results/naturalness_v1.md`). IndicF5 clones score 3.07, level with the same
speakers' genuine speech at 3.10; XTTS is half a point below its genuine speech and
Tortoise lower still. The newest unseen tool is also the most natural.

### The data chapter

`dataset.tex` now describes the new pools, the four new attack sets with their pass rates,
the extended training set, the language-controlled set, the AMR-NB check and the
naturalness, and its closing limits paragraph points to the language-control result
instead of saying the confound cannot be separated.

### Two codecs in training

The two-codec recipe (`src/data/codec_mix.py`, run 3) answered the AMR-NB loss: over three
seeds it reaches 7.59% on new-speaker XTTS through AMR-NB against 22.37% for the deployed
recipe, better in every seed (`docs/results/w13_runs_v1.md`). Mounika's final recipe
combines it with the 248 speakers.

### A third unseen tool: Chatterbox

`src/data/chatterbox_generation.py` clones the IndicF5 speakers' voices with Chatterbox
Multilingual (Resemble AI, MIT licence) from the same references and sentences as CM07, so
the two sets pair clip for clip. It is registered as held out, so `config_guard` and
`test_splits` refuse it in any training manifest. The Kaggle notebook failed twice before it
ran: first on the wrong code branch and a dataset path, then because an empty Hugging Face
token was sent as `Bearer ` and refused; the token is now dropped when blank, in the
loader itself, with the tests unchanged. 525 of 610 clips pass screening (86.1%; 78 of the
85 rejections stall at five characters a second or less). Chatterbox marks its output with
an imperceptible watermark; we keep the clips as generated and say so in the paper.

Scored on the laptop, it is the hardest tool: the English-only S1 sits at 55.0% EER over
the phone line, the deployed adapter at 35.8% and the final recipe at 32.0% (one seed).
The larger encoder, trained on the final data, reaches 12.9% over five seeds. The data
chapter lists CM09 with the other sets, and the results chapter has its own subsection.

### The shortcut floor for the final systems

The adapter (run 4) and the larger encoder (runs 7 and 8) both train on
`codemix_adapt_train_final.csv`, so one floor serves both. `src/training/floor_final.py`
fits the eight statistics of `floor_compare` on that set and scores them on exactly each
detector's clips, with the same speaker-level paired bootstrap, for every seed. The clips
sit in several bundle folders on the laptop, so each path is found by name, first folder
first, the rule the Kaggle runs used; features are cached clip by clip, which let the run
survive three out-of-memory stops and resume. Every seed of both systems beats the floor on
all eleven sets, unseen tools included: new-speaker Tortoise 8.8% for the adapter against a
floor of 36.5%, Chatterbox 12.9% for the larger encoder against 39.5%. On CM04 over either
codec and new-speaker Tortoise over AMR-NB the floor is above chance (54.6-71.4%), so the
report flags those rows as a weak bar (`docs/results/floor_final_v1.md`).

## What is still open

UTMOS was trained on English speech, so its absolute values on Hindi-English audio are
indicative; the comparison between sets of the same speakers is what the report claims.
The final recipe's adapter was scored on Chatterbox with one seed only (the other four
checkpoints stayed on Kaggle). Spontaneous speech (HiACC) has not been scored with the
final systems.
