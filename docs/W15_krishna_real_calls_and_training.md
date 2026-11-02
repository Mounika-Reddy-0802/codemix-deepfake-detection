# Week 15 — Krishna — the voice check in the repository, and the second SSL-AASIST checkpoint

The project's final output is a voice check: any recording in, "real" or "cloned" out. Until
this week its scoring code lived in a working folder, measured but never committed, and every
real-call result for the larger encoder rested on one training. This week put the scoring path
into the repository, scored the second SSL-AASIST checkpoint and the average of the two, and ran
the week's Kaggle training sessions from the laptop.

## What I did, and why it is built this way

### One scoring path for the page and for the numbers

`src/inference/any_audio.py` judges any file the same way: decode to 16 kHz mono, cut silent
ends (-45 dBFS), pass through an 8 kHz G.711 round trip (the detector was trained on
telephone-band audio only), score 4 s windows every 2 s at -23 dBFS, and compare the mean
P(genuine) over the windows with the frozen threshold. The page (Mounika's) and every accuracy
number go through the same `judge` function, so the demo cannot drift from what was measured.
`src/inference/any_audio_eval.py` measures audio the voice check has not met yet (clean
wideband copies, labelled field recordings); the one choice it makes, front end on or off, is
taken on the 506 development clips. `src/reporting/any_audio_accuracy.py` builds the accuracy
table from stored scores, every checkpoint at its dev-chosen threshold.

### The second checkpoint and the average

Seed 2025 was scored exactly as seed 1234 had been: the 506 dev clips through the live path,
its operating point chosen by the same rule (0.334, ladder two and four), then the 30 real
calls. Both checkpoints take every XTTS-v2 call to "likely cloned" and no genuine live call,
so that result no longer rests on one training. They disagree elsewhere: seed 2025 warns none
of the six Chatterbox calls (seed 1234: three) and 31 of 41 genuine spontaneous calls (seed
1234: 11). `src/inference/ensemble.py` averages the two checkpoints' P(genuine) window by window
and smooths the mean as the live path smooths one model; at its own dev-chosen point (0.540)
the average warns none of the 18 genuine calls (`docs/results/ensemble_v1.md`).

### Kaggle from the laptop

The week's GPU work (IndicVoices clones, the spontaneous recipe's adapter and SSL-AASIST runs)
was launched with `kaggle kernels push` and fetched back automatically. Two lessons: the quota
API prints the hours past whole days, which once made a 30 h quota read as 6 h; and two copies
of the queue script ran at once, so one session was started twice. The duplicate was left out
of every result, and one fetch-only watcher replaced both.

### CI

The front-end test needed `librosa`, which CI does not install; it now skips there, as the
other audio tests do.

## What is still open

The voice check is weak on the newest generators at its current point (IndicF5 12.5%,
Chatterbox 4.4% caught). The SSL-AASIST retrained with spontaneous speech catches far more of
them, but by the rule written before it was measured one of its two trainings keeps too little
read genuine speech, so the detector stays (P-040, `docs/voice_check.md`). The average of the two
frozen checkpoints is measured on the real calls but not yet in the voice check.
