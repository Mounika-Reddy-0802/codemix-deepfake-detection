# Week 14 — Mounika — the larger encoder on spontaneous speech, and its speed

Week 13 left two questions about SSL-AASIST, the larger encoder that beat our adapter on
eight of eleven sets: does it still reject spontaneous speech, and can it run on a live call?
Its week-13 models were not kept, so this week trained it again and answered both.

## What I did, and why it is built this way

`notebooks/kaggle_w14_encoder_spontaneous.ipynb` (run 9) is the week-13 run-7 notebook with
three changes: two seeds (1234, 2025), HiACC's 3,318 genuine spontaneous clips added as a
genuine-only scoring set (uploaded as a private Kaggle dataset, adult speakers only, never
in training), and a timing step that measures one 4 s window on the GPU and the CPU for
SSL-AASIST and for wav2vec2-base, the adapter's encoder. The notebook was simulated end to
end (normal run, failed training, failed scoring, failed timing, the time limit) before it
was handed over, and every step ran without error on Kaggle.

## What it found

**Speed.** SSL-AASIST has 316 million parameters against 94 million. One 4 s window takes
69 ms on a T4 GPU and 1.14 s on two CPU threads (wav2vec2-base: 25 ms and 0.40 s). The live
path scores a window every 2 s, so even that CPU keeps up, at 57% of the budget.

**Spontaneous speech.** One training calls 5.8% of the genuine spontaneous clips fake, the
first model in the project under 10%; the other calls 46.7%, close to the adapter's 43.9%.
Both reach 0% dev EER, so model selection cannot tell them apart. The larger encoder can
learn a notion of genuine speech that covers spontaneous talk, but not reliably without
spontaneous speech in training (`docs/results/encoder_spontaneous_v1.md`).

**Reruns differ.** The same seed on the GPU does not give the same model: seed 1234 reaches
25.8% on IndicF5 here and 12.7% in run 7. Every claim in the paper rests on several seeds
for this reason.

The results chapter, the conclusion and the four-page version now report the speed and the
spontaneous result.

## Viva preparation

`docs/viva/VIVA_QA.md` holds the questions an examiner is likely to ask, each with a short
answer and the one number that matters, read from the long paper: the problem, data and
ethics, method, results, the live system, the real-call test, the limitations said plainly,
and a table of numbers to remember. It is written to be understood and then answered in our
own words, not read out. `docs/viva/demo_script.md` was still the week-10 version; it now
quotes the current numbers (58.42%, not "about 54%"), adds the language-controlled finding and
the final recipe, says plainly that the Twilio leg cannot be shown live because the trial
credit ran out, points to the real-call test instead, and states the threshold rule correctly
(at most 5% warnings on genuine development calls, not "zero false alarms").

The real-call answers use the first 20 of 30 calls and are updated when the last speaker's
calls are in.

## What is still open

Two trainings are enough to show the spread, not to estimate it. The larger encoder has not
been put into the live system; it is the candidate for the next release.
