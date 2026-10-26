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

Two answers were wrong and are corrected. The early false alarms on genuine calls were first
put down to the phone's noise suppression, then to the quiet before the first word alone;
the window levels and a replay with the first window treated as silence show the quiet start
is part of the cause, not all of it (Q35, Q35a). And the demo script said the final recipe
fixes the Tortoise demo call. A rehearsal with the final recipe showed that call stays green
for both models (2 of 22 windows fake-leaning, against 0), although the final recipe catches
every Tortoise test call. The script now says so before the call is played, points to a
second set of demo calls verified on the final recipe (`demo_clips_final`), and explains why
an uploaded clip can read "spoof" while the same audio as a call stays green (Q31a).

A check of every answer against the paper corrected three more. The 94.8% of genuine speech
that S1 calls fake is measured on clean audio, so it does not explain the 58.42% over the
phone line by itself (Q4 and the opening summary). The threshold rule counts alerts of any
level, not only warnings. And five training seeds back the main comparisons, not every
claim: Chatterbox for the adapter, English with the final recipe and the spontaneous-speech
comparisons are one seed each (Q43). A new question, Q47, answers what changed after the
external review, since the mentor is likely to ask it.

The call-chain answer (Q34) said the real codec "is what breaks" the deployed model and that
training on it "is the fix". The chain shows the codec takes it below its threshold on average, and the
live calls cost it more, so the network and handset add damage; and the final recipe changes
voices and codec together, so the calls cannot say which one fixes it. The answer now says
that and gives the offline result that does isolate the codec (AMR-NB added to the deployed
recipe's own training, 7.59% against 22.37%). Two explanations in the results section that
no experiment tested (why the Tortoise floors reach chance on new speakers, why AMR-NB hurts
the channel-trained adapter more) are now marked as likely, not stated as findings.

Two tables in the long paper ran about a third of an inch past their column in the Overleaf
build: the training settings and the week-13 sets with their intervals. Both now use the
smaller type and tighter column spacing the paper's other wide tables use.

An independent read of the long paper, each point checked against the results files before
anything changed, found results-section wording that said more than the numbers:

- The language-control test: mixed speech is rejected significantly *less* often than
  English (-9.5 [-15.6, -3.5]), so "almost exactly as often" became "at least as often".
- The final-recipe floors: ten sets have five seeds and Chatterbox one, not "all eleven";
  the deployed recipe's unbeaten floor is CM04's; the paired and unpaired seed tests are
  now named side by side (the paired one cannot go below 0.0625 with five seeds).
- Chatterbox is the hardest tool for the adapters only (SSL-AASIST does better on it than on
  IndicF5, so for it a watermark cue is not ruled out); its smaller gain from the added
  voices is no longer explained by "a different family", since that model also changed
  codec and each row is one model; IndicF5's 13-point gain is now in the comparison.
- SSL-AASIST with 21 speakers is not lower than the adapter in false alarms (3.7 against
  3.4); IndicF5 is the most natural of the tools *scored*; the probe section says which
  numbers come from the five calls and which from 86 English clips; a model is scored in
  its training condition except in the cross-condition tests, which the method now says.

A read of the method chapter against the code fixed three statements: all three systems use
inverse-frequency class weighting (`class_weighted` defaults to true in
`src/training/train.py`, and S1 needs it most, at 8.7:1), not S2 and S3 only; S2's 1,082,371
trainable parameters include the attentive pooling as well as the adapters and the head; and
the model has 94,864,259 parameters, 95,454,083 with the adapters. The floor paragraph now
says which detectors have their own floor (S1 is read against the lowest code-mixed floor
and the AASIST baselines have none).

## What is still open

Two trainings are enough to show the spread, not to estimate it. The larger encoder has not
been put into the live system; it is the candidate for the next release.
