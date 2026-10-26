# Viva preparation: questions and answers

Every number here is from the long paper (`paper/main.tex`) or the results file it names.
Each of us should be able to answer every question in our own words, not read these
answers out. Answer the question asked, give the one number that matters, and say what we
did not show when that is the honest answer: examiners trust a team that knows its limits.

## 1. The project in two minutes

Voice-cloning scams in India arrive as phone calls in Hindi-English code-mixed speech.
Deepfake detectors are trained on English studio speech and report error rates below 1%.
We measured what happens to such a detector on Hindi-English speech over a telephone line:
its error rate goes from 0.85% to 58.42%, because it calls 94.8% of genuine Hindi-English
speakers fake. A language-controlled test on 183 unseen speakers showed the cause is the
recording domain, not the code-mixing. We fixed it cheaply with LoRA adapters that train
1.13% of the parameters, then showed that hearing more genuine voices through a real
mobile codec closes most of the remaining gap on attack tools the model never saw. Every
result is checked against a shortcut floor, so a low error rate cannot come from loudness or
bandwidth differences alone. The detector runs on live calls and warns only the person
receiving the call, about ten seconds into the speech, and we tested it on real phone calls
over a mobile network.

## 2. The problem

**Q1. Why is this problem important?**
Cloning a voice needs a few seconds of audio, and the obvious misuse is a phone call asking
for money or a one-time code. In India that call is usually code-mixed Hindi-English and
arrives over a narrowband phone line, which is exactly where detectors were never tested.

**Q2. What does "code-mixed" mean?**
Switching between Hindi and English inside one sentence, for example "kal ki meeting ka time
change ho gaya hai". MUCS 2021 writes the Hindi words in Devanagari and the English words
in Latin script.

**Q3. What is EER?**
Equal error rate: the error at the threshold where the false-alarm rate on genuine speech
equals the miss rate on fakes. 50% is chance; lower is better.

**Q4. What exactly fails when the English detector meets Hindi-English speech?**
It rejects real speakers rather than missing clones. At its operating point it calls 94.8% of
genuine Hindi-English speech fake but only 0.37% of genuine English, a 256-fold rise in
false alarms. Over the phone channel its AUC is 0.381: clones look more genuine to it than
real people.

## 3. Data and ethics

**Q5. Which data did you use?**
ASVspoof 2019 LA (English) to train the base detector; MUCS 2021 Hindi-English (52,825 clips,
89.5 hours, 520 speakers) for genuine code-mixed speech and cloning references; HiACC (3,318
adult clips, 24 speakers of spontaneous speech) for evaluation only.

**Q6. Where did the fakes come from?**
No code-mixed deepfake benchmark exists, so we made them from job tables committed before
each run: XTTS-v2 clones (the seen tool), RVC voice conversion, and three tools never used
in training: Tortoise, IndicF5 and Chatterbox.

**Q7. How did you stop test speakers leaking into training?**
Speaker pools were divided and committed before any audio was cloned, and
`tests/test_splits.py` fails the build if a speaker crosses a pool, a held-out tool reaches a
training manifest, or an excluded file appears anywhere.

**Q8. What about the child recordings in HiACC?**
HiACC has 1,858 child files out of 5,176. They are excluded from every use. The automatic
quarantine once failed silently and we caught it by hand; we report that in the paper,
because a safeguard nobody checks is not a safeguard.

**Q9. Is cloning people's voices ethical?**
The mentor signed an ethics sign-off before any generation, and the code refuses to load a
cloning model without it. The cloned audio is never released, only its metadata and
checksums. For the real-call test, the three of us signed a separate consent form and the
mentor approved it; our clones stay private and are deleted after the project.

**Q10. Why normalise the loudness of every clip?**
Before normalisation, eight simple signal statistics separated real from fake at 1.39% EER,
because the clones were louder. Our first headline number (1.34%) had no margin over that,
so we withdrew it and normalised every clip to -23 dBFS.

## 4. Method

**Q11. What is the model?**
A wav2vec 2.0 base encoder, attentive statistics pooling and a two-layer classifier, 95.5
million parameters, outputting P(genuine) for a 4-second window.

**Q12. What are S1, S2 and S3?**
S1 is trained on English only. S2 adds LoRA adapters to S1's frozen encoder and trains them on
code-mixed data. S3 fine-tunes the whole encoder on the same code-mixed data, starting from
the pretrained model.

**Q13. What is LoRA and why use it?**
Low-rank adaptation: small trainable matrices (rank 8) added to the attention projections,
1.13% of the parameters. We use it because it learns code-mixed speech while keeping English:
S2 keeps English at 2.42% EER, while S3 reaches the best seen-tool error (0.83%) but falls to
chance on English (50.90-56.42%).

**Q14. What is the "telephone channel" in your experiments?**
A simulated narrowband line: resample to 8 kHz, G.711 mu-law, noise at 20 dB SNR, back to
16 kHz, validated by measurement. We also pass test sets through the real AMR-NB mobile codec,
and the final recipe trains on both.

**Q15. What is the shortcut floor?**
A logistic regression on eight cheap signal statistics (loudness, peak, clipping, DC offset,
zero-crossing rate, spectral roll-off, high-frequency energy), fitted on the detector's own
training data and scored on its own test clips. A result counts only if the detector beats
its floor, tested with a bootstrap that resamples whole speakers.

**Q16. Why resample speakers rather than clips?**
Clips from one speaker are not independent, and a test set may hold only 12 to 15 speakers.
Resampling clips would make intervals look narrower than they are.

**Q17. How did you choose thresholds without cheating?**
Only on the development set, by a rule fixed in advance: the most sensitive setting whose
genuine 60-second dev calls raise at most 5% warnings. Test sets never choose anything.

## 5. Results

**Q18. Is the gap caused by code-mixing?**
No. On 183 unseen speakers, holding speaker and recording fixed, S1 rejects 84.3% of their
genuine English and 78.1% of their mixed speech, and the Code-Mixing Index does not predict its
score (rank correlation -0.01). The gap follows the recording domain.

**Q19. How much does LoRA fix?**
Over the phone channel the deployed adapter reaches 2.75% EER on unseen speakers (3.27% +/- 1.17
over five seeds), 7.21 points below its shortcut floor, with clean English at 2.42%.

**Q20. What about attack tools the model never saw?**
On clean audio the adapter generalises to Tortoise (5.44% against a 29.49% floor). Over the
phone line the deployed model averages 35.59% across five seeds against a 37.79% floor, so we
cannot claim it detects an unseen tool over a phone line. We say so in the abstract.

**Q21. What closed most of the remaining gap?**
More genuine voices, not more attacks. Retraining on 248 genuine speakers instead of 21, with no
unseen-tool audio, cut new-speaker Tortoise error from 29.82% to 13.59% and false alarms from
3.40% to 0.47%, better in every one of five seeds.

**Q22. What is the final recipe?**
The same adapter trained on 248 speakers with every clip through both the simulated G.711 line
and the real AMR-NB codec. Over AMR-NB, unseen Tortoise error falls from 36.3% to 6.0%, and every
seed beats its shortcut floor on all eleven test sets.

**Q23. Did you compare with established systems?**
Yes. AASIST and SSL-AASIST fail the same way zero-shot (they reject 82.3% and 73.1% of genuine
Hindi-English speech). SSL-AASIST, three times our encoder's size, trained on the final-recipe
data is the strongest system: 0.51% on new-speaker Tortoise over AMR-NB and 12.86% on
Chatterbox, against 31.98% for our adapter.

**Q24. Why not deploy SSL-AASIST then?**
It is the candidate for the next release: it keeps up with a live call even on two CPU threads
(1.14 s per window, inside the 2 s hop). But on spontaneous speech two trainings with equal
development scores call 5.8% and 46.7% of genuine clips fake, so it is not yet reliable.

**Q25. Which attack is hardest?**
IndicF5, whose clones are rated as natural as genuine speech (UTMOS 3.07 against 3.10), and
Chatterbox. IndicF5 stays at 15.8-21.8% even for the best system.

**Q26. How do you know the model is not just reading the language?**
Shuffling a call into 100 ms pieces destroys words, grammar and code-mixing, and the verdicts do
not move. The decision is already formed by the third transformer layer, before linguistic
content appears in wav2vec 2.0. Scrambling the phase while keeping the spectrum flips genuine
audio to fake, so the cue is in fine waveform structure.

**Q27. Does adapting to Hindi-English hurt English?**
On clean English the deployed adapter keeps 2.42% EER. Over the phone line English rises to
24.20%, so the deployed model is for Hindi-English calls; a service taking both would need to
route English calls to S1 or train on English telephone audio.

## 6. The live system

**Q28. How does the live system work?**
Audio arrives from a browser call (WebRTC) or a phone call (Twilio). A scorer takes a 4-second
window every 2 seconds, skips silence, normalises the level and scores it. A ladder escalates
only when consecutive windows agree: a caution after two fake-leaning windows, a warning after
four, recovery after three genuine-leaning ones.

**Q29. Why warn only the person receiving the call?**
A caller who learns they have been detected learns how to evade the detector.

**Q30. How fast is it?**
The first caution comes near 6 seconds of speech and the warning near 10 seconds.
wav2vec2-base scores a window in 25 ms on a T4 GPU and 0.40 s on two CPU threads.

**Q31. Is the system the same as the evaluation?**
Yes: the same encoder, windowing, silence gate and operating point, which is why a demonstration
counts as evidence and not just an illustration.

**Q31a. In the demo, why does the Tortoise call stay green even with the final recipe?**
Because one call is not the rate. Over all the Tortoise test calls the final recipe catches
every one (on CM04 the deployed model catches one in three), but on this speaker's demo
call only 2 of 22 windows lean fake, too few for a warning. We kept the call that fails
rather than pick one that works. The upload box can still call the same file "spoof", because it judges a clip by
its single lowest window, while a call needs several fake-leaning windows in a row.

## 7. The real phone-call test

These are the first 20 of the 30 planned calls (two of the three speakers); the numbers are
updated when the last speaker's calls are in (`docs/results/real_calls_v1.md`).

**Q32. Did you test on real phone calls?**
Yes, as a pilot. With written consent from all three of us and the mentor's approval, we made
calls over a normal mobile network, recorded on the receiving phone: genuine calls (prepared
scripts and free talk), our own readings replayed from a laptop speaker into a call, and clones
of our voices (XTTS-v2, the seen tool, and Chatterbox, an unseen one) replayed the same way.
Every call was scored through the live path at each model's frozen operating point; no
threshold was chosen on these calls.

**Q33. What happened?**
The deployed model warned every call, genuine or cloned: on real calls it cannot tell them
apart. The final recipe held up. As the live system's carrier would deliver the audio, it took
one of eight genuine live calls to "likely cloned" and cautioned two more, took three of four
XTTS-v2 clone calls to "likely cloned", and none of the four Chatterbox clone calls (it only
cautioned one), which matches
Chatterbox being the hardest unseen tool offline.

**Q34. Why does the deployed model fail on real calls?**
We rebuilt the call chain on each speaker's direct recordings, one stage at a time. The deployed
model's mean P(genuine) goes from 0.77 (direct) and 0.63 (simulated G.711 line) to 0.29 through
the real AMR-NB mobile codec; the call recorder adds little. The final recipe, which trained on
AMR-NB, stays at 0.91-0.98 through every stage. So the real mobile codec is what breaks the
deployed model, as the paper's offline AMR-NB result predicted, and training on it is the fix.

**Q35. What does it change?**
It is the strongest evidence for the final recipe: real codec audio and more voices in training
are what make the detector usable on real calls, so the final recipe should replace the deployed
adapter in the live system. False alarms on genuine calls come early in a call (mean score 0.73
in the first ten seconds against 0.98 after twenty). Part of the reason is the very first
window, which covers the quiet before the caller's first word: it is just loud enough to pass
the -50 dBFS gate, gets normalised, and is scored as if it were speech. But it is not the whole
reason (Q35a). So the next step is a set of real development calls, separate from these, to set
a stricter speech gate for the first seconds and the operating point together. Chatterbox-type
tools remain the open problem.

**Q35a. How much of the early false alarms is the quiet start?**
Part, not all. On the genuine live calls the first window averages about -30 dBFS, against -20
to -26 dBFS for the rest of each call, and it is the lowest-scoring window (mean 0.48; every
later window averages 0.69 or more). In every warned genuine call, the run of fake-leaning
windows that raised the warning starts there. But when we replay the calls with that window
treated as silence, as a check that chooses nothing, two of the three warned calls clear over
G.711, none clear as recorded, and one call still reaches "likely cloned" from speech windows
8-14 seconds in. We first guessed the phone's noise suppression, then the silence alone; the
data corrected us both times, and we say so.

## 8. Limitations, said plainly

**Q36. What is the weakest result?**
Detecting an unseen tool over a phone line with the deployed model: 35.59% against a 37.79%
floor, no seed better. The final recipe and SSL-AASIST improve it a great deal, but three of their
floors are above chance and so weak bars.

**Q37. What about natural conversation?**
All code-mixed training speech is read lectures. On spontaneous HiACC speech the deployed model
calls 54.37% of genuine clips fake (0.89% for read lectures) and the final recipe 43.9%. The first
future step is spontaneous genuine speech in training.

**Q38. Are your fakes realistic?**
They are our own, made with open tools, and most are not hard: the seen tool sits half a MOS point
below genuine speech. IndicF5 is as natural as real speech and is also the hardest to detect, so
numbers on easy fakes are optimistic.

**Q39. Did you test commercial cloning services?**
No. Only open tools.

**Q40. Is anything unexplained?**
On a live browser call a genuine caller sometimes loses consecutive windows that the same clip
never loses offline. Encoding the clips through the browser codec offline does not reproduce it,
so the cause lies in real-time transport (jitter buffering or packet-loss concealment); we have
not isolated it.

## 9. Questions examiners like to ask

**Q41. What is new here?**
No single ingredient is new. The contribution is the combination and the measurement discipline:
Hindi-English switching within an utterance, in audio, over the same telephone channel for
training and testing; every result against a matched shortcut floor with a speaker-level test;
adaptation judged by what it keeps of English; and the detector deployed on live calls and
tested on real ones.

**Q42. If you had three more months?**
Spontaneous genuine speech in training (from a speaker-disjoint corpus, with clones of the same
speakers so speaking style cannot become the shortcut); English routing; the larger encoder in the
live system; the Twilio leg on live numbers; and harder, commercial attacks.

**Q43. Why should we trust numbers from such small test sets?**
We do not hide it: intervals come from a speaker-level bootstrap, five training seeds back every
main claim, and seed-to-seed spread (three to five points on Tortoise) is reported, so a
single-run difference of that size is not treated as evidence.

**Q44. Did you tune anything on the test sets?**
No. Thresholds, checkpoints and recipe choices are made on development data by rules fixed in
advance. One variant chosen after a test score was seen is reported as exploratory.

**Q45. Did you use any tools to help write the code or the paper?**
Answer honestly and the same way the paper's acknowledgments do: which tools, what they were used
for, and that every result was run, checked and verified by us. Do not deny or play down any use.

**Q46. Who did what?**
Answer from the weekly records in `docs/W*_*.md` and the commit history; each of us should
describe our own parts and be able to explain the others'.

## 10. Numbers to remember

| What | Number |
|---|---|
| English detector on English | 0.85% EER |
| Same detector, Hindi-English over the phone line | 58.42% EER |
| Genuine Hindi-English speech it calls fake | 94.8% |
| Trainable share with LoRA | 1.13% |
| Deployed adapter, phone line, five seeds | 3.27% +/- 1.17 |
| Unseen Tortoise over AMR-NB: deployed vs final recipe | 36.3% vs 6.0% |
| False alarms: 21 vs 248 genuine speakers | 3.40% vs 0.47% |
| SSL-AASIST on Chatterbox (final data) | 12.86% |
| First caution / warning on a live call | about 6 s / 10 s |
| Speakers in the language-controlled test | 183 |
