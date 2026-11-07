# Viva preparation: questions and answers

Every number here is from the long paper (`Paper_Journal/main.tex`) or the results file it names.
Each of us should be able to answer every question in our own words, not read these
answers out. Answer the question asked, give the one number that matters, and say what we
did not show when that is the honest answer: examiners trust a team that knows its limits.

## 1. The project in two minutes

Voice-cloning scams in India arrive as phone calls in Hindi-English code-mixed speech.
Deepfake detectors are trained on English studio speech and report error rates below 1%.
We measured what happens to such a detector on Hindi-English speech over a telephone line:
its error rate goes from 0.85% to 58.42%, and on clean audio it already calls 94.8% of genuine
Hindi-English clips fake. A language-controlled test on 183 unseen speakers showed the cause is the
recording domain, not the code-mixing. We fixed it cheaply with LoRA adapters that train
1.13% of the parameters, then showed that hearing more genuine voices through a real
mobile codec closes most of the remaining gap on Tortoise, a tool the model never saw
(IndicF5 and Chatterbox improve less and stay hard). Every
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
It rejects real speakers rather than missing clones. At its operating point, on clean audio, it
calls 94.8% of genuine Hindi-English speech fake but only 0.37% of genuine English, a 256-fold
rise in false alarms. Over the phone channel its AUC on the evaluation pool is 0.381: clones look more genuine to it than
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
Low-rank adaptation: small trainable matrices (rank 8) added to the attention projections;
with the attentive pooling and the classifier head, 1.13% of the parameters train (the
adapters alone are about 0.6%). We use it because it learns code-mixed speech while keeping English:
S2 keeps clean English at 2.42% EER (over the phone line English rises to 24.20%), while S3
reaches the best seen-tool error (0.83%) but is at chance on English (50.90-56.42%). Be exact
if asked: S3 starts from the self-supervised checkpoint, not from S1, so it never trained on
English and did not "forget" it. The adapter keeps English because it is built on S1. Whether
fully fine-tuning S1 itself would keep English was not tested.

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
genuine 60-second dev calls raise at most 5% alerts of any level. Test sets never choose anything.

## 5. Results

**Q18. Is the gap caused by code-mixing?**
No. On 183 unseen speakers, holding speaker and recording fixed, S1 rejects 84.3% of their
genuine English and 78.1% of their mixed speech, so English at least as often. On the evaluation
pool the Code-Mixing Index does not predict its score (rank correlation -0.01 clean, +0.13 over
the channel). The gap follows the recording domain.

**Q19. How much does LoRA fix?**
Over the phone channel the deployed adapter reaches 2.75% EER on unseen speakers (3.27% +/- 1.17
over five seeds), 7.21 points below its shortcut floor, with clean English at 2.42%.

**Q20. What about attack tools the model never saw?**
On clean audio the XTTS-only adapter generalises to Tortoise (5.44% against a 29.49% floor). Over the
phone line the deployed model averages 35.59% across five seeds against a 37.79% floor, so we
cannot claim it detects an unseen tool over a phone line. We say so in the abstract.

**Q21. What closed most of the remaining gap?**
More genuine voices, not more attacks. Retraining on 248 genuine speakers instead of 21, with no
unseen-tool audio, cut new-speaker Tortoise error from 29.82% to 13.59% and false alarms from
3.40% to 0.47%, better in every one of five seeds. If asked why: the added speakers came with
seen-tool clones of them, so this run cannot say whether the genuine speech or the clones did
it. A broader idea of genuine speech is the likely reason, and the papers say only that.

**Q22. What is the final recipe?**
The same adapter trained on 248 speakers with every clip through both the simulated G.711 line
and the real AMR-NB codec. Over AMR-NB, unseen Tortoise error falls from 36.3% to 6.0%, and every
seed beats its shortcut floor on all eleven test sets. Say the caveat before you are asked:
three of those floors (CM04 over either codec, new-speaker Tortoise over AMR-NB) are above
chance, so beating them is a weak bar.

**Q23. Did you compare with established systems?**
Yes. AASIST and SSL-AASIST fail the same way zero-shot (they reject 82.3% and 73.1% of genuine
Hindi-English speech). SSL-AASIST, three times our encoder's size, trained on the final-recipe
data is the strongest system: 0.51% on new-speaker Tortoise over AMR-NB and 12.86% on
Chatterbox, against 31.98% for our adapter.

**Q24. Why not deploy SSL-AASIST then?**
It is the candidate for the next release: it keeps up with a live call even on two CPU threads
(1.14 s per window, inside the 2 s hop), and on our 30 real calls it did better than the final
recipe (Q33a). But on spontaneous speech two trainings with equal development scores call 5.8%
and 46.7% of genuine clips fake, so it is not yet reliable. Retrained with spontaneous speech,
both trainings call under 2% of HiACC fake, which removes that instability, and in the voice
check they catch far more clones; one of them misses the read-speech bar we set before measuring,
so it is the next version, not this one (Q37).

**Q25. Which attack is hardest?**
It depends on the detector. For our adapters, Chatterbox (32.0%, one seed). For the larger
encoder, IndicF5, whose clones are rated as natural as genuine speech (UTMOS 3.07 against 3.10)
and which stays at 15.8-21.8% even for the best system.

**Q26. How do you know the model is not just reading the language?**
On five held-out calls with the deployed model, shuffling a call into 100 ms pieces destroys
words, grammar and code-mixing, and the verdicts do not move. The decision is already formed by the third transformer layer, before linguistic
content appears in wav2vec 2.0. Scrambling the phase while keeping the spectrum flips genuine
audio to fake, which points to fine waveform structure. Five calls can refute an explanation,
not establish one.

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
With the deployed model the first caution comes near 6 seconds of speech and the warning near
10 seconds; the final recipe's ladder (3 and 5 windows) puts them near 8 and 12 seconds. On the
real calls the final recipe took one speaker's XTTS-v2 clones to "likely cloned" 12-14 s in,
and the other two speakers' only 30-46 s in.
wav2vec2-base scores a window in 25 ms on a T4 GPU and 0.40 s on two CPU threads.

**Q31. Is the system the same as the evaluation?**
The same encoder, windows, silence gate and normalisation, at an operating point calibrated on
development calls through the live path, which is why a demonstration
counts as evidence and not just an illustration.

**Q31a. In the demo, why does one Tortoise call stay green even with the final recipe?**
Because one call is not the rate. Over the mixed 60 s Tortoise test calls the final recipe
catches every one (on CM04 the deployed model catches one in three); built per speaker, 45 s
calls from 12 of the 15 CM04 Tortoise speakers are caught. The demo has one of each: a caught
call (the caught speaker with the most clips, warned 8 s in) and a missed one, where only 2 of
22 windows lean fake, too few for a warning. We kept the call that fails rather than show only
one that works. The upload box can still call the same file "spoof", because it judges a clip by
its single lowest window, while a call needs several fake-leaning windows in a row.

## 7. The real phone-call test

All 30 planned calls, from the three of us (`docs/results/real_calls_v1.md`).

**Q32. Did you test on real phone calls?**
Yes, as a pilot. With written consent from all three of us and the mentor's approval, we made
calls over a normal mobile network, recorded on the receiving phone: genuine calls (prepared
scripts and free talk), our own readings replayed from a laptop speaker into a call, and clones
of our voices (XTTS-v2, the seen tool, and Chatterbox, an unseen one) replayed the same way.
Every call was scored through the live path at each model's frozen operating point; no
threshold was chosen on these calls.

**Q33. What happened?**
The deployed model took every replayed call, genuine or cloned, to "likely cloned", and every
live call of two of us; the third speaker's four live calls passed. For two speakers of three
it cannot tell real from cloned. The final recipe held up on the seen tool. As the live
system's carrier would deliver the audio, it warned three of twelve genuine live calls (all
the same speaker's; one reached "likely cloned"), took five of six XTTS-v2 clone calls to
"likely cloned", and none of the six Chatterbox clone calls (it only cautioned one), which
matches Chatterbox being the adapter's hardest unseen tool offline. With twelve calls a
percentage moves in steps of 8 points: one of twelve is a 95% interval of 1.5-35%.

**Q33a. And the larger encoder on the same calls?**
It did better. We retrained SSL-AASIST on Kaggle with the week-13 settings so its weights could
be kept, chose its operating point on the same 506 dev clips by the same rule before scoring
any call (threshold 0.580, caution after two windows, "likely cloned" after four), and scored
the 30 calls. Over G.711 it warned one of twelve genuine live calls (a caution 48 s in, none
"likely cloned"), took all six XTTS-v2 calls to "likely cloned" (four of them 10-14 s in),
caught one Chatterbox call and cautioned two, and, like the final recipe, took one of six
genuine replays to "likely cloned". If asked how sure: we scored a second checkpoint (seed
2025) the same way in week 15. It agrees on what matters most here (all six XTTS-v2 calls
"likely cloned", no genuine live call warned) but catches none of the six Chatterbox calls,
and on held-out 60 s calls it warns 31 of 41 genuine spontaneous calls where seed 1234 warns 11.
Their average warns none of the 18 genuine calls and 23 of 41 spontaneous calls. So the larger
encoder does not replace the final recipe yet: which training generalises to spontaneous
speech is luck until spontaneous speech is in training (`docs/results/ensemble_v1.md`).

**Q34. Why does the deployed model fail on real calls?**
We rebuilt the call chain on each speaker's direct recordings, one stage at a time. Over the six
readings, the deployed model's mean P(genuine) goes from 0.83 (direct) and 0.75 (simulated G.711
line) to 0.50 through the real AMR-NB mobile codec, its threshold; the call recorder adds
little. The final recipe, which trained on AMR-NB, stays at 0.82-1.00 for every reading at
every stage. Per reading: two of six cross the deployed threshold at AMR-NB, one stays above,
one was below it before any codec, and the third speaker's two stay at 0.87-1.00 throughout.
So the codec is the largest step for the speakers the model fails on, not a failure of every
voice; the live calls cost it more still (0.38 on the genuine live calls), so the network and
handset add damage too, as the paper's offline AMR-NB result predicted. If asked whether the real codec
in training is what fixes it: the final recipe also has more voices, so the calls cannot
separate the two; offline, adding AMR-NB alone to the deployed recipe's training recovers
most of the loss (new-speaker XTTS over AMR-NB 7.59% against 22.37%, three seeds).

**Q35. What does it change?**
It is the strongest evidence for the final recipe: with real codec audio and more voices in
training it holds up on the seen tool on real calls (not on Chatterbox), though the calls cannot
say which change did it, so the final recipe should replace the deployed adapter in the live
system. False alarms on genuine calls come early in a call (mean score 0.79
in the first ten seconds against 0.98 after twenty). Part of the reason is the very first
window, which covers the quiet before the caller's first word: at about -30 dBFS it is well above
the -50 dBFS gate, so it gets normalised and is scored as if it were speech. But it is not the whole
reason (Q35a). So the next step is a set of real development calls, separate from these, to set
a stricter speech gate for the first seconds and the operating point together. Chatterbox-type
tools remain the open problem.

**Q35a. How much of the early false alarms is the quiet start?**
Part, not all. On the genuine live calls the first window sits at a median -30 dBFS, against
-21 to -26 dBFS for the rest of each call; in eleven of twelve calls it passes the -50 dBFS gate
and is scored as speech, at a mean of 0.60 against 0.96 for the later windows. In every warned genuine live call, the run of fake-leaning
windows that raised the warning starts there. But when we replay the calls with that window
treated as silence, as a check that chooses nothing, two of the three warned calls clear over
G.711; as recorded one clears but another call is cautioned instead, and one call still reaches
"likely cloned", its run including two speech windows 8-14 seconds in. We first guessed the phone's noise suppression, then the silence alone; the
data corrected us both times, and we say so. If asked why we did not just raise the gate: we
tried one set on the dev clips only (the first 10 s, quieter than -29.9 dBFS treated as
silence). Over G.711 it clears two of the three warned calls, but as recorded it warns one
more, it delays every model's warnings by 2-5 s, and it silences most early speech, because
real calls are much quieter than the dev clips. So the gate needs real development calls.

## 8. Limitations, said plainly

**Q36. What is the weakest result?**
Detecting an unseen tool over a phone line with the deployed model: 35.59% against a 37.79%
floor, no seed better. The final recipe and SSL-AASIST improve it a great deal, but three of their
floors are above chance and so weak bars.

**Q37. What about natural conversation?**
All code-mixed training speech is read lectures. On spontaneous HiACC speech the deployed model
calls 54.37% of genuine clips fake (0.89% for read lectures) and the final recipe 43.9%, one
seed inside the 38.5-54.4% spread of single trainings, so not evidence of a gain. A different
threshold does not fix it either: sweeping the final recipe's threshold over 60 s calls,
choosing nothing, every threshold that catches all Tortoise calls warns at least 38 of the 41
genuine spontaneous calls (`docs/results/spontaneous_tradeoff_v1.md`). In week 15 we added a
second spontaneous test set, 30 IndicVoices speakers (extempore Hindi): the final recipe calls
37.5% of their clips fake, so the failure is not a property of HiACC. The larger encoder's seed
1234, the one that did well on HiACC (11 of 41 calls warned), warns 79 of 94 IndicVoices calls,
so no model we have handles spontaneous speech (`docs/results/indicvoices_data_v1.md`). The
failure is on the genuine side: XTTS-v2 clones of the same 30 spontaneous speakers are still
caught (96.3% by the adapter, all of them by SSL-AASIST), but a third of their genuine clips
score like clones for the adapter. SSL-AASIST ranks the genuine clips above those clones (2-3% EER).
So with decision D1 (part a) we chose every operating point again by the unchanged rule, with 20
spontaneous IndicVoices speakers in the dev set and no retraining. For SSL-AASIST seed 1234 the
voice check then calls 0.7% of HiACC's and 5.0% of IndicVoices' genuine spontaneous files cloned
(7.6% and 15.3% before), but lets through 32% of Tortoise and 96% of IndicF5 and Chatterbox
clones (4%, 71%, 78% before). Threshold-free, its file score separates spontaneous genuine
speech from XTTS-v2 clones (1.5% EER) but not from IndicF5 or Chatterbox clones (40-44%): the
newest clones sound as genuine to the model as people speaking freely, so no threshold serves
both (`docs/results/operating_points_spont_v1.md`). We adopted the new point for the voice check,
as agreed before the results, because a false alarm on a real person is the costlier error, and
we say plainly what it lets through. The fix has to be in the model, so part (b) retrained with
150 spontaneous IndicVoices speakers, hypotheses written before launch. Both families are a
trade-off. HiACC, a corpus never trained on, falls below 2% of genuine clips called fake for every
seed (from 43.9% for the adapter, 26.2% for SSL-AASIST), and spontaneous genuine speech moves away
from every tool's clones: for SSL-AASIST, IndicF5 and Chatterbox go from 39-57% to 8-20% EER,
which no threshold could do. But the read-speech benchmarks pay more than the 2 points we
allowed: 8-11 for the adapter, 3.4-3.7 for SSL-AASIST on Tortoise and Chatterbox (IndicF5
+1.2), on the seed mean. The verdicts stand as
written and the frozen models stay the reported ones (`docs/results/spontaneous_training_v1.md`).
In the voice check the retrained SSL-AASIST catches far more unseen clones (Chatterbox 52% against
4%, the five unseen-tool rows 70-75% against 44%), but we had committed the adoption rule first,
and one of the two trainings judges only 97.6% of read genuine speech real against the 99% we
required. So the voice check keeps its detector (P-040, `docs/voice_check.md`).

**Q38. Are your fakes realistic?**
They are our own, made with open tools, and most are not hard: the seen tool sits half a MOS point
below genuine speech. IndicF5 is as natural as real speech and stays at 15.8-21.8% even for the larger
encoder (for our adapters Chatterbox is harder still), so numbers on easy fakes are optimistic.

**Q39. Did you test commercial cloning services?**
No. Only open tools.

**Q40. Is anything unexplained?**
On a live browser call a genuine caller sometimes loses consecutive windows that the same clip
never loses offline. An offline round trip through a default Opus encoder (not the browser's own)
does not reproduce it; real-time transport (jitter buffering, packet-loss concealment) or the
browser's adaptive encoder remain the likeliest causes, and we have not isolated it.

## 9. Questions examiners like to ask

**Q41. What is new here?**
No single ingredient is new. The contribution is the combination and the measurement discipline:
Hindi-English switching within an utterance, in audio, over the same telephone channel for
training and testing; every result for our systems against a matched shortcut floor with a
speaker-level test;
adaptation judged by what it keeps of English; and the detector deployed on live calls and
tested on real ones.

**Q42. If you had three more months?**
Spontaneous genuine speech in training (from a speaker-disjoint corpus, with clones of the same
speakers so speaking style cannot become the shortcut); English routing; the larger encoder in the
live system; the Twilio leg on live numbers; and harder, commercial attacks.

**Q43. Why should we trust numbers from such small test sets?**
We do not hide it: intervals come from a speaker-level bootstrap, five training seeds back the
main comparisons, and seed-to-seed spread (three to five points on Tortoise) is reported, so a
single-run difference of that size is not treated as evidence. The few one-seed results
(Chatterbox for the adapter, English with the final recipe, the spontaneous-speech
comparisons) are marked as one seed wherever they appear.

**Q44. Did you tune anything on the test sets?**
No. Thresholds, checkpoints and recipe choices are made on development data by rules fixed in
advance. One variant chosen after a test score was seen is reported as exploratory, and when we
tested it properly on the final recipe (five seeds, rule committed first) it was not confirmed:
worse on IndicF5 in every seed, 34.24% against 21.68%. Be ready
for the follow-up: the final recipe joins two changes whose gains were first seen on test sets,
so we report it with five seeds and shortcut floors, not as a blind selection. This week's two
decisions on spontaneous speech followed rules committed before the measurement; when the
better-looking retrained model missed one condition on one seed, we kept the old detector rather
than pick the seed that passed.

**Q45. Did you use any tools to help write the code or the paper?**
Answer honestly and the same way the paper's acknowledgments do: which tools, what they were used
for, and that every result was run, checked and verified by us. Do not deny or play down any use.

**Q46. Who did what?**
Answer from the weekly records in `docs/W*_*.md` and the commit history; each of us should
describe our own parts and be able to explain the others'.

**Q47. What did you change after the external review?**
All six comments were taken as mandatory (`docs/external_review_response.md`). Five training
seeds behind the main claims (two to four for the outside baselines) showed our published Tortoise number (30.99%) was a lucky draw, so the paper
now quotes the five-seed mean (35.59%). For the unseen tools, the answer was more genuine voices
(21 to 248) and the real AMR-NB codec in training: unseen Tortoise over AMR-NB from 36.3% to
6.0%. We added 60 new test speakers, 183 language-control speakers, 24 spontaneous speakers and
real calls; measured English clean for S1, S2, S3 and the final recipe and over the phone line for
S1, the deployed adapter and the final recipe; matched every
shortcut floor to its detector's own training data; and rebuilt the references (45, each
checked). IndicF5, Chatterbox and spontaneous speech are still open, and we say so.

**Q48. Can the system handle an English caller?**
Not with one detector: the final recipe misses English clones (25.95% EER over the phone line).
With a language gate in front (a spoken-language identifier; English goes to S1, everything else
to the Hindi-English detector), English EER on the routing test sample is 0.60% clean and 9.80%
over G.711 (25.00% for the final recipe alone on that sample), and Hindi-English
loses at most 0.4 points. The rule was chosen on development audio before any test output existed.
English spoken by our Hindi-English speakers stays with the right detector (0.4% routed to S1).
One catch, said plainly: the gate also seems to catch Tortoise, but only because Tortoise's Hindi
sounds English-accented; that is a confound, not detection (`docs/results/routing_v1.md`).

**Q49. Can P(genuine) be read as a probability?**
On the seen tool, nearly: calibration error 0.004-0.046, and SSL-AASIST stays close on Tortoise
(0.02-0.03). On the newest generators, no: the models are confidently wrong, and almost every
error at the fixed rule is a missed clone, not a false alarm (IndicF5: 73.7% of clones missed by
the final adapter, 61.7% by SSL-AASIST; calibration error 0.27-0.35 on IndicF5 and Chatterbox). The one system that raises many false alarms is the deployed adapter over the real
AMR-NB codec (30.5-66.0%) (`docs/results/calibration_v1.md`).

**Q50. How do you know IndicVoices did not leak from training into testing?**
The 200 speakers were split into pools (150 train, 20 dev, 30 test) from metadata alone, before
any audio was downloaded, and the pool file is frozen with its hash. `config_guard` reads every
training config's real manifests and refuses an `iv_test` speaker anywhere, an `iv_dev` speaker
in training, an `iv_train` speaker in dev, and any other IndicVoices speaker or a clone of one,
whatever its source column says. It runs as a test before every training launch. Clone
transcripts come only from the speaker's own pool. HiACC, the other spontaneous test set, was
never trained on, which is why its numbers are the ones we judge on.

**Q51. What if the attacker re-encodes the clone or adds noise?**
The usual sharing formats do not help an attacker: MP3, Opus, AAC and a second phone hop raise
no EER by more than 2 points, for the final adapter or either SSL-AASIST checkpoint. The weak
points are elsewhere. A reverberant room makes genuine speech look cloned (genuine clips called
fake: adapter 0.3% to 25.7%, SSL-AASIST 0.7% to 35.7%), loud background talk lets more clones
through, and a one-semitone pitch shift makes SSL-AASIST call 90-99% of genuine clips fake. The
transforms, clips and reading rule were fixed before any clip was scored
(`docs/results/robustness_v1.md`).

**Q52. IndicF5 and Chatterbox are your unseen tools. Did you then train on them?**
Two experiments did, and neither touches the paper's unseen-tool numbers. The 61 speakers those
tools cloned were split once into two frozen halves (31 and 30, hash checked by the tests). The
generator study and voice check v2 train only on the first half's clones (and that half's
genuine clips, so their voices are not "always fake"), and every number they report is computed
on the other half. `config_guard` admits those clones only in runs that declare the study, only
from the first half, and still refuses Tortoise in every run, so Tortoise stays a never-trained
check. The detectors in the paper's tables never saw either tool.

**Q53. The week-12 variant looked good. Why did you drop it?**
Because it failed a test written before its results. It had been picked after its CM04 score was
seen, so we retrained it on the final recipe with five seeds under a fixed rule: on IndicF5 it
was worse in all five seeds (34.24% against 21.68% EER), and Tortoise rose 1.81 points. The
likely reason: with 248 speakers and two codecs the final recipe already has the variety the
options added, and they made the model fit the training tool more closely (XTTS-v2 improved,
every unseen tool got worse). That is untested; the result itself is not (`variants_final_v1.md`).

**Q54. What does the voice check do with a clean file someone uploads?**
On 3,306 clean recordings it called no genuine recording with two seconds of speech cloned, and
caught 96.7% of XTTS-v2 and 66.7% of Tortoise clones, but none of the IndicF5 or Chatterbox
clones. Without the phone line's distortion those clones sit with genuine speech. The telephone
front end stays on because it was better on the development clips (97.4% against 93.2%
balanced accuracy), chosen before any test row was read (`docs/voice_check.md`).

**Q55. How can we check that a rule really came before its result?**
From the commit history. Each rule is a commit that lands before the run: the study's rule in
`a66b2a8`, voice check v2's in `cb33e57`. The training notebooks check out exactly that commit
and stop if the checked-out hash differs, so a run cannot use a rule edited afterwards. When the
first attempt at the study was lost (the AWS account was suspended before its results could be
fetched), the rerun used the same commit and the page records it.

## 10. Numbers to remember

| What | Number |
|---|---|
| English detector on English | 0.85% EER |
| Same detector, Hindi-English over the phone line | 58.42% EER |
| Genuine Hindi-English speech it calls fake (clean audio) | 94.8% |
| Trainable share with LoRA | 1.13% |
| Deployed adapter, phone line, five seeds | 3.27% +/- 1.17 |
| Unseen Tortoise over AMR-NB: deployed vs final recipe | 36.3% vs 6.0% |
| False alarms: 21 vs 248 genuine speakers | 3.40% vs 0.47% |
| SSL-AASIST on Chatterbox (final data) | 12.86% |
| Real calls, SSL-AASIST: genuine live warned / XTTS-v2 / Chatterbox "likely cloned" | 1 of 12 / 6 of 6 / 1 of 6 |
| First caution / warning on a live call | about 6 s / 10 s (deployed); 8 s / 12 s (final recipe) |
| Speakers in the language-controlled test | 183 |
| HiACC genuine called fake, frozen vs retrained with spontaneous speech (adapter / SSL-AASIST) | 43.9% vs 1.2% / 26.2% vs 1.5% |
| Spontaneous genuine against IndicF5 clones, threshold-free, SSL-AASIST frozen vs retrained | 49% vs 15-20% EER |
| Cost of that retraining on read-speech benchmarks (adapter / SSL-AASIST, seed mean) | +8-11 / +1-4 EER points |
| English over the phone line, routing test sample: final adapter alone vs behind the language gate | 25.00% vs 9.80% EER |
