# Real phone calls: a pilot over a mobile network

**Status: all 30 planned calls**, from three speakers (P1, P2, P3). Numbers are read from
`experiments/results/real_calls_v1/` (per-window scores, per-call verdicts and `summary.json`,
written by `python -m src.inference.real_calls report`) and
`experiments/results/real_calls_v1/codec_chain.csv` (`python -m src.inference.real_call_codecs`).
Protocol, what happened on the day, and consent: `docs/real_calls/PROTOCOL.md`,
`docs/ethics/real_call_consent_form.md`.

## What was done

Every earlier result passes audio through a simulated line or a reference codec offline. Here
the team called each other over a commercial mobile network and the receiving handset's call
recorder kept each call (8 kHz mono AAC, 49-88 s). Three kinds of call:

| Kind | Calls | What reaches the phone |
|---|---|---|
| Genuine, live | C01-C12 | the speaker on a normal call: Script A, Script B, or about a minute of free talk |
| Genuine, replayed | C13-C18 | the speaker's own reading of Script A or B, played from a laptop speaker into the calling phone |
| Clone, replayed | C19-C30 | XTTS-v2 (seen tool) or Chatterbox (unseen) clones of the same speaker reading the same script, played the same way |

The clones were made on Kaggle from about 20 s of each speaker's reference reading (XTTS-v2 69
of 72 sentences and Chatterbox 70 of 72 passed the quality screen; the five that stalled were
left out, so five playback calls hold 11 of 12 sentences). Genuine and cloned replays share
the loudspeaker path, so comparing them isolates the synthesis; the live calls show what a real
caller meets. Before scoring, every replay recording was matched back to the file played into
it (`python -m src.inference.real_calls check`): the speech-activity tracks of the recording and
its planned file correlate at 0.54-0.95, and no other playback file reaches more than 0.40, so
all 18 replay calls carry the right label (`replay_check.csv`). Each recording is scored
through the live path (4 s windows every 2 s, smoothed, through the warning ladder) at each
model's frozen operating point, as recorded and after a G.711 round trip (what the live
system's carrier delivers). No threshold was chosen on these calls.

Three models are scored: the deployed adapter, the final recipe, and SSL-AASIST, the larger
encoder (XLS-R 300M) fine-tuned on the final recipe's data. SSL-AASIST's operating point was
chosen the same way as the final recipe's, by the same rule on the same 506 dev clips, before
any call was scored with it: threshold 0.580, a caution after two fake-leaning windows and
"likely cloned" after four (`configs/threshold_ssl_aasist.yaml`; the final recipe's rule picked
three and five for it).

## Results

As the carrier delivers the calls (G.711):

| Calls | n | Deployed: likely cloned | Final: warned (caution or worse) | Final: likely cloned | SSL-AASIST: warned | SSL-AASIST: likely cloned |
|---|---|---|---|---|---|---|
| Genuine, live | 12 | 8 | 3 | 1 | 1 | 0 |
| Genuine, replayed | 6 | 6 | 1 | 1 | 1 | 1 |
| XTTS-v2 clones, replayed | 6 | 6 | 5 | 5 | 6 | 6 |
| Chatterbox clones, replayed | 6 | 6 | 1 | 0 | 3 | 1 |

Window-level EER, clones against the genuine replays that share their path: final recipe
XTTS-v2 19.0%, Chatterbox 50.0% (chance); SSL-AASIST 4.4% and 36.6%. As recorded, the verdicts
are close to these (final: genuine live 4 of 12 warned, 2 likely cloned; SSL-AASIST: none of
the 12 warned, and 2 of 6 Chatterbox calls likely cloned).

**The deployed model fails for two speakers of three.** It takes every replayed call, genuine
or cloned, to "likely cloned" (18 of 18), and every live call of P2 and P3 (mean window scores
0.02-0.22 for P2, below 0.001 for P3), but none of P1's four live calls, which it scores at
0.96-1.00. Even P1's own replayed readings score 0.0002, so for the deployed model the
loudspeaker path alone is enough to fail a speaker it accepts live.

**The final recipe holds up on the seen tool, not on the unseen one.** Mean score on genuine
live calls 0.78-1.00 over G.711; five of six XTTS-v2 clone calls caught (taken to "likely
cloned"); no Chatterbox call caught (one cautioned), which matches Chatterbox being the
adapter's hardest unseen tool offline (32.0% EER for this adapter). Against the genuine replays,
the fair control, one of six (a replay of P3's reading) reaches "likely cloned".

**Over G.711, the warnings on genuine live calls are one speaker's.** Per speaker (`by_speaker`
in the report), the three warned genuine live calls are all P2's (3 of 4; one reaches "likely
cloned"), and none of P1's or P3's eight is warned; as recorded, one of P3's is also
cautioned. The XTTS-v2 call that was not caught and
the one cautioned Chatterbox call are also P2's. With three speakers this says the rates above
depend on who is calling, not what they are for a population.

It is not P2's voice alone. Through the emulated chain, P2's direct readings stay at 0.82-1.00
for the final recipe (P1: 1.00, P3: 0.93-1.00), and P2's replayed readings score 1.00, yet P2's
live calls average 0.88 with the lowest at 0.78. P2's live calls came from P2's own phone and
network, P2's replays from the phone beside the laptop, so speaker and calling handset change
together here and this pilot cannot separate them.

**The larger encoder does better than the final recipe on three rows and level on the
fourth.** Over G.711, SSL-AASIST
warns one of the twelve genuine live calls (C08, P2's, a caution 48 s in; none as recorded)
and takes none to "likely cloned"; its mean score per genuine live call is 0.91-1.00, against
0.78-1.00 for the final recipe. It takes all six XTTS-v2 calls to "likely cloned", P1's and
P2's four 10-14 s into the call and P3's two at 28 and 32 s. Of the six Chatterbox calls it
takes one to "likely cloned" (C28, P2's, at 16 s) and cautions two (both P3's); P1's two pass.
Against the genuine replays, one of six reaches "likely cloned" (C18, P3's, 44 s in; for the
final recipe it is C17, also P3's). It shows no early-call dip: genuine live calls average 0.98
over their first 10 s. Its ladder is shorter (two and four windows, against three and five),
which makes it faster and, by itself, more prone to false alarms, yet it warns fewer genuine
calls. This is one checkpoint, and not a strong one offline: it is a retraining of seed 1234
with the week-13 settings, kept so it could be run here, and GPU training does not repeat
exactly, so its offline Chatterbox EER is 21.2% where the five week-13 trainings range 8.2-18.3%
(`experiments/results/ssl_checkpoint_v1/`). Chatterbox stays far from solved on real calls
(window EER 36.6%). On the held-out 60 s calls that validate the operating points
(`experiments/results/threshold_calibration_ssl_aasist.json`), the same checkpoint warns no genuine
read-speech call, catches every CM04 Tortoise call, and warns 11 of 41 genuine spontaneous
HiACC calls (one "likely cloned"), where the final recipe warns all 41 (40 "likely cloned").
Two earlier trainings of this encoder disagreed on spontaneous speech (5.8% and 46.7% of
clips called fake; this one scores 6.4% of its 1,000 HiACC clips below 0.5, the final recipe
44.5%), so a second checkpoint has to show the same before it replaces the final recipe.

**The final recipe's false alarms come early in a call.** Mean window score of the final recipe by time in the call:

| Calls | 0-10 s | 10-20 s | after 20 s |
|---|---|---|---|
| Genuine, live | 0.788 | 0.911 | 0.982 |
| Genuine, replayed | 0.924 | 0.900 | 0.960 |
| XTTS-v2 clones | 0.591 | 0.560 | 0.411 |
| Chatterbox clones | 0.897 | 0.967 | 0.954 |

Genuine live calls look least genuine in their first ten seconds. The first window spans the
quiet before the first word: its level is a median -30 dBFS (-34 to -23 where it is scored) against -21 to -26
dBFS for the speech that follows; in eleven of the twelve calls it passes the -50 dBFS silence
gate, is normalised and is scored as if it were speech, at a mean of 0.60 against 0.96 for the
later windows. Loudness itself does not predict the score over the whole call (rank correlation
0.00 over 352 windows). P1's XTTS-v2 clones are caught early (caution at 8 s, "likely cloned" at
12 and 14 s), but the three caught calls of P2 and P3 reach a caution 18-32 s into the call and
"likely cloned" 30-46 s in (`calls_final_g711.csv`), where the final recipe's five-window ladder
could reach "likely cloned" after 12 s. For two speakers of three the warning is late on real calls, and a scam
call could be well under way by then.

The quiet start is part of the story, not all of it. In every warned genuine live call, in both
conditions, the run of fake-leaning windows that raised the warning starts at the first window.
But replaying the calls with that window treated as silence (`first_window_skipped`, a
diagnostic in `summary.json` that chooses nothing) clears two of the three warned genuine live
calls over G.711; as recorded it clears one of the four (C10) but cautions another (C07), so
the count stays at four. One call over G.711 (C05; two as recorded) still reaches "likely
cloned", its run including two speech windows 8-14 s into the call (-22 dBFS, scoring 0.01). A stricter
speech gate for the first seconds of a call would help but not suffice; it and the operating
point should be set together on a separate set of real development calls, not on these. Set on
the dev clips alone, such a gate clears two of the three over G.711 but warns one call more as
recorded, delays every model's warnings and silences most early speech, because these calls are
far quieter than the normalised dev clips (`early_warning_v1.md`).

## Why the deployed model fails: the call chain, one stage at a time

Each speaker's direct reading of Script A and B (recorded on a phone, never sent over a network)
passed through the stages of a call and scored with both models; mean P(genuine) over the six
readings:

| Stage | Deployed | Final recipe |
|---|---|---|
| Direct recording | 0.830 | 0.983 |
| Simulated G.711 line | 0.752 | 0.971 |
| Real AMR-NB codec, 12.2 kbit/s | 0.500 | 0.963 |
| AMR-NB, then the call recorder (8 kHz AAC) | 0.485 | 0.939 |
| AMR-WB (4G voice), then the call recorder | 0.524 | 0.969 |
| The genuine live calls, as the carrier delivers them | 0.377 | 0.943 |

On average the real mobile codec takes the deployed model to its threshold; it only ever heard
the simulated G.711 line, and the call recorder adds little. Per reading, two of the six cross
the threshold at AMR-NB (P2 0.66 to 0.28, P3 0.74 to 0.11), one stays above it (P2 0.92 to
0.74), one is below it before any codec (P3 0.28), and P1's two stay at 0.87-1.00 at every
stage, so the codec is the largest single step for the speakers the model fails on, not a
failure of every voice. The final recipe, trained on AMR-NB, stays at 0.82-1.00 through every
stage for all six readings. The real calls cost the deployed model more still (0.38 on the
genuine live calls), so the network and the handset add damage beyond the codec, but the
direction is the one the offline AMR-NB results predicted. The final recipe also trains on more
voices, so these calls cannot say which of its two changes keeps it up; offline, adding AMR-NB
alone to the deployed recipe's training recovers most of the codec loss (new-speaker XTTS over
AMR-NB 7.59% against 22.37%, three seeds; `experiments/results/mixcodec_v1/`). The call-chain
figure is `experiments/figures/real_call_chain.pdf` (also `paper/figures/`).

## Limitations

- 30 calls, three speakers, one network: a pilot. A percentage over six calls moves in steps of
  17 points and over twelve in steps of 8, and its 95% interval is wide: five of six XTTS-v2
  calls caught is [44, 97]%, none of six Chatterbox calls is [0, 39]%, one of twelve genuine
  live calls "likely cloned" is [1.5, 35]% (Wilson intervals, in the report). Only the deployed
  model's failure on every replayed call and on two speakers' live calls is firm at this size.
- Speaker, session and handset are confounded. P1's calls were made a day later, P1's replays
  are about 9-10 dB louder than the other twelve, and two phones recorded the calls (P3's phone for
  C05-C08, P2's for C09-C12 and C15-C30; see the protocol).
- The speakers are the authors, not the corpus speakers, and read prepared scripts in half the
  live calls.
- Replayed calls add a loudspeaker and a room; the genuine replays are the control for that, so
  the fair comparison for the clones is against them.
- The calls were recorded on a handset and scored offline; the live system's own telephone
  carrier was not exercised.
- References and recordings came from different phones (each speaker's references on their own
  phone, in three formats), which a fixed protocol would avoid.
- SSL-AASIST is one checkpoint at one operating point. The call chain (next section) was rebuilt
  for the deployed model and the final recipe only.

## Reproduce

```
python -m src.inference.real_calls check --recordings data/real_calls/recordings --playback data/real_calls/playback --out experiments/results/real_calls_v1/replay_check.csv
python -m src.inference.real_calls prepare --recordings data/real_calls/recordings --out data/real_calls/scored
python -m src.inference.real_calls score --audio data/real_calls/scored --deployed <deployed ckpt> --final <final_s1234 ckpt> --out experiments/results/real_calls_v1
# SSL-AASIST: its operating point on the dev clips first (same rule), then the calls
python -m src.inference.calibrate score --kind ssl_aasist --checkpoint <ssl_aasist_final_s1234_fp16.pt> --manifest <dev clips> --data-root <dev audio> --out experiments/calibration_ssl/dev_windows.csv
python -m src.inference.calibrate choose --dev experiments/calibration_ssl/dev_windows.csv --checkpoint-name ssl_aasist_final_s1234 --out configs/threshold_ssl_aasist.yaml --report experiments/results/threshold_calibration_ssl_aasist.json
python -m src.inference.real_calls score --audio data/real_calls/scored --ssl <ssl_aasist_final_s1234_fp16.pt> --out experiments/results/real_calls_v1
python -m src.inference.real_calls report --audio data/real_calls/scored --windows experiments/results/real_calls_v1 --md experiments/results/real_calls_v1/report.md
python -m src.inference.real_call_codecs --references data/real_calls/references --speakers P1,P2,P3 --deployed <deployed ckpt> --final <final_s1234 ckpt> --out experiments/results/real_calls_v1/codec_chain.csv
python -m src.reporting.real_call_figure   # the call-chain figure, then copy the .pdf to paper/figures/
```

The recordings and clones are never committed (`data/` is ignored) and are deleted after the
project, as the consent form promises; only scores and verdicts are.
