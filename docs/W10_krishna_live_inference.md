# W10: Krishna, the live-call detection path

Branch: `week10-krishna-live-inference` · Tasks: **W5-T5** (streaming), **W8-T4**
(receiver dashboard, alert layer), **W9-T4** (escalation ladder, demo), **W8-T5**
(operating threshold, demo audio), **W11-T3** (demo script)

**Outcome:** the live detection system is complete and verified end to end with the
deployed model: a web demo for evaluators, a receiver dashboard, monitored WebRTC
calls, and a Twilio phone path that warns only the receiver and sends an SMS. The
operating threshold was calibrated through the live path on development calls and
validated on held-out calls: **no false alarms on genuine calls; every XTTS and RVC
cloned call alerted; 2 of 6 Tortoise calls** (P-029, P-031).

---

## What I did

Every module below had been an empty placeholder since week 1.

### 1. `src/inference/streaming.py`: windows over a live stream

- 4 s windows every 2 s, cut from 20 ms frames. The same windows come out however
  the stream is chunked (tested).
- **Silence is skipped, not scored.** Windows below -50 dBFS carry no speech, and a
  score for them is noise.
- **Each window is normalised to -23 dBFS** before scoring, exactly as every training
  clip was (P-026). A quiet phone line would otherwise feed the model a level it
  never saw.
- Scores are smoothed by an exponential moving average (alpha 0.5).
- Consumes any `PcmFrameSource`, the shared contract from week 1.

### 2. `live_call/verdict_engine.py`: a verdict a person can act on

`LISTENING → GENUINE ⇄ SUSPICIOUS → LIKELY_FAKE`

| Rule | Default |
|---|---|
| Scored windows before any verdict | 2 |
| Consecutive fake-leaning windows to `SUSPICIOUS` (warning **beep**) | 2 |
| Consecutive fake-leaning windows to `LIKELY_FAKE` (repeat + **SMS**) | 4 |
| Consecutive windows above threshold + 0.05 to recover | 3 |

Silence neither advances nor resets a run. The margin stops a score hovering at the
threshold from flapping. The call ends with a summary: peak verdict, fake-leaning
windows and worst score. `alerts.py` will deliver these events; the engine only
decides.

**It will not start without a threshold.** `configs/threshold.yaml` is set from the
DET curve at the results freeze (W8-T5). 0.5 is never a fallback: at 0.5 the clean
adapter passes 47.2% of Tortoise fakes (P-025).

### 3. `src/inference/predict.py`: one verdict per file

This loads any checkpoint (S1, S2 LoRA, S3) and scores a file through the same
streaming path. The file score is the **worst window**, so a clip with ten cloned
seconds inside a minute of real speech is not averaged away. Threshold precedence:
explicit value, then the frozen file, then a results JSON's EER point (always
labelled provisional).

### 4. `live_call/replay.py`: a repeatable call

The harness needs two browsers and a microphone, and Twilio needs an account.
Neither can be rerun the night before the viva. Replay plays a file as a call
through the same `PcmFrameSource`, scorer and verdict engine, and prints the
timeline a receiver would see. It is also how the W8-T5 demo clips get verified.

Tests: 36 new ones across `test_streaming.py`, `test_verdict_engine.py`,
`test_predict.py` and `test_replay.py`, all without torch, so they run in CI.

---

## What the real-audio check found (P-029)

This used the phone-line adapter `checkpoints/lora_codemix_channel/best.pt`, on CPU,
with 3 real and 3 cloned clips from the phone-line eval pool.

**1. The model separates the clips; the borrowed threshold does not.** Real clips
scored 0.987-0.997 and clones 0.0002. But the only threshold available was the EER
point of a batch scoring run, 0.9994, so all six were called fake. The scores sit
against 1.0, so a different crop or normalisation moves a real clip across a
threshold that close to the edge.

**2. Windows inside one clip disagree.** Replaying one 15 s clone gave windows of
0.0004, 0.0002, **0.81, 0.99, 0.99**, 0.13. The call reached `SUSPICIOUS` and never
`LIKELY_FAKE`, because the middle of the clone reads as real. Every EER in the
results documents scores **one deterministic 4 s crop per clip**, so they do not
predict what the live path does across a whole call.

**Consequence for W8-T5:** the operating threshold must be chosen from **window-level
scores produced by this streaming code**, on the frozen checkpoint, over whole clips.
Copying a threshold from a results JSON is not enough. The demo clips should be
checked with `live_call.replay`, not with `evaluate`.

Caveat: that checkpoint predates normalisation and was run on normalised audio, so
the scores themselves are not quotable. The two mechanisms above do not depend on
which checkpoint is used.

---

## The full system (second half of the week)

### Server, pages and calls

| Piece | What it does |
|---|---|
| `live_call/server.py` | one FastAPI app: overview + results page, upload API, receiver dashboard over a WebSocket, WebRTC signalling, Twilio webhooks and Media Stream |
| `live_call/session.py` | a monitored call: scorer → verdict engine → dashboard hub → alerts; a failing alert channel never stops monitoring |
| `live_call/detector.py` | the checkpoint loads once; one inference thread serves every call, so model time never blocks audio intake |
| `live_call/media_handler.py` | Twilio's 8 kHz G.711 μ-law to 16 kHz. Decoding matches Python's reference codec exactly on all 256 codes; the half-band upsampler keeps a 1 kHz tone and puts its image 55 dB down |
| `live_call/alerts.py` | dashboard tone; a warning announced to the **receiver's conference leg only**; SMS; dry run without Twilio credentials |
| `webrtc_harness/rtc_server.py` | now gives each participant their own audio source. The shared room queue mixed both voices frame by frame, which is neither speaker |
| `static/` | overview with the measured S1/S2/S3 tables, try-a-clip (upload or record), prepared calls, dashboard, WebRTC call page that can play a clip into the call |

The warning tone is built in memory. No audio file is shipped.

### The operating threshold (P-031)

`src/inference/calibrate.py` scored 506 dev clips and 300 clips each of the eval
pool, CM04 and the RVC test half through the exact live path, then chose the
threshold on dev only.

Two things surfaced and were fixed before anything was built on them:

1. **The verdict engine could never recover** at a calibrated threshold. The
   recovery bound was threshold + 0.05, above 1 for a threshold of 0.995. The margin
   is now a fraction of the gap to 1, with a test at 0.995.
2. **The window-level equal-error point is the wrong operating point for calls.** At
   0.995, 8% of genuine windows score low, so over a 60 s call two low windows in a
   row are likely: it alerted on **78% of genuine dev calls**. Clips were stitched
   into 30 s and 60 s calls and candidate thresholds swept; the chosen point is the
   most sensitive one with no false alarms on genuine dev calls: **0.500**, alert
   after 2 fake-leaning windows, strong warning after 4.

Held-out validation, 60 s calls:

| Set | Genuine calls alerted | Cloned calls alerted |
|---|---:|---:|
| Seen tool (XTTS) | 0 / 7 | 13 / 13 |
| RVC, unseen speakers | 0 / 9 | 8 / 8 |
| Unseen tool (Tortoise) | 0 / 15 | 2 / 6 |

### Demo calls, verified

`scripts/prepare_demo_clips.py` joins clips of one held-out speaker into ~45 s calls
and replays each through the live path. Results are recorded as observed, misses
included:

| Call | Fake-leaning windows | Result |
|---|---:|---|
| Genuine caller | 0 / 22 | no alert ✅ |
| XTTS-v2 clone | 23 / 23 | likely cloned ✅ |
| RVC conversion | 24 / 24 | likely cloned ✅ |
| Genuine caller (RVC source) | 0 / 23 | no alert ✅ |
| Tortoise clone (unseen tool) | 0 / 22 | missed ❌ (the known limit) |

### End to end, with the real model

The running server was driven the way an evaluator would use it:

- **Upload:** genuine call → genuine (lowest window 0.994); clone → cloned (0.0002).
- **Prepared call in real time:** 23 windows to the dashboard, then beep → warning → summary.
- **Twilio Media Stream** (the RVC call sent as 8 kHz μ-law in Twilio's JSON
  protocol): 24 windows, beep → warning → summary. The warnings were routed to the
  receiver's leg and to SMS, marked dry run because no account is configured.
- **WebRTC** (an aiortc peer playing the XTTS call into room `e2e`): its own session,
  scores 0.0002, **warning tone at 6 s, strong warning at 10 s**.

Tests: 50 new ones (`test_session.py`, `test_media_handler.py`, `test_live_server.py`,
`test_calibrate.py`, plus additions to the streaming and verdict tests). The server
tests use a stand-in detector, so they need no checkpoint.

## What is left

- **Activate the Twilio trial** and run the phone path on real numbers
  (`live_call/README.md`, section 2). Everything up to the REST calls is tested;
  the calls themselves need an account.
- Record the backup screen capture of a full run (W11-T3).
- The threshold rests on 18 genuine and 21 cloned dev calls from 2 speakers; more
  development speakers would tighten it.
