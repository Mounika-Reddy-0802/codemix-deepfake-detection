# W10: Krishna, the live-call detection path

Branch: `week10-krishna-live-inference` · Tasks: **W5-T5** (streaming), **W8-T4**
(verdict and alert logic), **W9-T4** (escalation ladder), groundwork for **W8-T5**

**Outcome:** a call can now be scored from end to end. Audio in, a stable verdict
out, escalating warnings, and a post-call summary. It runs the same way over the
WebRTC harness, over Twilio once it is activated, and over a replayed file. It was
checked on real clips with a real checkpoint. That check produced two findings
that change how the demo threshold has to be set (P-029).

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

## What is left

- `alerts.py`: deliver beep, SMS and summary (in-call audio needs Twilio).
- Wire the scorer and engine into the harness, plus the receiver dashboard
  (`live_call/static/`).
- Twilio activation, then `server.py` and `media_handler.py` (8 kHz mu-law in).
- Threshold from window-level scores once the checkpoint is frozen (with L, W8-T5).
