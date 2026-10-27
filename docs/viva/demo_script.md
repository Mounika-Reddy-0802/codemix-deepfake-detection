# Evaluator demo script (about 8 minutes)

> For the final review use the six-scene flow on the final recipe instead:
> [`final_review_demo.md`](final_review_demo.md) (`scripts\run_demo.bat`, then `/demo`).
> This script is the earlier walk-through of the overview, dashboard and call pages.

**Before evaluators arrive (10 minutes):** start the server (`live_call/README.md`,
section 1). Open three browser windows: `http://localhost:8000/`, `/dashboard`,
and `/call`. Click once anywhere on the dashboard so the browser allows the warning
tone. Check that the status pills read *model* and *threshold 0.500* (the deployed
adapter). To show the final recipe instead, start the server with `DFD_CHECKPOINT`
pointing at `final_s1234_best.pt`, `DFD_THRESHOLD_FILE=configs/threshold_final.yaml` and
`DEMO_CLIPS_DIR=<data>/demo_clips_final`; the pill then reads *threshold 0.687*. The
`demo_clips_final` folder holds the same five calls with what the final recipe does on each
(`python scripts/prepare_demo_clips.py --checkpoint <final checkpoint> --threshold-file
configs/threshold_final.yaml --data-root <data>/lora_bundle_norm_ch20 --out
<data>/demo_clips_final`), so the page never shows the deployed model's result beside the
final recipe's call.

**Backup:** if anything fails live, the recorded run (W11-T3) and the test suite
(`tests/test_live_server.py`) show the same path. Questions and answers for the viva
are in [`VIVA_QA.md`](VIVA_QA.md).

---

## 1. The problem (1 minute): overview page

- "Voice cloning needs seconds of audio. Scam calls in India come in Hindi-English."
- "A detector trained on English reaches 0.85% EER on English and 58.42% on Hindi-English
  speech over a phone line. Even on clean audio it calls 94.8% of genuine Hindi-English
  clips fake."
- "It is not the language mix: on 183 unseen speakers it rejects their genuine English at
  least as often as their mixed speech. It is the recording domain."
- Point at the three cards: **listen, score, warn**.

## 2. The results (2 minutes): overview page, *Measured results*

- **Shortcut floors first.** "Before trusting any number, we checked how far eight
  simple signal statistics get on the same clips. A result only counts if it beats
  that floor: green cells do, red cells don't."
- **Three systems.** S1 English-only; S2 adds LoRA adapters (1.13% of the weights);
  S3 trains the whole encoder on code-mixed audio.
- **The finding.** "S3 is the best on the attack it trained on (0.83%), but it
  was never trained on English and is at chance there: 51-56% EER. S2 keeps clean English at 2.4%. That's
  why the deployed model is S2."
- **What narrowed the rest.** "More genuine voices and a real mobile codec, not more attack
  tools: trained on 248 speakers over G.711 and real AMR-NB, unseen Tortoise clones over
  the mobile codec fall from 36.3% to 6.0% error, in every one of five seeds."
- **Honest limits.** "The deployed model does not reliably catch a tool it never saw over a
  phone line; the newest tools, IndicF5 and Chatterbox, stay hard; and spontaneous
  conversation still raises false alarms. We measured each of these and we say so."

## 3. The live system on prepared calls (2 minutes): overview + dashboard side by side

Play in this order, each from *Play a prepared clip as a live call* (deployed adapter):

| Play | What the dashboard does | Say |
|---|---|---|
| **Genuine caller** | stays green for 46 s, 0 of 22 windows fake-leaning | "A real held-out speaker: no false alarm." |
| **XTTS-v2 cloned voice** | amber at 6 s with the warning tone; red at 10 s | "A clone: the receiver is cautioned 6 seconds into the call and warned at 10." |
| **RVC voice conversion** | red, 24 of 24 windows | "Voice conversion keeps real prosody, and it's still caught, on speakers the model never saw." |
| **Tortoise clone** | stays green | "This is the tool we held out. The phone-line model misses it; that's the limitation we report, and the final recipe is what fixes most of it." |

With the final recipe the first four calls behave the same, and **the Tortoise call still
stays green** (2 of 22 windows fake-leaning, against 0 for the deployed model). Say so before
playing it: "Across the Tortoise test calls the final recipe catches every one (33% to 100%),
but not this speaker's call. One call is not the rate, and we keep the call that fails in
the demo." Do not swap it for a call it catches.

The *try a clip* upload judges a whole clip by its single lowest window, while a call needs
consecutive fake-leaning windows, so the same Tortoise file can read "spoof" when uploaded
and stay green as a call. If asked, that is the reason: one window is too noisy to warn a
caller on.

Point at the timeline: dots are 4 s windows, the line is the smoothed score, the red
dashed line is the threshold.

## 4. A live browser call (1.5 minutes): two `/call` tabs

1. Tab A: name **Receiver**, microphone, room `demo`, Connect.
2. Tab B: name **Caller**, microphone, Connect, speak a few Hindi-English sentences.
   The caller's card on the dashboard goes green.
3. Hang up Tab B. Reconnect it with **An audio file** and choose a cloned clip.
   The dashboard turns amber, then red, with the tone.

"Each participant is scored separately, so the receiver's own voice never
confuses the verdict."

## 5. Real phone calls (1.5 minutes)

The Twilio telephone leg is implemented and unit-tested but cannot be shown live: the
trial credit ran out. Show instead the real-call test we ran over a normal mobile network
(`docs/results/real_calls_v1.md`): genuine calls from the team and clones of our own
voices played into a call, recorded on the receiving phone and scored through the same
live path. "Over 30 real calls the deployed model warns every replayed call and every live
call of two of us, and the final recipe holds up on the seen tool, five of six clone calls
caught (not yet on Chatterbox). That is why the final recipe is the model to deploy
next." Section 7 of `VIVA_QA.md` has the numbers.

---

## Questions to expect

- **How is this different from AffectDF?** AffectDF measures a generalisation failure
  on acted emotional speech. We measure it on code-mixed telephone speech, trace it to the
  recording domain, close it cheaply with LoRA, show that full fine-tuning, never trained on English, is at chance on it,
  and deploy the result on live calls with an operating point chosen on development data.
- **Why trust results on fakes we made ourselves?** Every number is read against a
  shortcut floor measured on the same clips; three attack tools (Tortoise, IndicF5,
  Chatterbox) never enter training; RVC is tested on speakers disjoint from training; the
  anti-leakage rules are enforced by tests on every training config.
- **How was the threshold chosen?** On development calls only, through the live path:
  the most sensitive setting whose genuine 60 s development calls raise at most 5%
  alerts of any level. It was then validated on held-out speakers, never tuned on them.
