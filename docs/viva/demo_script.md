# Evaluator demo script (about 8 minutes)

**Before evaluators arrive (10 minutes):** start the server (`live_call/README.md`,
section 1). Open three browser windows: `http://localhost:8000/`, `/dashboard`,
and `/call`. Click once anywhere on the dashboard so the browser allows the warning
tone. Check that the status pills read *model* and *threshold 0.500*. For the phone
part, start ngrok and confirm `/api/health` shows Twilio `configured: true`.

**Backup:** if anything fails live, the recorded run (W11-T3) and the test suite
(`tests/test_live_server.py`) show the same path.

---

## 1. The problem (1 minute) — overview page

- "Voice cloning needs seconds of audio. Scam calls in India are in Hinglish."
- "Detectors trained on English collapse on code-mixed speech: our English-trained
  model goes from under 1% EER to about 54%."
- Point at the three cards: **listen, score, warn**.

## 2. The results (2 minutes) — overview page, *Measured results*

- **Shortcut floors first.** "Before trusting any number, we checked how far eight
  simple signal statistics get on the same clips. A result only counts if it beats
  that floor: green cells do, red cells don't."
- **Three systems.** S1 English-only; S2 adds LoRA adapters (about 1% of the
  weights); S3 trains the whole encoder on code-mixed audio.
- **The finding.** "S3 is the best on the attack it trained on (0.83%), but it
  forgets English entirely: 51–56% EER, chance. S2 keeps English at 2.4%. That's why
  the deployed model is S2."
- **Honest limit.** "Over a phone line, no model reliably catches a cloning tool it
  never saw (Tortoise). We measured that and we say it."

## 3. The live system on prepared calls (2 minutes) — overview + dashboard side by side

Play in this order, each from *Play a prepared clip as a live call*:

| Play | What the dashboard does | Say |
|---|---|---|
| **Genuine caller** | stays green for 46 s, 0 of 22 windows fake-leaning | "A real held-out speaker: no false alarm." |
| **XTTS-v2 cloned voice** | amber at 6 s with the warning tone; red at 10 s | "A clone: the receiver is warned 6 seconds into the call." |
| **RVC voice conversion** | red, 24 of 24 windows | "Voice conversion keeps real prosody, and it's still caught, on speakers the model never saw." |
| **Tortoise clone** | stays green | "This is the tool we held out. The phone-line model misses it; that's the limitation we report." |

Point at the timeline: dots are 4 s windows, the line is the smoothed score, the red
dashed line is the threshold.

## 4. A live browser call (1.5 minutes) — two `/call` tabs

1. Tab A: name **Receiver**, microphone, room `demo`, Connect.
2. Tab B: name **Caller**, microphone, Connect, speak a few Hinglish sentences.
   The caller's card on the dashboard goes green.
3. Hang up Tab B. Reconnect it with **An audio file** and choose a cloned clip.
   The dashboard turns amber, then red, with the tone.

"Each participant is scored separately, so the receiver's own voice never
confuses the verdict."

## 5. A real phone call (1.5 minutes, if Twilio is live)

1. Receiver phone ready. Evaluator dials the Twilio number from their phone.
2. The receiver's phone rings; talk normally. The dashboard shows *Phone call from +91…*.
3. Play a cloned demo call into the caller's phone. The receiver hears
   *"Caution. This caller's voice may be synthetic"*; the caller hears nothing. The
   SMS arrives.

"The phone line delivers 8 kHz μ-law audio, exactly the condition the model was
trained and evaluated under. The live demo is our evaluation protocol, deployed."

---

## Questions to expect

- **How is this different from AffectDF?** AffectDF measures a generalisation failure
  on acted emotional speech. We measure it on code-mixed telephone speech, close it
  cheaply with LoRA, show native training wrecks English, and deploy the result on
  real calls with a calibrated operating point.
- **Why trust results on fakes we made ourselves?** Every number is read against a
  shortcut floor measured on the same clips; one attack tool (Tortoise) was held
  out entirely; RVC is tested on speakers disjoint from training; the anti-leakage
  rules are enforced by tests on every training config.
- **How was the threshold chosen?** On development calls only, through the live
  path, for zero false alarms on genuine callers; validated on held-out speakers.
