# Live-call detection system — design note

The wow-factor demo: a real phone call in which a **cloned voice triggers a beep in
the receiver's ear**, a live dashboard alert, and SMS warnings. This note is the
integration design for the **Twilio Conference + Media Streams** path (Week 6+).
Weeks 1–5 develop everything for free on the WebRTC harness
(`webrtc_harness/`) — this document is the plan we execute once Twilio is on.

> **The whole point:** Twilio delivers **8 kHz narrowband μ-law telephony audio** —
> the exact condition our channel-matched evaluation protocol trains and tests
> under. **The live demo is our evaluation protocol, deployed.**

---

## Architecture

```
Caller (browser dialer,        Twilio Cloud                     Your FastAPI server
Twilio Voice JS SDK) ─PSTN─▶   US number → CONFERENCE  ──ws──▶  media_handler.py
                               (caller + receiver legs)         8 kHz μ-law frames
Receiver ◀──────────────────────┘       ▲                       → decode → PcmFrame
(phone / browser)                       │                       → 4 s rolling window
                                        │ Participant API:      → ONNX detector
                                        │ play warning_beep.wav → verdict_engine
                                        │ to RECEIVER ONLY ◀───────┤ (smoothing +
                                        │ (Alert layer 1)          │  escalation)
                                                                   ├─ ws ▶ Dashboard 🟢🟡🔴
                                                                   │        (Alert layer 2)
                                                                   ├─ SMS on sustained
                                                                   │  detection (layer 3)
                                                                   └─ post-call summary
                                                                      SMS + log (layer 4)
```

### Call flow

1. **Caller** opens `static/dialer.html` (Twilio Voice JS SDK) and dials the
   trial US number. Browser dialer → US number sidesteps Indian-number issuance
   and international charges.
2. Twilio hits the **inbound-call webhook** (`POST /voice` on our server). We
   return TwiML that puts the caller into a **`<Conference>`** and adds a
   **`<Start><Stream>`** tap that streams the mixed call audio to our websocket.
3. The **receiver** leg is dialed into the same Conference (verified teammate
   number on the trial).
4. **Media Streams** pushes base64 **8 kHz μ-law** frames over the websocket
   (`/media`). `media_handler.py` μ-law-decodes them, resamples to **16 kHz
   mono**, and emits **`PcmFrame`s** — implementing the exact **`PcmFrameSource`**
   contract (`live_call/audio_source.py`) the WebRTC harness already proved. So
   `inference/streaming.py` and `verdict_engine.py` are reused **unchanged**.
5. **Streaming inference**: rolling 4 s window / 2 s hop → ONNX int8 detector →
   per-window probability → exponential smoothing → `verdict_engine` state
   (Real ✅ / Suspicious ⚠️ / Cloned 🚨).

### The four alert layers (escalation ladder in `verdict_engine`)

| Layer | Trigger | Channel | Module |
|-------|---------|---------|--------|
| 1 — in-call audio (**primary**) | verdict crosses threshold | beep/whisper to **receiver only** via Conference **Participant API** (`hold`+`announce` / play URL) | `alerts.py` |
| 2 — dashboard | every verdict update | websocket → `static/dashboard.html` | `server.py` |
| 3 — escalation SMS | high confidence sustained 10+ s | Twilio SMS to receiver | `alerts.py` |
| 4 — post-call summary | call end, **always** | SMS + dashboard log | `alerts.py` |

**Design rules (non-negotiable):** never auto-disconnect the call; **never alert
the caller** (a false positive on a genuine caller is harmful, and warning a real
fraudster teaches evasion). Warn the potential victim, log everything, let the
human decide.

---

## Twilio console + local setup (Week 6, not before)

1. **ngrok** exposes the local server to Twilio webhooks:
   `ngrok http 8000` → note the `https://…ngrok-free.app` URL → set
   `PUBLIC_BASE_URL` in `.env`.
2. **Buy/confirm** the trial US number (voice + SMS capable).
3. **Voice webhook:** number → *A call comes in* → Webhook →
   `https://<public>/voice` (HTTP POST).
4. **Media Streams** are started from the TwiML we return (`<Start><Stream
   url="wss://<public>/media"/>`), not the console.
5. **Verified caller IDs:** add every teammate phone as a **Verified Caller ID**
   (trial calls/SMS only reach verified numbers).
6. **Voice SDK dialer:** create a TwiML App + API Key/Secret for
   `static/dialer.html`; put the SIDs in `.env`.

### Environment variables (see `.env.example`)

`TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_NUMBER`,
`TWILIO_TWIML_APP_SID`, `TWILIO_API_KEY_SID`, `TWILIO_API_KEY_SECRET`,
`RECEIVER_NUMBER`, `PUBLIC_BASE_URL`, `NGROK_AUTHTOKEN`. Secrets live only in
`.env` (gitignored) — never in code or history.

---

## Budget & trial timing

- Trial = **75 free voice minutes + 100 SMS**, **expires 30 days after signup**,
  verified recipients only.
- Activating in **Week 6** puts the 30-day window over **Weeks 6–9** — the entire
  integration + demo-recording phase.
- Keep dev calls **short**; track the 75-minute budget from day one.
- If the viva falls **after** the window, upgrade pay-as-you-go from the ₹1–2k
  budget in **Week 8** (US number ~$1.15/mo + ~$0.0085/min inbound + cents/SMS).
- Fallbacks if Twilio fails at the viva: the **WebRTC harness** (same detection
  pipeline, zero cost, live) and the **Gradio** upload demo.
- Production caveat for the paper: SMS *into* India needs a **DLT-registered
  sender**; the trial path is for the demo, not production.

---

## ⚠️ **ACTIVATE THE TWILIO TRIAL IN WEEK 6, NOT BEFORE.**

Creating or activating the trial early burns the 30-day window before the demo
phase. Until Week 6, **all** streaming-inference and alert work happens on the
free `webrtc_harness/`. This is a hard project rule (see `CLAUDE.md` §8).
