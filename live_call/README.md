# Live-call deepfake detection: running the demo

A cloned voice on a call makes the **receiver** hear a warning, the receiver's
dashboard turn red, and an SMS arrive. The caller hears nothing. The same pipeline
runs over a browser call (WebRTC) and over a real phone call (Twilio).

```
caller audio ──▶ 16 kHz PCM frames ──▶ 4 s windows every 2 s ──▶ detector ──▶ verdict ladder ──▶ alerts
 WebRTC (browser)   webrtc_harness/     src/inference/streaming.py   S2 LoRA     verdict_engine.py   dashboard tone + banner
 Twilio (8 kHz μ-law) media_handler.py   silence gate, −23 dBFS       phone-line  session.py          receiver-only announcement
 file replay        replay.py            normalisation, smoothing                                    SMS, call summary
```

| Module | Role |
|---|---|
| `server.py` | FastAPI app: web pages, upload API, dashboard WebSocket, WebRTC signalling, Twilio webhooks and media stream |
| `session.py` | one monitored call: scorer → verdict engine → dashboard hub → alert dispatcher |
| `detector.py` | loads the checkpoint once; one inference thread shared by every call |
| `verdict_engine.py` | `LISTENING → GENUINE ⇄ SUSPICIOUS → LIKELY_FAKE`, with hysteresis |
| `alerts.py` | dashboard, receiver-only conference announcement, SMS; dry run without Twilio credentials |
| `media_handler.py` | Twilio Media Streams: G.711 μ-law decode, 8 → 16 kHz |
| `webrtc_harness/rtc_server.py` | aiortc signalling; one audio source per participant |
| `replay.py` | a file played as a call through the same path |
| `static/` | overview + results, receiver dashboard, WebRTC call page |

---

## 1. Run it locally (no accounts needed)

```bash
pip install -r requirements.txt
```

1. **Checkpoint.** Download `lora_norm_rvc_channel_best.pt` from the private Kaggle
   dataset `saikrishnareddy9/codemix-w10-results` into a data folder outside the repo.
2. **Demo calls** (optional, recommended): build five verified 45 s calls from
   held-out audio. Needs the phone-line bundle (`codemix-bundle-normalised`,
   `channel/`):

   ```bash
   python scripts/prepare_demo_clips.py --checkpoint <data>/checkpoints/lora_norm_rvc_channel_best.pt \
       --data-root <data>/lora_bundle_norm_ch20 --out <data>/demo_clips
   ```

3. **Start the server** (Windows `set`; use `export` on Linux/macOS):

   ```bash
   set DFD_CHECKPOINT=<data>/checkpoints/lora_norm_rvc_channel_best.pt
   set DEMO_CLIPS_DIR=<data>/demo_clips
   uvicorn live_call.server:app --host 0.0.0.0 --port 8000
   ```

4. Open <http://localhost:8000>. The status pills at the top must read
   **model: lora_norm_rvc_channel_best.pt**, **threshold: 0.500** and
   **Twilio: dry run** (or **live**).

The first scored window loads the model, which takes 10-20 s on CPU. After that a
window scores in under a second, faster than the 2 s hop.

### What to click

| Page | What it shows |
|---|---|
| `/` | overview, **try a clip** (upload or 10 s recording), **play a prepared call**, the measured S1/S2/S3 tables |
| `/dashboard` | every live call: verdict light, P(genuine) timeline against the threshold, warning tone and banner, alert log |
| `/call` | a WebRTC call. Open it in two tabs, join the same room. A participant can speak, or choose **An audio file** to play a cloned clip into the call |

Microphone access works on `localhost` or `https` only. To join from a phone or a
second laptop, expose the server with ngrok (below) and open the https URL.

---

## 2. Real phone calls with Twilio

### How a call flows

1. Someone dials the **Twilio number**. Twilio calls `POST /twilio/voice`.
2. The server answers with TwiML that **starts a Media Stream of the caller's own
   voice** (`track="inbound_track"`) to `wss://…/twilio/media`, and puts the caller
   into a conference named `call-<CallSid>`.
3. The server **dials the receiver** (`RECEIVER_NUMBER`) into the same conference.
4. Every 20 ms of caller audio arrives as 8 kHz μ-law, the telephone condition the
   deployed adapter was trained on, and is scored window by window.
5. On `SUSPICIOUS` the server asks Twilio to **announce a warning to the receiver's
   conference leg only** (tone + "Caution. This caller's voice may be synthetic").
   On `LIKELY_FAKE` it announces a stronger warning and **sends an SMS** to the
   receiver. At hang-up, an SMS summary.

Every Twilio request is checked against its `X-Twilio-Signature`.

### Setup (about 15 minutes)

1. **Twilio trial account** at <https://www.twilio.com/try-twilio>. The trial gives
   free credit and one phone number.
2. **Buy the trial number** (Console → Phone Numbers → Buy a number, voice + SMS).
3. **Verify the receiver's phone** (Console → Phone Numbers → Verified Caller IDs).
   A trial account can only call and text verified numbers.
4. **ngrok** gives the laptop a public https address:

   ```bash
   ngrok config add-authtoken <your ngrok token>
   ngrok http 8000
   ```

   Copy the `https://….ngrok-free.app` address.
5. **`.env`** in the repository root (never committed; see `.env.example`):

   ```
   TWILIO_ACCOUNT_SID=AC…
   TWILIO_AUTH_TOKEN=…
   TWILIO_NUMBER=+1…            # the Twilio number
   RECEIVER_NUMBER=+91…         # the verified phone that gets warned
   PUBLIC_BASE_URL=https://….ngrok-free.app
   DFD_CHECKPOINT=<data>/checkpoints/lora_norm_rvc_channel_best.pt
   DEMO_CLIPS_DIR=<data>/demo_clips
   ```

6. **Point the number at the server**: Console → Phone Numbers → your number →
   *A call comes in*: **Webhook**, `https://….ngrok-free.app/twilio/voice`, **HTTP POST**.
7. Start the server, then check that `/api/health` shows `"configured": true`.

### Demonstrating it

- **Receiver:** keep the verified phone ready and `/dashboard` open on the laptop.
- **Caller:** dial the Twilio number from any phone. The receiver's phone rings.
- **Genuine call:** just talk; the dashboard stays green.
- **Cloned call:** hold the caller phone to a speaker playing a demo call, or a clone
  made with the project's generation pipeline. About 6 s in, the receiver hears the
  caution, the dashboard turns amber then red, and the SMS arrives.

Trial accounts play a short Twilio notice at the start of each call; press a key
when asked. SMS to Indian numbers can take a minute on a trial account.

---

## 3. The operating threshold

`configs/threshold.yaml` holds **threshold 0.500**, alert after **2** consecutive
fake-leaning windows and a strong warning after **4**. It was chosen by
`src/inference/calibrate.py`:

- Scores come from **the exact streaming path**, not from batch evaluation. A
  threshold copied from a batch result (0.9994) called every genuine clip fake (P-029).
- The threshold was chosen on **development calls only**, as the most sensitive
  point with no false alarms on genuine 60 s calls. The window-level equal-error
  point (0.995) was rejected: it alerted on 78% of genuine calls.
- Validation on held-out speakers, clips joined into 60 s calls:

| Held-out set | Genuine calls falsely alerted | Cloned calls alerted |
|---|---:|---:|
| Seen tool (XTTS) | 0% (0 / 7) | 100% (13 / 13) |
| RVC, unseen speakers | 0% (0 / 9) | 100% (8 / 8) |
| Unseen tool (Tortoise) | 0% (0 / 15) | 33% (2 / 6) |

Full sweep and per-clip numbers: `experiments/results/threshold_calibration.json`.
Demo call verification: `experiments/results/demo_clips_verification.json`.

**Known limit, say it in the demo:** the deployed phone-line model does not reliably
catch a cloning tool it has never seen (Tortoise). That is a measured result of the
project (P-027, P-030), not a demo fault.

---

## 4. Verifying the whole path without a browser or phone

```bash
python -m pytest tests/test_live_server.py tests/test_session.py tests/test_media_handler.py -q
python -m live_call.replay <data>/demo_clips/xtts_clone.wav --checkpoint <checkpoint>
```

`test_live_server.py` drives the real app: upload, Twilio webhooks and signature
check, and a simulated Twilio Media Stream becoming a monitored call. It uses a
stand-in detector, so it needs no checkpoint.
