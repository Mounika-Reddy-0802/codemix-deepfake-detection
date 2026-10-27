# Live-call deepfake detection: running the demo

A cloned voice on a call makes the **receiver** hear a warning, the receiver's
dashboard turn red, and an SMS arrive. The caller hears nothing. The same pipeline
runs over a browser call (WebRTC) and over a real phone call (Twilio). The browser path is
verified end to end; the Twilio path is implemented and tested, but has not yet run on live
numbers, because the trial credit ran out.

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
| `lanes.py` | several detectors on one call, each the live path with its own model and threshold (`DFD_LANES`) |
| `demo_routes.py` | the final-review scenes' routes: lanes, the phone's address and QR code, results, shuffle proof, try it |
| `results_view.py` | every number the scene pages show, read from a committed result file with its source |
| `rehearsal.py` | compares every demo call's observed verdict with `configs/demo_rehearsal.json` |
| `static/` | the six scene pages and their shared look (`stage.css`, `stage.js`), overview, dashboard, handsets, console |

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

   To run the **final recipe** instead (248 speakers, both telephone codecs: no warnings on
   genuine new speakers and every Tortoise call caught on 60 s test calls), point both
   variables at it; its threshold was chosen on the dev set by the same rule
   ([live_final_recipe_v1](../docs/results/live_final_recipe_v1.md)). The pills then read
   **final_s1234_best.pt** and **threshold: 0.687**.

   ```bash
   set DFD_CHECKPOINT=<data>/checkpoints/final_s1234_best.pt
   set DFD_THRESHOLD_FILE=configs/threshold_final.yaml
   set DEMO_CLIPS_DIR=<data>/demo_clips_final
   ```

   `demo_clips_final` holds the same five calls verified on the final recipe (step 2 with
   `--checkpoint <data>/checkpoints/final_s1234_best.pt --threshold-file
   configs/threshold_final.yaml --out <data>/demo_clips_final`). Its Tortoise demo call stays
   green, as it does for the deployed model; say so before playing it.

4. Open <http://localhost:8000>. The status pills at the top must read
   **model: lora_norm_rvc_channel_best.pt**, **threshold: 0.500** and
   **Twilio: dry run** (or **live**).

The first scored window loads the model, which takes 10-20 s on CPU. After that a
window scores in under a second, faster than the 2 s hop.

### What to click

| Page | What it shows |
|---|---|
| `/` | overview, the measured results of every system (final recipe first), **try a clip** judged as a call, **play a prepared call** |
| `/demo` | the final review's guided flow: six scenes in order with the presenter's lines (section 5) |
| `/console` | both handsets, the signal path and the live analysis on one screen |
| `/dashboard` | every live call: verdict light, P(genuine) timeline against the threshold, warning tone and banner, alert log |
| `/phone/caller`, `/phone/receiver` | the two handsets on two devices; only the receiver sees a verdict |
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

Every Twilio webhook request is checked against its `X-Twilio-Signature` (unless
`TWILIO_VALIDATE=0`). The media WebSocket itself is not authenticated; anyone who knows the
public URL could open it, so the server should not be left exposed between demonstrations.

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
Demo call verification: `experiments/results/demo_clips_verification.json` (deployed
adapter) and `demo_clips_final_verification.json` (final recipe, demo calls in
`<data>/demo_clips_final`). The Tortoise demo call stays green for both models (0 and 2 of
22 windows fake-leaning), although the final recipe catches every Tortoise test call.

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

---

## 5. The final-review demo

`scripts\run_demo.bat` starts everything below on the final recipe and opens `/demo`; set
`DFD_NO_BROWSER=1` to start the server alone. It expects, under `DFD_DATA` (default
`C:\dfdata`): `checkpoints\final_s1234_best.pt`, `checkpoints\lora_norm_rvc_channel_best.pt`,
`demo_calls_v2\` (or `demo_clips_final\`) and `stage_player\`; the English-only checkpoint
comes from Git LFS (`checkpoints/baseline/best.pt`).

| Scene | Page | Routes behind it |
|---|---|---|
| 1 The threat | `/threat` | `GET /api/headline` |
| 2 Three detectors | `/lanes` | `GET /api/lanes`, `POST /api/replay_lanes/{id}` (real time, streamed to `/ws/dashboard` tagged by lane and run; `?realtime=false` returns every lane's series), `POST /api/replay_lanes_stop` |
| 3 Live call | `/live`, `/phone/receiver` | `GET /api/tunnel` (the https tunnel, else this laptop's Wi-Fi address), `GET /api/qr?text=` |
| 4 Real network | `/network` | `GET /api/real_calls`, `GET /api/stage_player`, `GET /api/stage_player/{stage}/audio` |
| 5 Why trust it | `/trust` | `POST /api/shuffle/{id}`, `GET /api/shuffle/{id}/audio`, `GET /api/headline` |
| 6 Limits + try it | `/limits` | `GET /api/limits`, `POST /api/try` |

| Variable | What it sets |
|---|---|
| `DFD_LANES` | the scene-2 detectors, `name=checkpoint=threshold_file` separated by commas; a checkpoint with `ssl` in its name loads as SSL-AASIST (`predict.load_score_fn(kind="ssl_aasist")`); `run_demo.bat` adds it as a fourth lane when its weights are present |
| `STAGE_PLAYER_DIR` | the stage player's audio (`scripts/prepare_stage_player.py`); its scores are committed in `experiments/results/stage_player_v1.json` |
| `TUNNEL_URL_FILE` | where `scripts/start_tunnel.py` writes the tunnel's https address |

**Before showing it**, every call must replay to the verdict written in
`configs/demo_rehearsal.json`, in the live model and every lane, twice in a row:

```bash
python scripts/rehearse_demo.py --server http://localhost:8000
```

**The phone.** Open the address `/live` shows (as a QR code) on the receiver's phone. Over
`python scripts/start_tunnel.py` (cloudflared, no account; or `--ngrok`) the phone gets https,
its microphone and two-way audio; over the room's Wi-Fi on plain http it joins listen-only,
still hears the caller and still gets every verdict. If the phone misbehaves, the same call
runs on one screen at `/console`.

**`POST /api/replay/{id}?realtime=false&wait=true`** plays a prepared call through the live
path as fast as it scores and returns its peak verdict and warning times: the rehearsal's
building block, and a quick way to check one call.
