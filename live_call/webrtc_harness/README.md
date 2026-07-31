# WebRTC dev harness (Weeks 1–5)

Free, no-expiry, browser ↔ server call for building the live-call pipeline before
Twilio is ever activated. The server (`rtc_server.py`, aiortc + FastAPI) receives
each browser's mic audio, resamples it to **16 kHz mono PCM**, and exposes it as a
`PcmFrameSource` — the **same interface** `live_call/media_handler.py` implements
for Twilio Media Streams in Week 6. Streaming-inference and alert code written
against this harness ports to Twilio unchanged.

```
browser (call.html) ──WebRTC──▶ rtc_server.py ──▶ resample 16 kHz mono
                                      │             ──▶ PcmFrameSource.frames()
                                      └──▶ (best-effort relay to the other peer)
```

## Run

```bash
pip install -r requirements.txt          # needs aiortc, av, fastapi, uvicorn
uvicorn live_call.webrtc_harness.rtc_server:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000/> in **two browser tabs**, put the same room name in
both, grant mic permission, and click **Connect**. The mic-level meter confirms
audio is captured; the server logs show tracks arriving and (via `log_levels`)
the RMS of the tapped 16 kHz stream.

For an external device or the Twilio-style webhook test later, expose the port
with ngrok: `ngrok http 8000`.

## The shared contract (why this matters)

`live_call/audio_source.py` defines `PcmFrame` (16 kHz mono s16) and the
`PcmFrameSource` protocol. Both backends implement it:

| Backend | Module | Weeks | Audio in |
|---------|--------|-------|----------|
| WebRTC harness | `webrtc_harness/rtc_server.py` | 1–5 (+ viva fallback) | browser mic via WebRTC |
| Twilio Media Streams | `media_handler.py` | 6+ | 8 kHz μ-law from the phone call |

Downstream (`inference/streaming.py`, `verdict_engine.py`, `alerts.py`) depends
only on `PcmFrameSource`, so it never changes when the backend does.

## Known limitation (documented on purpose)

Two-way audio uses a **best-effort** relay with **no renegotiation**: a joiner
hears a participant already in the room, but the earlier participant won't hear a
later joiner until they reconnect. This is fine for the harness — the server-side
**tap** (the part reused for Twilio + inference) works for every participant
regardless. Full mutual audio (websocket signaling + renegotiation) is polish for
later; it isn't on the research critical path.
