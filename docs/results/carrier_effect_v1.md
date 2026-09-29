# Carrier effect v1 — the codec does not explain the live-call score drop

**Owner SK, Week 12.** Reproduce with
`python -m src.inference.carrier_sim --clips <demo clips>`; numbers in
`experiments/results/carrier_effect.json`.

## The claim being tested

On a live WebRTC call, a genuine caller's score dips: in the week-11 check, one
genuine clip scored 0.001–0.087 on three consecutive windows, while offline the same
clip scores 0.993 or better on every window. The explanation written into the week-11
record — **and into the draft system-design chapter** — was the codec: the browser
carries the call with Opus, the detector is trained on G.711 telephone audio, and the
cue it uses is phase-level, so a low-bitrate encode of a real voice might look
synthetic.

That was plausible and it was never tested. This is the test.

## Method

Each demo clip is scored through the identical streaming path the live system uses
(4 s window, 2 s hop, −50 dBFS silence gate, −23 dBFS normalisation), three times:

| Carrier | What it does |
|---|---|
| `none` | control: the clip as the evaluation scores it |
| `opus_48k` | the browser path — resample 16 kHz → 48 kHz, Opus encode and decode, resample back |
| `g711_20db` | the telephone path the model was trained on |

The resampling is included deliberately. A browser sends 48 kHz Opus and the server
resamples to the 16 kHz the model expects, so simulating the codec alone would leave
out half of what the carrier does.

## Results

Windows below the 0.500 threshold, out of all scored windows:

| Clip | Truth | none | opus_48k | g711_20db | min score under Opus |
|---|---|---:|---:|---:|---:|
| Genuine caller | real | 0 / 22 | **0 / 22** | 0 / 22 | 0.984 |
| Genuine caller (RVC source) | real | 0 / 23 | **0 / 23** | 0 / 23 | 0.996 |
| XTTS-v2 clone | cloned | 23 / 23 | 23 / 23 | 23 / 23 | 0.000 |
| RVC conversion | cloned | 24 / 24 | 24 / 24 | 22 / 24 | 0.000 |
| Tortoise clone (unseen) | cloned | 0 / 22 | 1 / 22 | 0 / 22 | 0.058 |

## What this says

**The codec explanation is wrong.** Neither genuine clip loses a single window to
the Opus round-trip, and the worst any window gets is 0.984 — nowhere near the 0.001
seen on the live call. Encoding and resampling are not what degrades a genuine
caller in the demo.

**Clone detection is carrier-robust.** Both seen attacks stay fully detected through
every carrier, which is worth stating in its own right: the decision does not depend
on codec artefacts, which is the sort of shortcut a reviewer is right to suspect.

**So what does cause the live drop?** By elimination it is in the real-time
transport, not the coding: jitter buffering, packet-loss concealment, or the
bitrate adaptation a browser applies while a connection is still settling. Packet
loss concealment is the most interesting candidate, because it *synthesises* audio
to fill gaps — a detector correctly reporting that machine-generated audio is not a
real human would produce exactly this signature, early in the call, in bursts.

That remains a hypothesis. It is not tested here, and it should not be stated as
fact anywhere until it is.

## Honest limitations

- **Five clips.** Enough to refute a causal claim, not enough to characterise one.
- **The Opus encode is libsndfile's default**, not a browser's adaptive encoder.
  A browser at a poor bitrate under congestion may behave differently; that is
  precisely the untested part.
- **No live-path measurement in this document.** Correlating window scores against
  RTP loss and jitter statistics is the experiment that would settle it. It was
  attempted and the browser was killed twice by this machine running out of memory.
- The Tortoise clip losing one window under Opus is a single window on a clip the
  system misses anyway. It is not evidence of anything.

## Consequences

1. `paper/sections/system.tex` said the codec explained the drop. Corrected.
2. `docs/W11_krishna_phone_demo.md` said the same. Corrected, with a pointer here.
3. The high-bitrate, DTX-off negotiation added in week 11 stays — it is sensible for
   a demo regardless — but it must no longer be described as the fix for this, since
   the dip survived it.
4. The operating point still needs calibrating on real live calls rather than on a
   simulated carrier. Simulating the carrier offline, as shown here, would calibrate
   against a transform that does not reproduce the problem.
