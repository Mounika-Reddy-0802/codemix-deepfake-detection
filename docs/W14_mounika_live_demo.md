# Week 14 — Mounika — the live demo for the final review (Track B)

The final review needs a demo that shows a complete research system — the gap, the cheap
fix, the better fix, a real-call pilot and the limits with their numbers — and that cannot
misfire in front of the mentor and the external reviewer. Track B of the review plan built
it on the final recipe: six scenes in a guided flow, every number read from a committed
result file, every call checked against a written-down verdict before it is shown.

## What I did, and why it is built this way

**A server on the final recipe, and a health check that says so.** `/api/health` now names
the threshold file (`configs/threshold_final.yaml`, forward slashes on every machine) and the
warning ladder (caution after 3 low windows, likely cloned after 5), because a server started
in the wrong terminal silently falls back to the deployed model's 0.500. The pages read the
ladder from there instead of printing "2 and 4", which is the deployed model's ladder.

**The rehearsal check (B1).** `configs/demo_rehearsal.json` writes down, for each of the eight
demo calls, the verdict the live model and each detector lane must reach, with the time limits
the presenter's lines depend on. The expected verdict is what the demo will show, not the
ground truth: the two known-limit calls (a Tortoise clone the final recipe lets through, a
genuine spontaneous caller it warns) are kept and expected to fail, never swapped for calls
that look better. `POST /api/replay/{id}?realtime=false&wait=true` plays a call through the
exact live path as fast as it scores, so a full check of all eight calls in four detectors
takes about two and a half minutes. `scripts/rehearse_demo.py` runs it against a running
server and prints the table; `tests/test_demo_rehearsal.py` runs it with the real checkpoint
where the checkpoint and the calls exist, and checks the file and the comparison logic
everywhere. The check also refuses a demo folder holding a call with no expected verdict or a
team member's voice.

**Three detectors on one call (B2).** `DFD_LANES` names extra detectors as
`name=checkpoint=threshold_file`; `live_call/lanes.py` scores one call through all of them,
each with its own `StreamingScorer` and `VerdictEngine`, so every lane is the live path and
differs only in the model and its frozen operating point. A lane on the primary checkpoint
shares its model rather than loading a second copy. In real time the call is paced by the
wall clock, not by a sleep after each frame, so slow windows never drift the scores away from
the audio the browser is playing. `/lanes` shows the three timelines on one clock, each
lane's light, the time of its first caution and warning, and at the end the strongest warning
each detector gave with a one-line outcome. Scene 2 uses `genuine_new_speaker`, not the
plan's `genuine` call: on `genuine` the English-only lane reaches "likely cloned" only at
20 s, on `genuine_new_speaker` at 10 s, which is what the presenter says.

**The receiver's phone (B3).** The receiver page was rebuilt for a phone held in the hand and
for the projector: a large verdict, the time of the first caution and warning, "Verdicts reach
only the receiver" in the header, a full-screen red alarm that speaks the warning and
vibrates, and the screen kept awake during the call. If the phone's browser gives no
microphone (plain http over the room's Wi-Fi), it joins listen-only, so it still hears the
caller and still gets every verdict; that makes the tunnel optional. `/live` puts the caller
handset on the laptop beside a QR code for the phone and a live mirror of what the phone
shows, so the audience sees the warning the mentor is holding. `scripts/start_tunnel.py`
opens a cloudflared or ngrok tunnel and hands its address to the page.

**The real network, scores only (B4).** `/network` draws the call chain (deployed adapter
against final recipe over the six stages, each threshold dashed) from
`experiments/results/real_calls_v1/codec_chain.csv`, and agrees with the paper figure to the
fourth decimal. The stage player renders one held-out corpus speaker's clean recording through
the same codec chain (`scripts/prepare_stage_player.py`, Krishna's
`real_call_codecs.chain_versions`) and scores every stage on the live path, so the audience
hears what a phone line does without hearing anyone from the team. The 30-call table is
counted from the per-call CSVs and checked against the summary JSON's percentages.

**Why it can be trusted (B5).** `/trust` cuts the call into 100 ms pieces
(`interpretability.shuffle_chunks`), scores both versions through the live path and plays
both, beside the shortcut floors of the final recipe from `floor_final_v1.json`.

**The flow (B6).** `/demo` is the presenter's start page: the six scenes in order with their
lines and a readiness check (model, lanes, calls, stage player, phone address). `/threat`
(scene 1) and `/limits` (scene 6) are new; the one-screen console moved to `/console`. Every
scene page shares one projector-first look (`stage.css`, `stage.js`): a scene stepper, live
status pills, the presenter's lines in a drawer (`N`), arrow keys between scenes, and the
source file of every number on hover. Scene 6 shows the four limits with their numbers and
what was done about each overnight, read at request time, so Track A's results appear the
moment they are committed; the spontaneous-speech trade-off is already drawn from
`spontaneous_tradeoff_v1/tradeoff.csv` with the live threshold marked. "Try it" judges a
microphone recording exactly as a call (`POST /api/try`), not by its single lowest window,
which would warn a real voice on one dip. `scripts/run_demo.bat` sets every variable, starts
the server and opens `/demo`. The overview page now shows the final recipe; no page carries
the week-10 numbers.

**Ready for a fourth lane.** A checkpoint with `ssl` in its name loads through
`predict.load_ssl_aasist`, so SSL-AASIST becomes the fourth lane with one more `DFD_LANES`
entry once its checkpoint and threshold file arrive.

**One rule for every page.** No page holds a number. They ask `/api/headline`,
`/api/real_calls` and `/api/limits`, which answer from `live_call/results_view.py` with each
value's file and field; `tests/test_results_view.py` resolves every cited field in its file,
recounts the real-call table from the per-call CSVs, and fails if a result is typed into any
page.

## Numbers

All measured tonight on this laptop (CPU, final recipe, `C:\dfdata\demo_calls_v2`).

| Check | Result |
|---|---|
| Rehearsal, 8 calls x (live model + 3 lanes) | 32 of 32 rows PASS, twice in a row (about 150 s a round) |
| Scene 2, genuine new speaker | English-only: caution 6 s, likely cloned 10 s; deployed and final: no low window |
| Scene 2, Tortoise call (unseen tool) | final: caution 8 s, likely cloned 12 s; deployed: caution 12 s, never likely cloned; English-only: likely cloned 16 s |
| Scene 3, XTTS clone through WebRTC (two headless browsers) | receiver: caution 0:08, likely cloned 0:12, full-screen alarm; caller page: no verdict |
| Scene 3, genuine call through WebRTC, receiver with no microphone | listen-only: no warning in 48 s; every verdict arrived |
| Scene 4, stage player (held-out speaker 596505), mean P(genuine) | deployed 0.76 direct, 0.73 G.711, 0.00 AMR-NB, 0.00 AMR-NB + recorder, 0.23 AMR-WB + recorder; final 1.00 at every stage (`stage_player_v1.json`) |
| Scene 5, shuffle into 100 ms pieces | XTTS clone: 23 of 23 windows low before and after (likely cloned); genuine new speaker: 0 of 21 low before, 1 of 21 after (genuine both times) |
| Cold start of `run_demo.bat` | `/demo` served at 1.2 s, all three models loaded at 8.4 s |
| CPU cost of three lanes | about 0.6 s per 2 s hop (one 46 s call scored in 13-16 s offline) |
| Test suite | 1,110 passing, the console's browser tests included (run against the live server) |

## Honest limitations

- **No person has held the phone yet.** The phone scene was checked with two headless
  browsers on fake devices through the real WebRTC path, not with the mentor's phone over the
  room's network. The human rehearsals (B7.2) are what test it.
- **The tunnel tool is not installed on this laptop.** Downloading cloudflared was not done
  tonight; without it the phone uses the laptop's Wi-Fi address over plain http, listen-only
  (checked headlessly). Windows may ask to allow Python through the firewall on first run.
- **"Try it" has not heard a real voice.** It records from the laptop microphone, clean,
  while the final recipe was trained on phone-line audio. Try it with your own voice before
  the review; if it warns, leave it out.
- **The rehearsal checks file replay, the room uses WebRTC.** Opus can move a score; the
  WebRTC path was checked once per call type, not twice per call.
- **The deployed adapter does not "waver" on the genuine corpus calls**, as the plan
  expected: it stays green. The presenter's line says only what happens.
- **The fourth lane is untested on a real checkpoint**: the SSL-AASIST loader was tested with
  a stand-in network; its checkpoint and threshold file have not arrived.
- **Two of the night's Track A results are still pending** (SSL-AASIST on the real calls, the
  early-warning study); scene 6 says so until their files are committed.
- **The stage player starts from the speaker's clean MUCS recording**, the source of the demo
  call, not from the exact clips joined into it.

## How to run it

```bat
scripts\run_demo.bat
python scripts\rehearse_demo.py --server http://localhost:8000
python scripts\start_tunnel.py
python scripts\prepare_stage_player.py --clean-dir C:\dfdata\processed\clean\mucs2021 ^
    --recording mSiZUO1cgwb8nNpA --speaker 596505 ^
    --deployed C:\dfdata\checkpoints\lora_norm_rvc_channel_best.pt ^
    --final C:\dfdata\checkpoints\final_s1234_best.pt --out C:\dfdata\stage_player
pytest tests -q --deselect tests/test_env_check.py
```

## What's next

The human part of B7: two timed rehearsals with the phone in hand, the backup screen
recording, then review and merge. If Track A's SSL-AASIST checkpoint arrives, add it as the
fourth lane and its expectations to `configs/demo_rehearsal.json`.
