# Week 14 — Mounika — the live demo for the final review (Track B)

The final review needs a demo that shows a complete research system — the gap, the cheap
fix, the better fix, a real-call pilot and the limits with their numbers — and that cannot
misfire in front of the mentor and the external reviewer. Track B of the review plan built
it on the final recipe, every number read from a committed result file, every call checked
against a written-down verdict before it is shown. On 28 October, after the morning's
feedback ("confusing, too much information"), Track 2 cut it to four scenes with one message
each; in the afternoon the scenes became a working product, Asli, whose pages are named for
what they do. Both parts are at the end of the next section.

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

**The larger encoder, as it landed overnight (#63).** Krishna's SSL-AASIST results are merged
into this branch and shown in scene 6 beside the final recipe, each number read from its file:
on the 30 real calls it takes 1 of 6 Chatterbox and 6 of 6 XTTS-v2 calls to likely cloned and
warns 1 of 12 genuine live calls, and on the held-out spontaneous calls it warns 11 of 41 where
the final recipe warns all 41. The card says what Krishna's record says: one checkpoint, the
weakest of six offline on Chatterbox, so the next candidate and not a replacement. The demo
loads it through Krishna's `predict.load_score_fn(kind="ssl_aasist")` (my own loader is gone),
and `run_demo.bat` adds it as a fourth lane at its own threshold
(`configs/threshold_ssl_aasist.yaml`) as soon as `ssl_aasist_final_s1234_fp16.pt` is in
`C:\dfdata\checkpoints`; scene 2 then reads "Four detectors" and fits four lanes on a
768-line projector.

**The fourth lane on the real checkpoint (morning of 28 October).** With the weights copied
over, every demo call was replayed twice through all four lanes and the SSL-AASIST verdicts
were identical both times: green on the three genuine calls, likely cloned on the four
clone calls (caution 6 s, likely cloned 10 s, two seconds ahead of the final recipe each
time), including the Tortoise call the final recipe lets through, and only a caution (16 s)
on the spontaneous caller. Those observed verdicts are now the lane's expectations in
`configs/demo_rehearsal.json`, marked optional: the rehearsal judges the lane when the server
runs it and skips it on a laptop without the weights. They are eight calls chosen for the
final recipe, so they are a rehearsal check, not evidence; the evidence is the real-call
table in scene 6. Wiring it in also found a bug of mine: `run_demo.bat` had a tab where
`configs\threshold_ssl_aasist.yaml` needed a backslash, so the lane loaded with no threshold.
The stand-in check had set the lanes by hand and missed it; the rehearsal through the script
caught it.

**The early-warning study, as it landed (on main).** Krishna's stricter gate for the first
10 s of a call, set on dev only, is now on scene 6's third card, without and with the gate,
from `experiments/results/early_warning_v1/summary.json`: over G.711 it clears two of the
final recipe's three warned genuine live calls, but as recorded it warns one more, it delays
the first caution on the XTTS-v2 calls, and it silences 83 of the 120 early windows, speech
included. The card ends with Krishna's conclusion: the gate needs real development calls.

**Track 2: the demo made simple (28 October, Krishna's plan, freeze 15:30).** The detector,
the live path, the thresholds and the call set did not change; only what a first-time viewer
reads did.

- *Four scenes, not six, about eight minutes.* The threat, four detectors on one call, the
  live call on the mentor's phone, the limits. `/network`, `/trust` and the try-it box (now its
  own page, `/try`) left the flow and sit in a **More** menu at the foot of scene 4, for
  questions. Try it moved because it warns most real voices speaking freely, which is the first
  limit; it should not be the last thing the panel sees.
- *One sentence at the top of every page* (`h1.title.one`), in plain words, and at most three
  numbers in the default view; the rest opens under **Details** (`Stage.details`), which is a
  real toggle, not a second page, so the presenter can answer a question without leaving the
  scene.
- *The same four names everywhere* (English-only, Deployed adapter, Final recipe, Larger
  encoder) from `Stage.NAMES`, keyed by each lane's role, so the screen never depends on what a
  lane is called in `DFD_LANES` and `lanes.py` did not have to change. *The same three verdict
  words and colours everywhere*, the console's and the phone's included, with one key line at
  the foot of every page: "Caution = a few suspicious seconds in a row. Likely cloned = more of
  them." It replaced the presenter bar's scene name and clock, which were for the presenter
  only and now sit inside the notes.
- *Scene 2 is four lights.* Each detector is a card with a big light, its verdict word and two
  times (first caution, likely cloned), and at the end of the call the strongest warning it
  gave with a one-line outcome. The timelines, thresholds and warning ladders open under
  Details, where each detector becomes a row with its timeline. The two calls are one click
  each and are named so the title does not give the answer away ("A genuine caller", "A
  Tortoise clone"); the other calls stay in a small menu for questions.
- *Scene 4 is three cards with one headline number each*: genuine callers speaking freely
  warned by the final recipe (41 of 41, the larger encoder 11 of 41), real-network Chatterbox
  calls caught (0 of 6, the larger encoder 1 of 6), and the deployed adapter's error on English
  over a phone line (24.2%, against 2.3% on the same clips without it), plus one sentence for
  the call chain (the codec alone takes the deployed adapter to its threshold; the values are in
  its tooltip). Details holds the larger encoder's real-call table, the early-call gate, the
  30-call table and the trade-off chart.
- *Nothing moves unless the presenter clicks.* Scene 1's chat bubbles no longer animate in; the
  live page lost its per-window sparkline (the phone's verdict and its three times say it); the
  top bar's three status pills (model, threshold, call count) became one light that reads
  "Ready", with the details on hover.
- *No new numbers.* Two existing results are now served for these screens, both cited to their
  files: the clip counts behind scene 1's rate (1,484 of 1,566 genuine clips,
  `classification_metrics.json`, under Details) and the English-over-the-phone-line pair
  (`english_channel_v1/summary.json`). `tests/test_demo_routes.py` now checks the four-scene
  order, the More pages, a one-sentence title and the shared key on every page, the timelines
  behind Details, three limit cards, and that no page shows a system code or a checkpoint name.
- *The branch.* Krishna's plan named a new branch, `week14-mounika-demo-simplify`; the team's
  rule is one branch per member per week (GIT_RULES §11), so this went on
  `week14-mounika-demo`, fast-forwarded to main after #62 merged: the same starting point,
  without a third week-14 branch.

**The product: Asli (28 October, afternoon, after #64 merged).** Shown the four scenes, the
question was why the demo reads like slides ("Scene 1, 2, 3, 4") and not like a working
project. So the same pages became one product with the features a user would look for. The
detector, the live path, the thresholds, the call set and `configs/demo_rehearsal.json` are
unchanged.

- *A navigation rail instead of a scene stepper.* Six pages named for what they do: Overview,
  Live call, Check audio, Call log, Compare, Performance; the status light (Ready) sits at the
  bottom of the rail. No page says "scene"; a test now fails if one does. The evidence pages
  (why it is needed, the real phone network, the trust check) are tabs inside Performance.
  The presenter keeps the eight-minute walkthrough on the arrow keys (Overview, Compare, Live
  call, Performance) and the notes on `N`; keys `1`-`6` open the rail's pages.
- *An Overview that is a home screen.* What it does in one line, two actions (start a
  protected call, check a recording), a Protection card (on or off, the live model, the four
  detectors, how the phone pairs, how many calls it has judged), three measured results with
  their files (94.8% of genuine Hindi-English speech called fake by the English-only detector;
  5 of 6 XTTS-v2 calls over a real mobile network warned; 6 of 6 Tortoise clones warned in 60 s
  calls), the first limit in one line, the features as tiles, the five latest calls and how it
  works in four steps.
- *A call log (`/history`, new).* Every call the server has judged, newest first: its
  strongest warning (not the light it ended on), the time of the first warning, suspicious
  windows and length; for a prepared call, whose truth is known, whether the clone was caught
  or the real caller let through. "Run a test call" plays a prepared call through the live
  path and the row arrives with its verdict. To keep the strongest warning, the session
  snapshot (`live_call/session.py`) now also reports `peak_state`, `duration` and
  `first_warning_s`; a test covers it.
- *Check audio (`/try`) grew from a microphone button into a checker.* Upload a file (drag and
  drop), record a line, or pick one of the prepared samples; the result is the verdict, what
  to do about it, three facts, the window timeline against the threshold and a player for what
  was checked, with the spontaneous-speech limit under it. A recording shorter than the
  ladder's first two windows (6 s) is refused with that reason and the page reads "Too short to
  judge": a 5 s upload used to come back with the word "Listening" and no advice, because one
  window can never reach a verdict.
- *Performance leads with what it catches* (the three results above, as cards) before the
  three limits, which stay as they were, with Details below.
- *Prepared calls are named for what they are, never for how they turned out*: "Tortoise cloned
  voice, call 1" and "call 2", not "caught" and "it misses", wherever a title is shown.
- *The name.* "Asli" (असली, genuine) is one constant in `stage.js`; the team can change it in
  one line.

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
| Rehearsal, 8 calls x (live model + 4 lanes, SSL-AASIST weights present) | 40 of 40 rows PASS, twice in a row (about 300 s a round) |
| Scene 2, genuine new speaker | English-only: caution 6 s, likely cloned 10 s; deployed and final: no low window |
| Scene 2, Tortoise call (unseen tool) | final: caution 8 s, likely cloned 12 s; deployed: caution 12 s, never likely cloned; English-only: likely cloned 16 s |
| Scene 3, XTTS clone through WebRTC (two headless browsers) | receiver: caution 0:08, likely cloned 0:12, full-screen alarm; caller page: no verdict |
| Scene 3, genuine call through WebRTC, receiver with no microphone | listen-only: no warning in 48 s; every verdict arrived |
| Scene 4, stage player (held-out speaker 596505), mean P(genuine) | deployed 0.76 direct, 0.73 G.711, 0.00 AMR-NB, 0.00 AMR-NB + recorder, 0.23 AMR-WB + recorder; final 1.00 at every stage (`stage_player_v1.json`) |
| Scene 5, shuffle into 100 ms pieces | XTTS clone: 23 of 23 windows low before and after (likely cloned); genuine new speaker: 0 of 21 low before, 1 of 21 after (genuine both times) |
| Cold start of `run_demo.bat` | `/demo` served at 1.2 s, all three models loaded at 8.4 s |
| CPU cost of three lanes | about 0.6 s per 2 s hop (one 46 s call scored in 13-16 s offline) |
| CPU cost of four lanes | about 1.3 s per 2 s hop (the 45 s Tortoise call scored in 27 s offline) |
| Test suite | 1,133 passing after Track 2; 1,136 after the product redesign, the console's browser tests included (run against the live server) |
| Track 2: rehearsal after the simplification (live model + 4 detectors) | 40 of 40 rows PASS, twice in a row (243 s and 285 s): "PASS on every row, 2 rounds in a row." |
| Track 2: screens checked at 1366x768 and 1920x1080 | each scene's default view reads on one screen at 1366x768, scene 2's Details view included; no page errors on any page |
| Product: click-through in two headless browsers (1366x768) | 28 of 28 checks: samples judged (XTTS-v2 likely cloned, genuine caller genuine), an uploaded RVC conversion likely cloned, a test call in the log as "clone caught", the Overview's status and recent calls, keys 1-6 and the walkthrough arrows, Performance's tabs and six headline numbers, the live call over WebRTC (phone and mirror "Likely cloned", caller never told) and the call in the log; no page errors |
| Product: rehearsal after the redesign (live model + 4 detectors) | 40 of 40 rows PASS, twice in a row: "PASS on every row, 2 rounds in a row." Round 1 took 425 s; round 2 took 3,946 s, a stall most likely from the laptop sleeping mid-round, and every row still passed |
| Product: backup recording | `backup_demo.mp4` recorded again on the product pages (3:51, 1920x1080): the walkthrough's four pages, then the call log and Check audio, with the call audio laid back in and the phone beside the laptop in the live call; 9 of 9 checks, every lane's outcome on time (45.3 s for the 45.1 s Tortoise call). The four-scene recording is kept as `backup_demo_four_scenes.mp4` |
| Product: Check audio, a 5.3 s upload | refused with "record at least 6 seconds" and shown as "Too short to judge"; a prepared sample still judged (headless, 1366x768, no page errors) |

## Honest limitations

- **No person has held the phone yet.** The phone scene was checked with two headless
  browsers on fake devices through the real WebRTC path, not with the mentor's phone over the
  room's network. The human rehearsals (B7.2) are what test it.
- **The tunnel tool is not installed on this laptop.** Downloading cloudflared was not done
  tonight; without it the phone uses the laptop's Wi-Fi address over plain http, listen-only
  (checked headlessly). Windows may ask to allow Python through the firewall on first run.
- **Check audio's recorder has not heard a real voice.** It records from the laptop
  microphone, clean, while the final recipe was trained on phone-line audio. Try it with your
  own voice before the review; if it warns, show uploads and samples only. Uploads and samples
  were checked headlessly.
- **The call log lives in memory.** It lists the calls since the server started (the last
  twenty that ended, plus any live), not a history kept across restarts.
- **The product redesign came after Track 2's freeze.** #64 merged at 13:10 with the four
  scenes; this redesign needs Krishna's review before it replaces them, and the four-scene
  version on main stays the fallback until then.
- **The rehearsal checks file replay, the room uses WebRTC.** Opus can move a score; the
  WebRTC path was checked once per call type, not twice per call.
- **The deployed adapter does not "waver" on the genuine corpus calls**, as the plan
  expected: it stays green. The presenter's line says only what happens.
- **Four lanes leave less headroom.** SSL-AASIST is the slowest model: one 45 s call scores
  through all four lanes in 27 s offline, about 1.3 s of every 2 s hop, against 0.6 s for
  three. Scene 2 keeps up on this laptop on mains power; on battery, or if the timelines lag
  the audio, start the server without the weights (three lanes).
- **The SSL-AASIST lane's expectations were observed, not predicted**: written down after two
  identical replays, unlike the other lanes, whose verdicts were stated before the check.
- **The stage player starts from the speaker's clean MUCS recording**, the source of the demo
  call, not from the exact clips joined into it.
- **Scene 4's counts are derived.** "41 of 41" and "11 of 41" are the stored share of genuine
  spontaneous calls warned times the stored number of calls: the calibration files keep the
  share, not the count. The tooltip shows both.
- **"At most three numbers" holds for the default view only.** Details and the More pages
  still carry the full tables; a question can open them, a first look does not.
- **The simplified screens have been seen by me, not yet by a first-time viewer.** Whether
  the panel now reads each page in one look is what the next rehearsal with someone outside
  the team will tell.

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

Krishna's review of the product pages, then the human part of B7: two timed rehearsals with
the phone in hand. Show the product once to someone outside the team and change any sentence
they have to ask about. Four detectors need the laptop on mains power; without the
larger encoder's weights Compare falls back to three and says so.
