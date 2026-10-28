# Final review: the eight-minute demo (presenter's sheet)

Print this. Four scenes, one message each. Every page says its point in one sentence at the
top, shows at most three numbers, and keeps the rest behind **Details**. Every number on screen
is read from a committed result file; hover it to show which. The same four names everywhere:
**English-only, Deployed adapter, Final recipe, Larger encoder**; the same three verdicts:
**Genuine** (green), **Caution** (amber), **Likely cloned** (red), with the key at the foot of
every page: "Caution = a few suspicious seconds in a row. Likely cloned = more of them."

## Before anyone arrives (30 minutes)

1. Double-click `scripts\run_demo.bat`. Chrome opens `/demo`; wait until every line of
   **Ready to present?** is green (amber "no tunnel" is fine if the phone uses the Wi-Fi), and
   the light at the top right reads **Ready**.
2. In a second terminal: `python scripts\rehearse_demo.py --server http://localhost:8000 --rounds 2`.
   It must end with **PASS on every row, 2 rounds in a row**. If any row fails, do not show
   that scene; tell Krishna.
3. The phone: on `/live`, scan the QR code with the mentor's phone (same Wi-Fi as the laptop,
   or `python scripts\start_tunnel.py` for https). The receiver page opens; tap once anywhere
   so it may play sound. `/live` must read **Phone connected**.
4. Press `F` for full screen. `N` opens these lines on screen; arrow keys move between
   scenes; `1`-`4` jump to a scene.
5. Laptop on mains power (four detectors use most of the CPU), phone charged, hotspot tested,
   `backup_demo.mp4` on the desktop.

## The four scenes (8 minutes)

| # | Time | Page | The one sentence on screen | Do | Say (one breath each) |
|---|---|---|---|---|---|
| 1 | 0:45 | `/threat` | A familiar voice calls in Hindi-English and asks for an OTP. The usual detector cannot help: it calls (number) of genuine Hindi-English speech fake. | read the sentence; point at the big number | The usual detector, trained on English, calls almost every real Hindi-English caller fake. It cannot protect this call. |
| 2 | 2:00 | `/lanes` | Same call, four detectors. | click **A genuine caller**; when it ends, click **A Tortoise clone**; read the four lights | The English-only detector rejects a real person; ours let them through. Now a clone made with a tool none of them was trained on: the final recipe catches it. The larger encoder is one checkpoint, our next candidate. |
| 3 | 3:00 | `/live` | A clone is calling your phone. Only the receiver is warned. | hand the phone over; on the laptop the XTTS-v2 call is already chosen: press call; the mentor accepts | Please hold this phone. A clone is calling you. [amber] A caution. [red, spoken] Likely cloned. The caller sees nothing, because a caller who knows they were caught learns to evade. |
| 4 | 2:00 | `/limits` | What it cannot do yet, with numbers. | read the three cards left to right | Genuine callers speaking freely are still warned; Chatterbox, a tool it never saw, gets through on real calls; and the deployed adapter's English collapses over a phone line. The larger encoder does better on the first two, but it is one checkpoint. |

Scene 4's **Details** opens the larger encoder's real-call table, the early-call gate study,
the 30-call table and the spontaneous-speech trade-off chart. Its **More** menu, for
questions only: **Real network** (call chain and stage player), **Why trust it** (the 100 ms
shuffle and the floors), **Try it** (read a line into the microphone). Try it is not the last
thing the panel sees: it warns most real voices speaking freely, which is the first limit.

## What each call must do (checked by the rehearsal)

| Scene | Call | Must show |
|---|---|---|
| 2 | A genuine caller (new speaker) | English-only red by 10 s; Deployed adapter, Final recipe and Larger encoder green throughout |
| 2 | A Tortoise clone | Final recipe "likely cloned", first caution 8 s, likely cloned 12 s; Deployed adapter only a caution; Larger encoder likely cloned at 10 s; English-only red, but it warns real callers too |
| 3 | XTTS-v2 cloned voice | caution, then the red "likely cloned" screen on the phone by 12 s; nothing on the caller's screen |
| 3 | Genuine caller | no warning for the whole call |
| Details | Genuine caller speaking freely | warned by the three earlier models, a caution from the larger encoder: the spontaneous-speech limit |
| Details | Tortoise clone it misses | green on the final recipe (kept on purpose, never swapped); the larger encoder catches it |

The deployed adapter does not waver on the genuine corpus calls; do not say it does. The
larger encoder's lights are one checkpoint on eight calls chosen for the final recipe: a
rehearsal check, not evidence. The 30 real calls on scene 4 are the evidence.

## If something goes wrong

- **The phone does not ring or the page will not load:** open `/console` (scene 3 on one
  screen: both handsets side by side) and play the same XTTS call there.
- **The four lights lag the audio:** the laptop is on battery or busy; restart
  `run_demo.bat` without the larger encoder's weights in `C:\dfdata\checkpoints` and scene 2
  reads "Same call, three detectors."
- **A page shows an error:** move on to the next scene; the numbers are the same in the papers.
- **Anything else:** play `backup_demo.mp4`.

## Say it honestly

- The carrier path on live numbers: "implemented and unit-tested; not yet run on live
  numbers."
- The real calls are a pilot: 30 calls, three speakers, recorded on the receiving phone and
  scored offline through the same path, at thresholds fixed before the calls.
- The larger encoder: one checkpoint, offline the weakest of six trainings on Chatterbox; the
  next candidate, not a replacement.
- Questions and answers: [`VIVA_QA.md`](VIVA_QA.md).
