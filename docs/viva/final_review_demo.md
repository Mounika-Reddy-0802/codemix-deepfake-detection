# Final review: the eight-minute demo (presenter's sheet)

Print this. The demo is the product itself, **Asli**: a left rail of six pages, each named for
what it does. The review walks four of them; the other two answer questions. Every number on
screen is read from a committed result file; hover it to show which. The same four detector
names everywhere: **English-only, Deployed adapter, Final recipe, Larger encoder**; the same
three verdicts: **Genuine** (green), **Caution** (amber), **Likely cloned** (red), with the key
at the foot of every page: "Caution = a few suspicious seconds in a row. Likely cloned = more
of them."

| Rail | Page | What it does |
|---|---|---|
| 1 | Overview (`/demo`) | protection status, three measured results, the features, recent calls, how it works |
| 2 | Live call (`/live`) | the test caller on this laptop calls the protected phone; only the phone is warned |
| 3 | Check audio (`/try`) | upload a file, record a line, or pick a prepared sample: judged like a call |
| 4 | Call log (`/history`) | every call judged since the server started, its strongest warning, and "clone caught" / "false alarm" for prepared calls; "Run a test call" adds one |
| 5 | Compare (`/lanes`) | one call through the four detectors, a light each |
| 6 | Performance (`/limits`) | what it catches and what it cannot do yet; tabs for why it is needed, the real phone network and the trust check |

## Before anyone arrives (30 minutes)

1. Double-click `scripts\run_demo.bat`. Chrome opens the Overview; wait until **Protection**
   reads **On**, **Comparison** reads 4 detectors ready, and the rail's light reads **Ready**.
2. In a second terminal: `python scripts\rehearse_demo.py --server http://localhost:8000 --rounds 2`.
   It must end with **PASS on every row, 2 rounds in a row**. If any row fails, do not show
   that call; tell Krishna.
3. The phone: on **Live call**, scan the QR code with the mentor's phone (same Wi-Fi as the
   laptop, or `python scripts\start_tunnel.py` for https). The receiver page opens; tap once
   anywhere so it may play sound. The page must read **Phone connected**.
4. Press `F` for full screen. `N` opens these lines on screen; `→` and `←` move along the
   walkthrough below; `1`-`6` open a rail page.
5. Laptop on mains power (four detectors use most of the CPU), phone charged, hotspot tested,
   `backup_demo.mp4` on the desktop.

## The walkthrough (8 minutes)

| # | Time | Page | Do | Say (one breath each) |
|---|---|---|---|---|
| 1 | 0:45 | Overview | point at the headline, then the red 94.8% | A familiar voice calls in Hindi-English and asks for an OTP. The usual detector, trained on English, calls almost every real Hindi-English caller fake. Asli listens to the call as it happens and warns only the person being called. |
| 2 | 2:00 | Compare | click **A genuine caller**; when it ends, click **A Tortoise clone**; read the four lights | The English-only detector rejects a real person; ours let them through. Now a clone made with a tool none of them was trained on: the final recipe catches it. The larger encoder is one checkpoint, our next candidate. |
| 3 | 3:00 | Live call | hand the phone over; the XTTS-v2 call is already chosen: press call; the mentor accepts | Please hold this phone. A clone is calling you. [amber] A caution. [red, spoken] Likely cloned. The caller sees nothing, because a caller who knows they were caught learns to evade. |
| 4 | 2:00 | Performance | read the top row, then the three limits | It caught five of six cloned calls over a real mobile network, and every Tortoise clone in sixty-second calls, a tool it never saw. What it cannot do yet: genuine callers speaking freely are still warned; Chatterbox gets through on real calls; and the deployed adapter's English collapses over a phone line. The larger encoder does better on the first two, but it is one checkpoint. |

If there is time, or for questions: **Call log** shows the phone call just made, with its
strongest warning; **Run a test call** adds a prepared call with "clone caught". **Check
audio** judges an uploaded file or a sample in a few seconds. Performance's **Details** opens
the larger encoder's real-call table, the early-call gate study, the 30-call table and the
spontaneous-speech trade-off chart; its tabs open **Why it is needed**, **Real phone network**
(call chain and stage player) and **Trust check** (the 100 ms shuffle and the floors). Do not
end on recording a free-speech line: it warns most real voices speaking freely, which is the
first limit.

## What each call must do (checked by the rehearsal)

| Page | Call | Must show |
|---|---|---|
| Compare | A genuine caller (new speaker) | English-only red by 10 s; Deployed adapter, Final recipe and Larger encoder green throughout |
| Compare | A Tortoise clone | Final recipe "likely cloned", first caution 8 s, likely cloned 12 s; Deployed adapter only a caution; Larger encoder likely cloned at 10 s; English-only red, but it warns real callers too |
| Live call | XTTS-v2 cloned voice | caution, then the red "likely cloned" screen on the phone by 12 s; nothing on the caller's screen |
| Live call | Genuine caller | no warning for the whole call |
| Compare, More prepared calls | Genuine caller speaking freely | warned by the three earlier models, a caution from the larger encoder: the spontaneous-speech limit |
| Compare, More prepared calls | Tortoise cloned voice, call 2 | green on the final recipe (kept on purpose, never swapped); the larger encoder catches it |

The deployed adapter does not waver on the genuine corpus calls; do not say it does. The
larger encoder's lights are one checkpoint on eight calls chosen for the final recipe: a
rehearsal check, not evidence. The 30 real calls on Performance are the evidence.

## If something goes wrong

- **The phone does not ring or the page will not load:** open `/console` (the live call on one
  screen: both handsets side by side) and play the same XTTS call there.
- **The four lights lag the audio:** the laptop is on battery or busy; restart
  `run_demo.bat` without the larger encoder's weights in `C:\dfdata\checkpoints` and Compare
  reads "Same call, three detectors."
- **A page shows an error:** move on to the next page; the numbers are the same in the papers.
- **Anything else:** play `backup_demo.mp4`.

## Say it honestly

- The carrier path on live numbers: "implemented and unit-tested; not yet run on live
  numbers."
- The real calls are a pilot: 30 calls, three speakers, recorded on the receiving phone and
  scored offline through the same path, at thresholds fixed before the calls.
- The larger encoder: one checkpoint, offline the weakest of six trainings on Chatterbox; the
  next candidate, not a replacement.
- The call log lives in memory: it lists the calls since the server started, not a history
  kept across restarts.
- Questions and answers: [`VIVA_QA.md`](VIVA_QA.md).
