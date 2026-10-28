# Final review: the ten-minute demo (presenter's sheet)

Print this. It is the plan's presenter script (section 6) set against the pages that now
exist. Every number on screen is read from a committed result file; hover it to show which.

## Before anyone arrives (30 minutes)

1. Double-click `scripts\run_demo.bat`. Chrome opens `/demo`; wait until every line of
   **Ready to present?** is green (amber "no tunnel" is fine if the phone uses the Wi-Fi).
2. In a second terminal: `python scripts\rehearse_demo.py --server http://localhost:8000`.
   It must end with **PASS on every row, 2 rounds in a row**. If any row fails, do not show
   that scene; tell Krishna.
3. The phone: on `/live`, scan the QR code with the mentor's phone (same Wi-Fi as the laptop,
   or `python scripts\start_tunnel.py` for https). The receiver page opens; tap once anywhere
   so it may play sound. `/live` must read **Phone connected**.
4. Press `F` for full screen. `N` opens these lines on screen; arrow keys move between
   scenes; `1`-`6` jump to a scene.
5. Laptop and phone charged, hotspot tested, `backup_demo.mp4` on the desktop.

## The six scenes

| # | Time | Page | Do | Say (one breath each) |
|---|---|---|---|---|
| 1 | 0:30 | `/threat` | nothing: the call script appears by itself | A familiar voice calls in Hindi-English and asks for an OTP. An English-trained detector calls most real Hindi-English speakers fake (read the number on screen), so it cannot protect this call. |
| 2 | 2:00 | `/lanes` | play **Genuine caller, a new speaker**, then **Tortoise cloned voice, caught** | Same call, three detectors. The English one rejects a real person. Our adapter fixes that. The final recipe also catches the tool none of them ever saw. |
| 3 | 3:00 | `/live` | hand the phone over; on the laptop pick **XTTS-v2 cloned voice**, press call; the mentor accepts | Please hold this phone. A clone is calling you. [amber] A caution. [red, spoken] Likely cloned. The caller sees nothing, because a caller who knows they were caught learns to evade. |
| 4 | 2:00 | `/network` | point at the chart; press **AMR-NB** then **Direct** in *Hear it* | We called each other over a real mobile network. Hear what the codec does. For two of us the old model collapses at the codec; the final recipe holds for all three, and catches most seen-tool clones (read the table). Scores only; we never play our own voices. |
| 5 | 1:00 | `/trust` | **Shuffle into 100 ms pieces and score both** on the XTTS clone | Scramble the words: same verdict. It reads how the voice was produced, not what is said. Every number is tested against a cheap-statistics floor. |
| 6 | 1:30 | `/limits` | point at each limit; then **Record 12 seconds** while the mentor reads the line | What it cannot do yet, with numbers, and what we did about each overnight. Now read this line into the microphone. |

## What each call must do (checked by the rehearsal)

| Scene | Call | Must show |
|---|---|---|
| 2 | Genuine caller, a new speaker | English-only lane red by 10 s; deployed and final lanes green throughout |
| 2 | Tortoise cloned voice, caught | final lane "likely cloned", first caution at 8 s; deployed lane only a caution; English-only red |
| 3 | XTTS-v2 cloned voice | caution, then the red "likely cloned" screen on the phone by 12 s; nothing on the caller's screen |
| 3 | Genuine caller | no warning for the whole call |
| 6 | Genuine caller speaking freely | warned by all three: the spontaneous-speech limit (open it from scene 6) |
| 6 | Tortoise cloned voice it misses | green on the final recipe: kept on purpose, never swapped |

The deployed adapter does not waver on the genuine corpus calls; do not say it does.

**With the larger encoder as a fourth lane** (SSL-AASIST, when its weights are on the laptop,
at its own threshold chosen on dev): green on both genuine calls; "likely cloned" on the
Tortoise call 2 s sooner than the final recipe (caution 6 s, likely cloned 10 s); it also
catches the Tortoise call the final recipe misses, and stops at a caution on the spontaneous
caller. Say what that is: one checkpoint on eight calls chosen for the final recipe, the next
candidate and not a replacement. The 30 real calls in scene 6 are the evidence.

## If something goes wrong

- **The phone does not ring or the page will not load:** open `/console` (scene 3 on one
  screen: both handsets side by side) and play the same XTTS call there.
- **A lane or page shows an error:** move on to the next scene; the numbers are the same in
  the papers.
- **"Try it" warns the mentor's real voice:** say so plainly. It is the spontaneous-speech
  and laptop-microphone limit of scene 6: the model was trained on read speech over phone
  lines. (Try it yourself before the review; if it warns you, skip it.)
- **Anything else:** play `backup_demo.mp4`.

## Say it honestly

- The carrier path on live numbers: "implemented and unit-tested; not yet run on live
  numbers."
- The real calls are a pilot: 30 calls, three speakers, recorded on the receiving phone and
  scored offline through the same path, at thresholds fixed before the calls.
- Questions and answers: [`VIVA_QA.md`](VIVA_QA.md).
