# Real phone-call test: protocol

Every result in the paper comes from recorded corpora passed through a simulated or
reference telephone codec. This test puts the detector on real calls over a real mobile
network: genuine calls from the three of us, and calls in which a clone of one of our
voices is played into the phone. It is small (30 calls, 3 speakers) and is reported as a
pilot.

The team members are **P1, P2 and P3**. Who is who is written only on the signed consent
forms, never in the repository.

## 0. Before anything is recorded

1. Each of us signs the consent form
   ([`docs/ethics/real_call_consent_form.md`](../ethics/real_call_consent_form.md)), and
   the mentor signs its approval box. Scan or photograph the signed pages into
   `docs/ethics/` as `real_call_consent_<P>.pdf` (or `.jpg`). `.gitignore` keeps every
   PDF and image in that folder out of git.
2. Save the mentor's approval message the same way (`real_call_mentor_approval.png`).

## What you need

- **Two phones on the mobile network**: one *calling phone* and one *receiving phone*,
  the same two for every call. Normal phone calls, not WhatsApp or any internet call.
- **Call recording on the receiving phone.** Most Android phones have it in the dialler
  settings; on an iPhone (iOS 18.1 or later) it is the record button during a call.
  Everyone on the call has consented, so the phone's "this call is being recorded"
  announcement is fine.
- **A laptop with a speaker** for the replayed calls, and a quiet room.
- The scripts: [`SCRIPTS.md`](SCRIPTS.md) (print it or open it on another screen).

**Test first.** Make one 20-second call, record it, and send the file to be checked
(format, loudness, that only the caller is heard) before the real calls start.

## 1. Reference recordings (Monday morning, about 30 minutes)

Each person, in a quiet room, with a phone's voice-recorder app about 20 cm from the mouth:

| File name | What to read | Length |
|---|---|---|
| `P1_ref` | the four Reference sentences | about 20 s |
| `P1_read_A` | all of Script A | about 60 s |
| `P1_read_B` | all of Script B | about 60 s |

(P2 and P3 the same, with their own code.) Any file type is fine (m4a, mp3, wav). Put all
nine files in `data/real_calls/references/`.

## 2. Make the clones (about 40 minutes, on Kaggle)

1. Build the cloning packs:
   `python -m src.data.real_call_pack pack --recordings data/real_calls/references --out data/real_calls/pack`
2. Kaggle -> Datasets -> New Dataset -> upload the folder `data/real_calls/pack` (both
   subfolders, `rc_xtts` and `rc_chatterbox`) -> name it `codemix-w14-team-voices` ->
   visibility **Private**.
3. Kaggle -> Code -> Import notebook -> `notebooks/kaggle_w14_team_clones.ipynb`. Add Input:
   that dataset and `saikrishnareddy9/mentor-signoff`. Settings: GPU T4 x2, Internet On.
   Save Version -> Save & Run All (Commit).
4. When it finishes, download `team_clones_results.zip` from the notebook's Output, check
   that `summary.txt` says `FINISHED` and `every step ran without error`, and unzip it into
   `data/real_calls/clones/`.
5. Build the playback files:
   `python -m src.data.real_call_pack calls --clones data/real_calls/clones --recordings data/real_calls/references --out data/real_calls/playback`
   This writes 18 files: each person's own readings and the two tools' clones of them,
   all at the same level.

The live calls (C01 to C12) need no clones, so make them while Kaggle runs.

## 3. The 30 calls

**Every call:**

1. Call from the calling phone to the receiving phone.
2. The receiver answers, says nothing, and **mutes the microphone** straight away, so the
   recording holds only the caller.
3. The caller waits two seconds, then speaks or plays for about a minute, then hangs up.
4. Rename the recording to the call's ID (`C01`, `C02`, ...) and tick it below.

**Live calls (C01 to C12).** The person holds the calling phone normally and speaks:
Script A or B read from `SCRIPTS.md`, or *free talk*: about a minute of natural speech in
the Hindi-English mix you normally use, on any topic (your day, a film, weekend plans),
with no script.

**Replay calls (C13 to C30).** Put the calling phone on the table with its microphone 10 to
15 cm from the laptop speaker. Set the laptop volume once (for example 60%) and do not
change it for any replay call. Start the call, wait two seconds, play the file named
below from `data/real_calls/playback/`, and hang up when it ends. The genuine replays
(C13 to C18) go through exactly the same speaker as the clones, so they are the control
for the loudspeaker.

| Call | Speaker | Kind | Read or play | Done |
|---|---|---|---|---|
| C01 | P1 | speaks live | Script A | |
| C02 | P1 | speaks live | Script B | |
| C03 | P1 | speaks live | free talk | |
| C04 | P1 | speaks live | free talk | |
| C05 | P2 | speaks live | Script A | |
| C06 | P2 | speaks live | Script B | |
| C07 | P2 | speaks live | free talk | |
| C08 | P2 | speaks live | free talk | |
| C09 | P3 | speaks live | Script A | |
| C10 | P3 | speaks live | Script B | |
| C11 | P3 | speaks live | free talk | |
| C12 | P3 | speaks live | free talk | |
| C13 | P1 | own reading, replayed | genuine_P1_A.wav | |
| C14 | P1 | own reading, replayed | genuine_P1_B.wav | |
| C15 | P2 | own reading, replayed | genuine_P2_A.wav | |
| C16 | P2 | own reading, replayed | genuine_P2_B.wav | |
| C17 | P3 | own reading, replayed | genuine_P3_A.wav | |
| C18 | P3 | own reading, replayed | genuine_P3_B.wav | |
| C19 | P1 | clone, replayed | xtts_v2_P1_A.wav | |
| C20 | P1 | clone, replayed | xtts_v2_P1_B.wav | |
| C21 | P2 | clone, replayed | xtts_v2_P2_A.wav | |
| C22 | P2 | clone, replayed | xtts_v2_P2_B.wav | |
| C23 | P3 | clone, replayed | xtts_v2_P3_A.wav | |
| C24 | P3 | clone, replayed | xtts_v2_P3_B.wav | |
| C25 | P1 | clone, replayed | chatterbox_P1_A.wav | |
| C26 | P1 | clone, replayed | chatterbox_P1_B.wav | |
| C27 | P2 | clone, replayed | chatterbox_P2_A.wav | |
| C28 | P2 | clone, replayed | chatterbox_P2_B.wav | |
| C29 | P3 | clone, replayed | chatterbox_P3_A.wav | |
| C30 | P3 | clone, replayed | chatterbox_P3_B.wav | |

The same plan, as a table the scoring code reads, is in [`call_plan.csv`](call_plan.csv).

## 4. Score the calls

Copy the recordings into `data/real_calls/recordings/` (`C01.m4a`, `C02.m4a`, ...), then:

```
python -m src.inference.real_calls prepare --recordings data/real_calls/recordings --out data/real_calls/scored
python -m src.inference.real_calls score --audio data/real_calls/scored --deployed <deployed checkpoint> --final <final_s1234 checkpoint> --out experiments/results/real_calls_v1
python -m src.inference.real_calls report --audio data/real_calls/scored --windows experiments/results/real_calls_v1 --md experiments/results/real_calls_v1/report.md
```

Each call is scored twice: as the receiving phone recorded it, and after an 8 kHz G.711
round trip, which is what the live system receives from its telephone carrier. Both models
are scored, the deployed adapter and the final recipe, each through the live verdict ladder
at its frozen operating point. No threshold is chosen on these calls. Scoring takes about
15 to 30 minutes on a laptop CPU.

## What is kept, and where

| Item | Where | In git? |
|---|---|---|
| Scripts, call plan, this protocol | `docs/real_calls/` | yes |
| Per-window scores, per-call verdicts, report | `experiments/results/real_calls_v1/`, `docs/results/` | yes (no audio) |
| Reference recordings, clones, playback files, call recordings | `data/real_calls/` | **never** (`data/` is ignored) |
| Signed consent forms, mentor approval | `docs/ethics/` | **never** (PDFs and images there are ignored) |
| Clone dataset and notebook output | Kaggle, **private** | no |

The recordings and clones are evaluation-only: `realcall` is listed in
`EVAL_ONLY_SOURCES` (`src/data/build_manifests.py`), so no training manifest can contain
them. When the project ends, delete the Kaggle dataset, the notebook's output and the
`data/real_calls/` folder, as the consent form promises.

## What happened on the day (26 and 27 October 2026)

The plan above assumes the three of us in one place with two fixed phones. On the day we were
at home in three places, so the setup changed, and these changes are part of the result:

- **Phones.** The live calls of P2 (C05-C08) were made from her own phone to P3's phone, which
  recorded them. The other way round, P2's phone received and recorded P3's live calls
  (C09-C12) and every replay call from C15 to C30. TO CONFIRM: which phone recorded P1's live
  calls (C01-C04) and her first two replays (C13, C14). Every call recording is 8 kHz mono
  AAC at the same bit rate, with no tags naming the phone.
- **References** were recorded on each speaker's own phone (P2: AAC at 44.1 kHz; P3: MP3 at
  48 kHz) rather than on one shared phone, and reached the laptop as a WhatsApp document (P2,
  then deleted from the chat) or by USB cable (P3).
- **P1 was unwell**, so her references, her live calls (C01-C04) and her replay calls (C13,
  C14, C19, C20, C25, C26) were made the next day, with her clones from a second Kaggle run.
  That session's replays are about 9 dB louder than the other twelve (integrated loudness
  -29 to -33 LUFS against -33 to -47), so speaker and replay level are confounded within the
  replay calls.
- **Clones.** XTTS-v2 passed 45 of 48 sentences and Chatterbox 47 of 48 for P2 and P3, and 24
  of 24 and 23 of 24 for P1 (69 and 70 of 72 in all); the five that stalled were left out, so
  five playback calls hold 11 of 12 sentences.
- **A 20 s test call** (P2) checked the format first; it is not one of the 30 calls.
- **Labels checked.** With the calls made from home, nobody else watched which file was played
  into which call, so before scoring every replay recording is matched back to its playback
  file by speech activity (`python -m src.inference.real_calls check`); all 12 replays of P2
  and P3 hold their planned file.

## What this test can and cannot show

- 30 calls, 3 speakers, one pair of phones and one network: a pilot. A percentage over
  6 or 12 calls moves in steps of 8 to 17 points.
- The readers are not corpus speakers and read prepared scripts in half the calls; the
  free-talk calls are the only spontaneous speech.
- Replayed calls add a loudspeaker and a room. The genuine replays are the control for
  that, so the fair comparison for the clones is against them; the live calls show what a
  real caller meets.
