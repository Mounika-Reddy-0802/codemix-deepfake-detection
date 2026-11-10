# Corpus replay calls: protocol

The 30 real calls replayed only the team's own clones, six per tool, and every model misses
the replayed Chatterbox clones (0 of 6, `docs/voice_check.md`). That is too few calls, and too
few tools, to say how the voice check does on a clone played into a phone. This test plays
**corpus** audio into real mobile calls instead: test-set clips of speakers no model trained on,
from every cloning tool the project measures, with genuine clips of the same speaker pool
through the same loudspeaker as the control. Nobody new is recorded.

## The playback files (fixed before any call is made)

`python -m src.data.corpus_replay_pack --clean <clean clip list> --out D:/replay_pack` writes
30 files and their plan (`corpus_replay_plan.csv` here, with file names only):

| Kind | Files | From | Label |
|---|---|---|---|
| Genuine (control) | 6 | CM07 / CM09 genuine clips | genuine |
| IndicF5 clones | 6 | CM07 | cloned |
| Chatterbox clones | 6 | CM09 | cloned |
| RVC conversions | 6 | RVC test half | cloned |
| Tortoise clones | 3 | CM04 | cloned |
| XTTS-v2 clones | 3 | new speakers | cloned |

- **Speakers.** Test split only, never one of the 31 `gen_train` speakers, one speaker per
  file, drawn at random (generator seed 20261110) among those with at least 12 s of that kind,
  so that no file is too short to judge after a call.
- **Clips.** Up to six of that speaker's clips in random order, joined with 0.6 s gaps until the
  file passes 25 s (12.9-40.7 s in the end), resampled to 16 kHz, all at one loudness
  (-26 dBFS RMS).
- **Nothing is chosen by score**: no score of these clips was looked at before the files were
  fixed. The files have neutral names (`R01.wav` ...) in a shuffled order, so whoever plays them
  does not see the label.

## Making the calls (`D:\Wednesday_Recording_Sheet_11_Nov.pdf`, part 2)

As the real-call replays (`PROTOCOL.md`, C13-C30): a normal mobile call from the playing phone
to a team member's phone; the receiver answers, mutes and records; the playing phone lies with
its microphone 10-15 cm from the laptop speaker at one volume, never changed; wait two seconds,
play the file, hang up. Each recording is named after its file (`R01` ...). The files are played
only into calls between the team's own phones and never shared.

## Scoring (fixed now; reported, not judged)

The recordings are scored by the voice check's own path (`src.inference.any_audio`, telephone
front end) at its frozen operating point; nothing is chosen on them. Reported per kind: the
number judged correctly out of the calls, with a Wilson interval, and the genuine replays read
as the control (a clone caught while the genuine replays of the same pool pass is caught for
being a clone). With six calls per kind a percentage moves in steps of 17 points; these are a
pilot, like the real calls.

## Kept where

| Item | Where | In git? |
|---|---|---|
| This protocol, the plan | `docs/real_calls/` | yes |
| Playback files | `D:/replay_pack/` | never |
| Call recordings | `data/real_calls/corpus_replay/` | never (`data/` is ignored) |
| Scores and the report | `experiments/results/` | yes (no audio) |
