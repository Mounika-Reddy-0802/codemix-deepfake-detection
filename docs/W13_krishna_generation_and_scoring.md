# Week 13 — Krishna — new clones, scoring on new people, and the paper reframed

The external review asked for more unseen speakers and a better answer on unseen
tools. This week I made the clones those tests need, scored every model on them,
and rewrote the paper's frame around what the language-control test found.

## What I did, and why it is built this way

### Four clone sets, made on Kaggle as saved runs

`src/data/generate_cli.py` runs one tool on one job table, sharded across both T4s,
and `notebooks/kaggle_w13_generate_{A,B,C}.ipynb` drive it:

| Set | Tool | Speakers | Made | Passed QA |
|---|---|---|---|---|
| CM05 | XTTS-v2 | 60 new test speakers | 1,800 | 1,770 |
| CM06 | Tortoise | the same 60 | 240 | 222 |
| CM07 | IndicF5 | 54 test speakers | 610 | 526 |
| CM08 | XTTS-v2 | 227 extra adaptation speakers | 1,816 | 1,759 |

Three things went wrong and are fixed in the code, not worked around:

- **Tortoise ran at eleven minutes a clip.** tortoise-tts 3.0.0 ships with its key-value
  cache off. Turning it on changes nothing about the audio and brought it to about 80
  seconds a clip per GPU (`heldout_tts.load_tortoise`).
- **IndicF5 never imported.** First Jupyter's plotting backend leaked into the tool's
  environment (`generate_cli` now sets a plain one), then the newest pyarrow refused
  numpy 1.26, which IndicF5 pins. pyarrow 19.0.1 and datasets 3.6.0 are pinned; the
  whole import chain was checked locally before the rerun.
- **Runs stopped when the laptop did.** The notebooks now run as saved versions: code,
  environments and raw clones live in `/kaggle/temp`, only the results zip is output,
  and every run stops itself at 10.5 hours so it always saves inside Kaggle's limit.

CM07 rejected 84 clips, more than the others. I checked the screen was not unfair to
Devanagari: the rejected clips stall at 2-6 characters a second against a median of
12.3 for the kept ones, so they are real failures.

### Scoring sets that cannot compare different people

`src/data/score_sets.py` unpacks a run's zip into the extension bundle and stacks each
attack under the genuine clips of the same speakers. A spoof speaker with no genuine
clips, or a training attack, stops the build. `score_ext` holds the 60 new speakers
(2,400 genuine, CM05, CM06); `score_cm07` the IndicF5 speakers' 651 all-Hindi genuine
clips and CM07.

### Every model on the new sets, on the laptop CPU

Five models (S1 and the four S2 adapters), clean and phone line, 20 runs, resumable
(`D:/dfmodels/run_week13_scoring.py`). `src/reporting/new_speakers.py` reports each
attack with a speaker-bootstrap interval and the per-speaker spread
(`docs/results/new_speakers_v1.md`):

- the deployed model reaches **9.4%** on new speakers with the seen tool over the
  phone line; the median speaker is 2.9% and the worst 17.1%;
- **Tortoise 29.4% and IndicF5 30.6%** over the phone line: the second unseen tool is
  no easier than the first;
- S1 stays at 44.8-52.7%.

`src/training/floor_new_sets.py` runs the review's matched-floor comparison on these
sets with `floor_compare`'s own functions (`docs/results/floor_new_sets_v1.md`). Every
system beats its floor on the new speakers, but the Tortoise floors there sit at
40-58%, so that bar is weak; on IndicF5 no system clears its floor.

### The release

`src/reporting/release.py` builds one archive of everything needed to audit a number:
manifests, job tables, pools, configs, per-clip scores and summaries, with sha256
checksums. It takes only git-tracked files from an allowlist and refuses audio,
weights, archives and secrets (`docs/release.md`).

### The paper

The title now leads with the finding: *Recording Domain, Not Code-Mixing*. The abstract,
introduction and conclusion state the language-control result, the 60-speaker result
and the second unseen tool. `paper/short/main.tex` is a four-page version built from
the same numbers. `docs/external_review_response.md` records the week's answers to
comments 2, 3 and 5.

## What is still open

The runs that were queued when I wrote this have landed, and Mounika's and Lahari's
records carry them: 248 training speakers beat 21 in every seed, the final recipe (248
speakers over both telephone codecs) and the larger encoder beat their shortcut floors in
every seed, and Chatterbox became the third unseen tool. Two things stay with me. The
live call system still runs the deployed S2; moving it to the final recipe, or to the
larger encoder, needs its real-time cost measured on the call path first. The four-page
version has not been compiled here (no LaTeX on this machine), so its page count is still
to be checked in Overleaf.
