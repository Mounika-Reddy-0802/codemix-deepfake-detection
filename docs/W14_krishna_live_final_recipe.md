# Week 14 — Krishna — the live system on the final recipe, and what a caller meets

Week 13 ended with a better detector on paper: the final recipe (248 speakers, both
telephone codecs) beat the deployed adapter on ten of eleven test sets. The live call system
still ran the deployed adapter, and every number we had was per clip. This week moved the
live system's operating point to the final recipe and measured both models the way a caller
meets them: whole calls through the live scoring path.

## What I did, and why it is built this way

### The operating point, chosen on dev only

`src.inference.calibrate` already scores clips through the live path (4 s windows every 2 s,
level-normalised, smoothed) and replays them through the verdict engine. I scored the dev
set and 300-clip label-stratified subsets of seven test sets (1,000 clips of HiACC) with the
final recipe's seed-1234 model, and the four new sets with the deployed model, on the laptop
CPU. The clips sit in several bundle folders, so a combined data root of hard links points at
all of them without copying audio.

The final recipe's threshold was chosen by the week-8 rule, on dev only: the most sensitive
threshold and ladder whose genuine 60 s dev calls raise at most 5% alerts of any level. It
came out at 0.687, with a caution after three fake-leaning windows and "likely cloned" after
five (`configs/threshold_final.yaml`).
Re-running the rule on the deployed model's dev windows gives its published threshold
exactly, so the two are compared like for like.

### What a caller meets

On 60 s calls the final recipe warns no genuine read-speech caller in any set, where the
deployed model warned 44% of the new speakers' calls, and it catches every Tortoise call
(33% for the deployed model). It catches fewer IndicF5 (50% against 67%) and Chatterbox
(17% against 33%) calls. Spontaneous speech is worse at the call level than per clip: a
single clip is rarely warned, but over a minute every genuine spontaneous call is warned
(a caution or worse) by either model (`docs/results/live_final_recipe_v1.md`).

### Using it

The live system reads its model and threshold from the environment, so the final recipe runs
with `DFD_CHECKPOINT` pointing at its checkpoint and `DFD_THRESHOLD_FILE` at
`configs/threshold_final.yaml`; `configs/threshold.yaml` stays the deployed model's, so the
published demo still reproduces. `live_call/README.md` and the system chapter
(`sec:system-calls`, Table `tab:calls`) say so.

### The real phone-call test, ready to run

The team and the mentor agreed to a small test on real calls: genuine calls from the three of
us and calls with clones of our voices played in, over a normal mobile network. Everything
except the calls themselves is built and tested:

- `docs/real_calls/PROTOCOL.md`: the steps for the day, the 30-call plan, and what is kept
  where; `docs/real_calls/SCRIPTS.md`: two scripts of twelve code-mixed sentences (an
  everyday call and an urgent request for money and a code) plus four reference sentences,
  none needing a gendered first-person verb; `docs/ethics/real_call_consent_form.md`: the
  consent form and the mentor's approval box, an addendum to the v2 sign-off, which covered
  public corpora only.
- `src/data/real_call_pack.py` turns each speaker's reference recording into cloning packs
  for XTTS-v2 (the seen tool) and Chatterbox (the hardest unseen tool), in the job-table
  format `src.data.generate_cli` already reads, and joins the clones into one-minute calls
  at the same level as each speaker's own reading, so cloned and genuine replays reach the
  phone through the same loudspeaker.
- `notebooks/kaggle_w14_team_clones.ipynb` makes the clones on Kaggle with the existing
  generation code, ethics gate and quality screen; its post-generation steps were run on
  fake clips before handing it over.
- `src/inference/real_calls.py` decodes the phone's recordings (m4a and the like, through
  ffmpeg), scores each call as recorded and after a G.711 round trip (what the carrier
  delivers), with both models through the calibration path, and replays every call through
  the verdict ladder at each model's frozen operating point. A dry run with synthetic audio
  and both real checkpoints went from recording to report on the laptop's CPU.
- `realcall` is an evaluation-only source in `src/data/build_manifests.py`, with a leakage
  test, so no training manifest can contain the recordings or the clones.

### The weights package, ready for the release decision

Publishing the final recipe's weights waits for the team and the mentor, but the package is
built and checked so that decision is a single upload. `src/inference/model_package.py`
exports a checkpoint into `model.safetensors` (the whole state dict, since the adapters sit on
Stage-1's fine-tuned encoder), `config.json` (training config, dev metrics, the live operating
point and the windowing), the model card with Hugging Face metadata, and checksums that are
verified before loading. safetensors holds tensors only, so loading runs no pickled code.
Exported from `final_s1234` (95,454,083 parameters) and loaded back, it scores random windows
identically to the checkpoint. The steps to publish are in `docs/release.md`. I took this on
rather than Lahari, who owns the release, because she already had the most commits this week.

### The real phone calls

With everyone's written consent and the mentor's approval we called each other over a mobile
network: genuine live calls, our own readings replayed from a laptop speaker, and clones of our
voices replayed the same way, recorded on the receiving handset and scored through the live path
(`docs/results/real_calls_v1.md`). All 30 calls are in, from the three of us. The deployed
model takes every replayed call and every live call of two of us to "likely cloned", but none of
the third speaker's live calls. The final recipe warns three of twelve genuine live calls and
takes five of six XTTS-v2 clone calls to "likely cloned", but no Chatterbox call (it cautions
one).

To find out why the deployed model fails I rebuilt the call chain on our direct recordings, one
stage at a time (`src/inference/real_call_codecs.py`): over the six readings its mean
P(genuine) falls from 0.83 direct and 0.75 through the simulated line to 0.50 through the real
AMR-NB codec, while the call recorder adds little; the final recipe, trained on AMR-NB, stays at
0.82-1.00 for every reading at every stage. So the codec is the largest step for the speakers the
deployed model fails on: per reading, two of six cross its threshold there, one stays above it,
one was below it before any codec, and P1's two stay at 0.87-1.00 throughout. Which of the final recipe's two changes, more
voices or the real codec, keeps it up these calls cannot say; offline, adding AMR-NB alone to
the deployed recipe's training recovers most of the codec loss (7.59% against 22.37% on
new-speaker XTTS over AMR-NB, three seeds). The report also
gives scores by time in the call (`score_by_phase`): genuine calls look least genuine in their
first seconds, and the first window spans the quiet before the first word, which passes the
silence gate and is scored as speech. I first put the early false alarms down to the handset's
noise suppression settling; the window levels said otherwise. I then said they came mostly from
that first window, which was also too strong. Replaying the calls with the first window treated
as silence (`without_first_window`, a diagnostic that chooses nothing) clears two of the three
warned genuine live calls over G.711; as recorded it clears one and cautions another, and one
call still reaches "likely cloned" with two speech windows 8-14 s in. So the quiet start is where
every warned live call's run begins, but not the
whole cause, and the record now says exactly that.

The chain is also a figure in both papers (`python -m src.reporting.real_call_figure`, copied to
`paper/figures/real_call_chain.pdf`): every point is read from the result files, with the genuine
live calls as the last stage (with all 30 calls in, 0.38 for the deployed model, 0.94 for the
final recipe). While
setting the server up on the final recipe I found that the upload API named
`configs/threshold.yaml` as its threshold source whatever file the server was started with; the
value was right, the label was not. It now names the file in use, and a test holds it there.

A swapped playback file would put a clone under a genuine call's number, and nothing in the
scores would show it. So every replay recording is now matched back to the file played into it
(`python -m src.inference.real_calls check`, run before scoring). My first try correlated fine
energy envelopes and matched only 3 of 12 calls: the phone's gain control and the codec
reshape the envelope. Coarse speech-activity tracks (100 ms frames, active within 25 dB of the
loud frames) survive both, and all 18 replays match their planned file (0.54-0.95, at most 0.40
for any other file). A test plays one file under the wrong call number and checks it is caught.

The report now also breaks every result down by speaker (`by_speaker`), because the totals hid
something: all three warned genuine live calls are P2's (3 of 4) and none of P1's or P3's
eight is warned, and the deployed model splits by speaker too (none of P1's live calls warned,
every one of P2's and P3's). The rates depend on who is calling, and both papers now say so. And since the table in the paper is typed by hand, a test now reads it
back and checks every cell against `summary.json`, so a count left over from the interim run
fails the suite rather than reaching the paper.

I rehearsed the demo with the final recipe on the CPU: the server reports the right threshold
file, and genuine, XTTS-v2 and Tortoise clips get the expected verdicts when uploaded. Played as
a call, the Tortoise demo call stays green for both models, so I built a second set of demo
calls verified on the final recipe (`demo_clips_final`, outside git) and the demo script now
warns the presenter before that call.

For the final review I added three calls to that set (`demo_calls_v2`, outside git; verdicts
in `experiments/results/demo_calls_v2_verification.json`): a genuine new speaker, a Tortoise
call the final recipe catches, and a genuine spontaneous HiACC speaker. The Tortoise call is not
hand-picked from a list of hits: I built a 45 s call for every one of the 15 CM04 Tortoise
speakers, replayed each, and the final recipe catches 12 of 15; the kept call is the caught
speaker with the most clips, and the missed call stays in the set, labelled. The spontaneous
call is warned by both models, as the call study predicts, and is in the set to show that limit.

The day did not go as planned: we were at home in three places, so the receiving and calling
phones differ by call, the references came from different phones, and P1's calls followed a day
later. The protocol records each change.

## Numbers

| What | Number | Source |
|---|---|---|
| Final recipe's live operating point, chosen on dev only | threshold 0.687; caution after 3 windows, "likely cloned" after 5 | `configs/threshold_final.yaml` |
| 60 s calls: genuine new speakers warned | deployed 44%, final recipe 0% | `live_final_recipe_v1.md` |
| 60 s calls: CM04 Tortoise calls caught | deployed 33%, final recipe 100% | same |
| Real calls (30, three speakers): deployed model, calls taken to "likely cloned" | 26 of 30 (all 18 replays; live calls 8 of 8 for P2 and P3, 0 of 4 for P1) | `real_calls_v1.md` |
| Real calls: final recipe, genuine live calls warned / "likely cloned" | 3 of 12 / 1 of 12 | same |
| Real calls: final recipe, XTTS-v2 / Chatterbox calls caught | 5 of 6 / 0 of 6 | same |
| Call chain, deployed, mean of six readings: direct, simulated G.711, real AMR-NB | 0.83, 0.75, 0.50 | `codec_chain.csv` |
| Call chain, final recipe, every reading at every stage | 0.82-1.00 | same |
| Replay recordings matched to their planned file | 18 of 18 (0.54-0.95; others at most 0.40) | `replay_check.csv` |
| Weights package against the checkpoint | identical scores | `docs/release.md` |

## How to run

```bash
# the real-call test, after the recordings are in data/real_calls/ (consent first)
python -m src.inference.real_calls check --recordings data/real_calls/recordings \
    --playback data/real_calls/playback --out experiments/results/real_calls_v1/replay_check.csv
python -m src.inference.real_calls prepare --recordings data/real_calls/recordings --out data/real_calls/scored
python -m src.inference.real_calls score --audio data/real_calls/scored --deployed <deployed .pt> \
    --final <final .pt> --out experiments/results/real_calls_v1
python -m src.inference.real_calls report --audio data/real_calls/scored \
    --windows experiments/results/real_calls_v1 --md experiments/results/real_calls_v1/report.md
python -m src.inference.real_call_codecs --references data/real_calls/references --speakers P1,P2,P3 \
    --deployed <deployed .pt> --final <final .pt> --out experiments/results/real_calls_v1/codec_chain.csv
python -m src.reporting.real_call_figure

# the weights package
python -m src.inference.model_package export --checkpoint <final .pt> --out <package dir>
python -m src.inference.model_package verify --checkpoint <final .pt> --package <package dir>
```

Each was run on the laptop's CPU before committing.

## What is still open

The counts are small (6 to 41 calls per cell). The final recipe's live threshold uses one
seed. The real-call test is a pilot of 30 calls in our own voices; the live system's own
telephone carrier is still unexercised, and spontaneous callers would be warned by either model.
The real calls support running the final recipe as the live default, which is not yet done.
On real calls the final recipe warns late for two speakers (their XTTS-v2 calls it catches reach
a caution 18-32 s in), its false alarms on live calls start with the quiet before the first word
but do not end there, and the warned genuine live calls over G.711 are one speaker's from one
handset. A stricter speech gate and the
operating point both need a separate set of real development calls.
