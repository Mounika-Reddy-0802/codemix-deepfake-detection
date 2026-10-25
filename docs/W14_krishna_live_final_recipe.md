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
threshold and ladder whose genuine 60 s dev calls raise at most 5% warnings. It came out at
0.687 with a warning after three fake-leaning windows (`configs/threshold_final.yaml`).
Re-running the rule on the deployed model's dev windows gives its published threshold
exactly, so the two are compared like for like.

### What a caller meets

On 60 s calls the final recipe warns no genuine read-speech caller in any set, where the
deployed model warned 44% of the new speakers' calls, and it catches every Tortoise call
(33% for the deployed model). It catches fewer IndicF5 (50% against 67%) and Chatterbox
(17% against 33%) calls. Spontaneous speech is worse at the call level than per clip: a
single clip is rarely warned, but over a minute every genuine spontaneous call raises a
warning with either model (`docs/results/live_final_recipe_v1.md`).

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

## What is still open

The counts are small (6 to 41 calls per cell). The final recipe's live threshold uses one
seed. No caller has yet been tested on a real phone network, and spontaneous callers would be
warned by either model.
