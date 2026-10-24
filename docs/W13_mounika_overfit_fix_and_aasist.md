# Week 13 — Mounika — more training voices, an outside baseline, and the results chapter

Two of the improvements the paper needed were mine: train the deployed recipe on far
more genuine speakers, and compare our detector with an established architecture on
the same data. Both are built and checked; both run on Kaggle as saved versions.

## What I did, and why it is built this way

### The corpus-overfit fix: 21 genuine speakers become 248

The deployed S2 adapts on 21 genuine speakers. Its false alarms on new recordings and
its unseen-tool errors could come from having heard too few voices, so the test changes
one thing only: the training manifest. `src/data/extended_training.py` appends the 227
extra adaptation speakers (3,405 genuine clips) and their CM08 XTTS clones (1,759) to
the deployed manifest: 8,018 clips, 248 genuine speakers. It refuses any added speaker
outside the `adapt_ext` pool or in the dev set, and `config_guard` passes the new
config (`configs/train_lora_ext_rvc_channel.yaml`), which differs from the deployed one
only in the manifest and the output folder.

`notebooks/kaggle_w13_overfit_fix.ipynb` trains the extended recipe and the deployed
recipe with the same five seeds, alternating so every finished pair compares, and scores
each model over the phone line on the original test pool, CM04, the RVC half, the 60 new
speakers (XTTS and Tortoise), IndicF5 and the language-control false alarms. Nothing is
chosen on a test set: the comparison is fixed in advance and reported seed for seed.

### The AASIST baseline

`src/baselines/aasist_model.py` is the AASIST network copied unchanged from
clovaai/aasist (MIT; licence in `third_party/aasist/`), with its published ASVspoof 2019
LA weights, checked by sha256. `src/baselines/aasist.py` scores any manifest with our own
EER, per-attack and false-alarm code and fine-tunes from the published weights with
class weights and best-dev-epoch selection. Two decisions worth recording:

- **The port is checked against the paper.** Zero-shot on 300 clips of our ASVspoof
  evaluation subset it gives 0.67% EER; the published figure is 0.83%.
- **Scores are log-odds, not probabilities.** A confident network saturates P(bonafide)
  to 0 or 1 in float32 and the ties blur the EER; the logit difference ranks clips the
  same way and never ties. False alarms are read at log-odds 0, which is P = 0.5.

Zero-shot on an 80-clip sample of phone-line CM04 it behaves like S1: most genuine
clips are called fake. `notebooks/kaggle_w13_aasist_baseline.ipynb` scores it zero-shot
on every set, then fine-tunes it on the deployed training set and on the extended set
with the same five seeds, and scores every model on the same sets.

### Both notebooks are built for saved runs

Saved Kaggle runs failed to mount `codemix-bundle-normalised` twice in October, and a run
that ends that way keeps no output. The 8,403 phone-line clips these runs need from it,
and the Stage-1 checkpoint, now come from a new dataset, `codemix-w13-core`; the clones
come from `codemix-week13-clones`. Each notebook links the three datasets' clips into one
data root, checks every manifest clip is present before training, keeps only its results
zip as output and stops itself at 10.5 hours. Both were simulated end to end (normal run,
a training failure, a scoring failure, the time limit) before being committed.

### The results chapter

Three additions to `results.tex`: the language-control result (the gap follows the
recording, not the language mix), the new-speaker and IndicF5 table, and the reading of
those results against their matched floors, including that the Tortoise floors on new
speakers sit near chance and that no system clears its IndicF5 floor.

### What the runs found

**More voices (run 1).** The extended recipe beat the deployed one in every one of five
seeds on new speakers (0.91% against 10.75% EER), on both unseen tools (Tortoise 13.59%
against 29.82%, IndicF5 27.25% against 35.40%) and in false alarms (0.47% against 3.40%).

**AASIST (run 2).** Zero-shot it rejects 82.3% of genuine Hindi-English speech, the same
failure as our English-trained S1. Fine-tuned on the 248 speakers it gains as much as our
adapter, better on Tortoise and worse on the seen tool, RVC and false alarms.

**The final recipe (run 4).** `src/data/final_recipe.py` stacks the 248-speaker set over
both telephone codecs (simulated G.711 and real AMR-NB, 16,036 clips). Five paired seeds
against the deployed recipe: better on ten of eleven test sets, in every seed on nine;
over the real AMR-NB codec new-speaker Tortoise falls from 36.3% to 6.0% and IndicF5 from
46.5% to 21.3%. The cost is RVC, worse in all five seeds (8.5% against 6.1%)
(`docs/results/final_recipe_v1.md`).

**SSL-AASIST, the larger encoder (runs 5, 7 and 8).** `src/baselines/ssl_aasist_model.py`
loads the published XLS-R 300M checkpoint into a Hugging Face network without fairseq.
Run 5 first failed when scoring its fine-tuned models: our own saved weights held the
positional convolution's folded weight, which a fresh network did not accept. The fix
folds it before loading, with a test that saves and reloads a state dict. Zero-shot it
rejects 73.1% of genuine speech. Trained on the final-recipe data with the same five
seeds as the adapter, it is better in every seed on eight of eleven sets (exact
Mann-Whitney p = 0.008 on seven): new-speaker Tortoise over AMR-NB 0.51% against 6.04%,
CM04 1.75% against 17.73%, Chatterbox 12.86% against 31.98% (one adapter seed). False
alarms and both IndicF5 sets are not separated (`docs/results/ssl_vs_adapter_v1.md`,
built by `src/reporting/w13_runs.py`).

### The paper

New results subsections for the final recipe and both outside baselines, each with its
table; the abstract cut from about 450 words to about 240 and rewritten around the final
recipe and the larger encoder; the conclusion and the four-page version updated to match.
Every number in them comes from a committed results file. Once Lahari's floor run
landed, the final-recipe and larger-encoder sections, the abstract, the conclusion and the
results summary state that both final systems beat their shortcut floor in every seed, with
the caveat that three of those floors are above chance; the abstract is back under 250
words, and the Chatterbox subsection now comes before the tables that use CM09.

## What is still open

The larger encoder's cost on a live call is not measured; the adapter is what runs there.
IndicF5 stays at 15.8-21.8% for both encoders. Neither system has been tested on the
spontaneous HiACC speech after the final recipe.
