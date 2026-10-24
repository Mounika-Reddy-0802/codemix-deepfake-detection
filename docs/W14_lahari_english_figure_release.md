# Week 14 — Lahari — English with the final recipe, one overview figure, the model card

Three pieces this week, each closing a gap a reader of the paper would find.

## What I did, and why it is built this way

### English with the final recipe (review comment 4)

The reviewer asked what adaptation costs on English; we had measured it for the deployed
adapter but not for the final recipe, our best system. On the same fixed 6,004-clip ASVspoof
sample, clean and through the G.711 chain, the final recipe's seed-1234 model reaches 13.44%
EER clean and 25.95% over the phone line, against 2.30% and 24.20% for the deployed model.
The error is the opposite kind: it almost never calls genuine English fake (0.90% over the
line, against 7.45% for the deployed model) but catches only 26% of English fakes at the
operating point. More Hindi-English voices and real codec audio broadened its idea of
genuine speech, and English clones now fall inside it. `src/reporting/english_channel.py`
carries the new model; the paper, conclusion and review response say it
(`docs/results/english_final_v1.md`).

### One overview figure

The final systems were spread over several tables. `src/reporting/final_figure.py` reads every
number from the committed results files and draws one figure: four systems against every
attack tool, over the simulated G.711 line and the real AMR-NB codec, with seed spread, the
final-data shortcut floor and chance. The colours are checked for colour-blind separation,
hatching repeats them for greyscale print, and a test rebuilds the table from the result
files. It is in the paper as `fig:overview`.

### The model card

`docs/MODEL_CARD.md` drafts the card for the final recipe's adapter: what it is, intended and
out-of-scope uses (spontaneous speech and English calls are out of scope, with the numbers),
evaluation, limits and training-data licences. Publishing the weights waits for the team's
and the mentor's decision; the card proposes CC BY-NC-SA 4.0, since the weights derive from
MUCS (share-alike) and XTTS-v2 output (non-commercial).

## What is still open

The English result is one seed, and the larger encoder was not scored on English. The
release decision is the team's.
