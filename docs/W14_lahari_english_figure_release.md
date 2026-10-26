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

### The conference paper

`paper/short/main.tex` is now a six-page paper in the IEEE two-column format, the page limit
most IEEE conferences use, instead of four. It adds what the four-page version had to leave
out: the data and attack tables, the training settings, the final recipe against the
deployed adapter and the larger encoder in one table, the overview figure, the live-call
system with its architecture figure and the 60 s call results, and a limitations section.
Every number is taken from the long paper. The page count is checked in Overleaf, since the
laptop has no LaTeX.

### A consistency pass over both papers

Before submission I checked every number that appears in more than one place, in both papers
and the model card, against the result files and the manifests. Every reference, citation and
table row resolves. These did not agree, and are fixed:

- The deployed recipe is retrained inside each comparison (the seed study, the 248-speaker run
  and the final-recipe run), so its five-seed numbers differ a little between tables: CM04 over
  the channel is 35.59%, 33.46% and 36.2%. Both papers now say so. Every comparison is still
  made within one run, and the differences sit inside the seed spread.
- The attack table listed CM01 (3,998 clones of the training pool) as the seen-tool test set.
  The seen-tool test set is the 2,400 clones of the evaluation pool (`codemix_eval.csv`). CM01
  was kept out of training so that S2 and S3 see the same data (P-028).
- IndicF5 and Chatterbox share 651 genuine clips of 75 speakers, and the clones that pass
  screening cover 54 and 61 of them. The tables had given 54 and 61 as the speakers of each set.
- Smaller fixes:
  - The language-control set has equal numbers per speaker only where the recordings allow
    (439, 412 and 439 clips).
  - The measured SNR is 18.8 dB against the 20 dB requested; the text had said the two match.
  - The figure of 12 to 15 speakers per test set holds for the original test sets only.
  - S3's 0.83% is the lowest seen-tool error of the three systems, not of the whole study:
    SSL-AASIST is lower.
- In the 60 s call tables, "warned" counts any alert, a caution or worse. The final recipe's
  ladder is a caution after three windows and "likely cloned" after five, but the model card
  had called the first level a warning.

The real-call numbers were left alone here, since they change when the last calls are in.

### The datasheet, brought up to date

`docs/datasheet.md` still described week 9. Its attack table listed CM05-CM09 as
"not made yet" under IDs first reserved for kNN-VC, IndicSynth and an AffectDF subset, none
of which was built, and it said S3 trains on CM01. The table now lists the ten sets that
exist, with the jobs from each committed table and the clips and speakers in each manifest
(CM05 1,770 of 1,800; CM06 222 of 240; CM07 526 of 610; CM08 1,759 of 1,816; CM09 525 of
610), the eval-pool set, and who sees each one. The pools table now names the later
extensions, and a new command at the end reproduces the new rows. The SSL-AASIST tables in
both papers also gained the original test pool (0.25% against 1.13%), the one set of the
eleven they had left out, so "better in every seed on eight of eleven sets" can be checked
against the table.

### Figure fonts for submission

matplotlib writes Type 3 fonts into PDF figures by default, and IEEE PDF eXpress rejects
them, so the paper would have failed the submission check. The shared figure style and the
architecture figure now embed TrueType fonts (`PDF_FONTS` in `src/reporting/figures.py`), the
two committed PDF figures are drawn again (the PNGs come out byte for byte the same, so
nothing in the drawing moved), and two tests fail if a Type 3 font comes back. Any figure added
later must be drawn through `_style()` or set `PDF_FONTS` itself.

## What is still open

The English result is one seed, and the larger encoder was not scored on English. The
release decision is the team's.
