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
- A second check, of the slides, the viva answers and the README against the paper, found
  two errors in the paper itself. "Over the telephone channel every AUC falls below 0.5"
  is true for two of the three sets (0.381 and 0.425), not CM04 (0.561). And the 94.8% of
  genuine speech that S1 calls fake was measured on clean audio, so it explains the clean
  failure, not the 58.42% over the channel. The six-page paper also said every test set was
  passed through AMR-NB (only the new-speaker, IndicF5 and CM04 sets are) and that more
  voices close the gap on three unseen tools (on Tortoise; IndicF5 and Chatterbox stay hard).

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

### A reviewer's read of the six-page paper

A read of the six-page paper as a reviewer would read it found places where the text said more
than the experiment shows, or left a reader to guess. Fixed in that paper, and in the long
paper where it says the same thing:

- The 248-speaker run added genuine speakers and seen-tool clones of them together, so it
  cannot say the unseen-tool gains come from more genuine speech. Both papers now say the
  gains do not come from exposure to the attacks and that a broader idea of genuine speech
  is the likely reason, which the experiment does not separate.
- "Keeps English" became "keeps clean English" (English over the line is 24.20%), and the
  table column says clean. "Closes most of the gap on unseen generators" became "on the
  unseen Tortoise generator".
- Missing context a reader needs: who the 21 training speakers are (8 adaptation speakers
  and 13 RVC source speakers), that SSL-AASIST is fine-tuned in full with RawBoost, that the
  AASIST comparison is over G.711 in three seeds against our adapter trained on the same
  data, that Chatterbox is left out of the "eight of eleven sets" count (one adapter seed),
  that SSL-AASIST was not scored on English, and that three of the five deployed seeds beat
  the eval-pool floor.
- "Recording domain" is now defined where it is first claimed: the corpus as a whole, its
  recording conditions and its speakers, which the language-control test does not separate.
  HiACC is described as differing from the lectures in both speaking style and recording.
- The ladder's levels now have one name each, as in the long paper and the code: a caution,
  then "likely cloned" with a spoken warning. The six-page paper had called the top level "a
  warning" in the live-system section and "likely cloned" in the real-call paragraph, and
  said every spontaneous call "raises a warning" where the table counts a caution or worse.

A second, independent read of the six-page paper, each point checked against the source
before it was changed, found claims whose comparison changes two things at once:

- S3 starts from the self-supervised checkpoint, never from S1, so it never had English to
  lose. Both papers now say it is at chance on English, on which it was never trained, and
  that the adapter keeps clean English because it is built on the English detector. Whether
  fully fine-tuning S1 would keep English was not tested, and the long paper says so.
- Adding RVC to training also added the 13 genuine speakers its conversions came from, and
  the XTTS-only adapter beat its Tortoise floor in only one seed of five. "Adding RVC is what
  removed the Tortoise advantage" is now stated as what was measured, with the confound.
- AASIST does not gain more from the added voices than it differs from the adapter on every
  set: on IndicF5 it barely moves and its RVC error doubles. The papers say where it holds.
- The floors that sit above chance are called uninformative, not weak; the floor claims are
  limited to our systems (the AASIST baseline has none); Table 2 says it is one run; Table 4
  says what its SSL-AASIST column and "genuine called fake" are and that Chatterbox is left
  out of the set counts. The six-page paper drops the probe evidence, which was about the
  adapter, not S1, and rested on five calls.

A third read, of the data chapter against the repository itself, corrected what the chapter
said the data and checks are: the 25/10/15 pools come from 50 shortlisted speakers, not all
520, and the extension pools hold the other 470; the build check on real training configs is
`tests/test_config_guard.py`, while `tests/test_splits.py` tests the checks on synthetic
manifests; the code-mixed sets range from 0.3:1 (CM04) to 1.5:1, not all near 1.5:1;
ASVspoof is 109.7 hours; AMR-NB carries 2G and 3G calls, AMR-WB 4G; IndicF5 is no longer
called the newest generator, since Chatterbox came later; and the licence paragraph now
states that XTTS-v2 output is non-commercial under the Coqui Public Model Licence, as the
licence register requires.

A second fresh read of the six-page paper, after those fixes, found what they had left: the
adapters keep clean English in the deployed configuration only (the final recipe is at
13.44%); "not more attack tools" was never tested, so it now says no new tool was added; the
overview figure's caption said S1 was not scored on AMR-NB, but it was (40-50%; it is just
not drawn), and its floor tick belongs to the two final-data systems only, which the caption
and `final_figure.py` now say; a false alarm is defined (P(genuine) below 0.5); S2's trained
parameters include the pooling; the 43.9% spontaneous figure is one seed inside the spread of
single trainings; Chatterbox is named among the open problems; and the 0.85% detector is
ours on the same benchmark, not a replication of the cited XLS-R recipe.

## What is still open

The English result is one seed, and the larger encoder was not scored on English. The
release decision is the team's.
