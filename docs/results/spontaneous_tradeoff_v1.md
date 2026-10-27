# Spontaneous speech and the operating point: a trade-off that chooses nothing (v1)

At its live operating point the final recipe warns every genuine spontaneous 60 s call
(HiACC, 41 calls from 24 adult speakers; `live_final_recipe_v1.md`). This asks whether a
different threshold would have avoided that, and at what cost. HiACC is evaluation-only, so
**nothing is chosen here**: the team decided on 27 October not to split HiACC into a
calibration part, and the live threshold stays the one chosen on dev (0.687).

## Method

The cached live-path window scores of the final recipe (`final_s1234`,
`experiments/calibration_final/*_windows.csv`) are stitched into 60 s calls exactly as the
call study stitches them (`src.inference.calibrate.stitch_calls`), and every call is replayed
through the verdict ladder at the live ladder lengths (caution after 3 fake-leaning windows,
"likely cloned" after 5) for thresholds from 0.05 to 0.95 and the live one. "Warned" and
"caught" count a caution or worse. At the live threshold the sweep reproduces the call
study's table exactly (HiACC 41 of 41 warned, no genuine read-speech call warned, Tortoise 6
of 6, IndicF5 3 of 6, Chatterbox 1 of 6 caught).

Several read-speech sets share their genuine side, so the read-speech column is the worst of
the six sets rather than a pooled rate.

## Results

| Threshold | Genuine spontaneous warned | Genuine read speech warned (worst set) | Tortoise caught | IndicF5 caught | Chatterbox caught | All clone calls caught |
|---|---|---|---|---|---|---|
| 0.15 | 1 of 41 (2%) | 0% | 3 of 6 | 0 of 6 | 0 of 6 | 66% |
| 0.25 | 17 of 41 (41%) | 0% | 3 of 6 | 0 of 6 | 1 of 6 | 70% |
| 0.35 | 31 of 41 (76%) | 0% | 4 of 6 | 0 of 6 | 1 of 6 | 72% |
| 0.45 | 38 of 41 (93%) | 0% | 6 of 6 | 1 of 6 | 1 of 6 | 79% |
| **0.687 (live)** | **41 of 41 (100%)** | **0%** | **6 of 6** | **3 of 6** | **1 of 6** | **83%** |
| 0.80 | 41 of 41 | 20% (3 of 15, CM04) | 6 of 6 | 3 of 6 | 2 of 6 | 85% |
| 0.90 | 41 of 41 | 20% | 6 of 6 | 6 of 6 | 5 of 6 | 98% |

The XTTS-v2 and RVC calls are caught at every threshold from 0.20 up; the full table is
`experiments/results/spontaneous_tradeoff_v1/tradeoff.csv`, per set and label in
`calls_by_threshold.csv`, and the figure is `experiments/figures/spontaneous_tradeoff.pdf`.

## Reading

- **No threshold fixes spontaneous speech.** Every threshold that catches all six Tortoise
  calls (0.45 and above) warns at least 38 of the 41 genuine spontaneous calls. Bringing
  those warnings below half needs 0.25 or lower, where the final recipe catches half the
  Tortoise calls and 4 of the 18 unseen-tool calls (22%).
- **The read-speech false alarms are not what binds.** No genuine read-speech call is warned
  at any threshold up to 0.75, so the live threshold could rise that far for free on read
  speech; the spontaneous calls are already all warned well below it. Genuine spontaneous
  windows score lower than most unseen-tool clone windows: the median spontaneous window is
  at 0.62, the median Tortoise clone window at 0.92 and the median IndicF5 and Chatterbox
  ones at 1.00. That is a ranking problem, not a threshold one.
- **Raising the threshold buys unseen tools at the cost of read speech.** At 0.90 the final
  recipe catches 6 of 6 IndicF5 and 5 of 6 Chatterbox calls, but warns 3 of 15 genuine
  read-speech calls in the worst set. That is a second trade-off, also not chosen here: it
  would need a dev set built from a held-out tool, which the paper's future work names.
- **The fix is in the training data**: spontaneous genuine speech from a speaker-disjoint
  corpus, with clones made from the same speakers' voices so that speaking style cannot
  become the shortcut.

## Limits

One seed of the final recipe. 41 spontaneous calls, stitched across clips and so across
speakers within a call, as in the call study; 6 to 15 calls per clone or read-speech set, so
a step is 7 to 17 points. The ladder lengths are held at the live ones.

## Reproduce

```
python -m src.reporting.spontaneous_tradeoff
```

It reads only the committed window scores, so it runs on any laptop in a few seconds.
