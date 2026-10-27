# Early warning: a stricter speech gate for the first seconds of a call

**Status: measured, chooses nothing.** Numbers from `experiments/results/early_warning_v1/`
(`summary.json`, `report.md`), written by `python -m src.inference.early_warning` from the
real-call window scores already in `experiments/results/real_calls_v1/`.

## Why

On the 30 real calls the final recipe's false alarms on genuine live calls start in the first
window, which spans the quiet before the caller's first word: it passes the live path's
-50 dBFS silence gate, is normalised and is scored as speech (`real_calls_v1.md`). The obvious
fix is a stricter gate for the first seconds of a call. The question here is whether such a
gate can be set without real development calls, from the dev clips alone.

## What was done

- **The gate, set on dev only.** Windows that end in the first 10 s of a call and are quieter
  than -29.87 dBFS are treated as silence (not scored, the smoothing restarts after them).
  -29.87 dBFS is the level below which 1% of the dev windows fall
  (`experiments/calibration_final/dev_windows.csv`, every one of them speech), so on the dev
  clips the gate would drop at most one speech window in a hundred.
- **Replayed, not re-scored.** Each model's stored window scores on the real calls are passed
  through the same verdict engine at the same frozen operating point, without and with the
  gate, as recorded and over G.711. No setting is chosen on the real calls.

## Results

Over G.711 (as the carrier delivers the calls):

| Calls | n | Final: warned (likely cloned), no gate | with gate | SSL-AASIST, no gate | with gate | Deployed, no gate | with gate |
|---|---|---|---|---|---|---|---|
| Genuine, live | 12 | 3 (1) | 1 (0) | 1 (0) | 1 (0) | 8 (8) | 8 (8) |
| Genuine, replayed | 6 | 1 (1) | 1 (0) | 1 (1) | 1 (1) | 6 (6) | 6 (6) |
| XTTS-v2 clones | 6 | 5 (5) | 5 (5) | 6 (6) | 6 (6) | 6 (6) | 6 (6) |
| Chatterbox clones | 6 | 1 (0) | 0 (0) | 3 (1) | 4 (1) | 6 (6) | 6 (6) |

Median time of the first caution on the XTTS-v2 calls: final recipe 18 s without the gate,
20 s with it; SSL-AASIST 10 s and 14 s; deployed 9 s and 14 s.

**Over G.711 the gate does what it was meant to for the final recipe.** Two of its three warned
genuine live calls are cleared (C06, C08), the third (C05) only cautions, at 16 s, and the
genuine replay C17 no longer reaches "likely cloned"; its XTTS-v2 catches are unchanged.

**As recorded it does not.** The final recipe then warns five genuine live calls instead of
four (C07 is added; one "likely cloned" instead of two), and its cautioned Chatterbox calls
fall from two to one.

**It costs every model time, and it silences speech.** It treats 83 of the 120 windows that end
in the first 10 s of these calls as silence, not only the quiet before the first word: the
real calls are much quieter than the dev clips (speech in the first 10 s at a median of about
-36 dBFS here, against -23 dBFS on dev, which is normalised), so a level that spares the dev
speech still removes most early speech on a real handset. Warnings come later for all three
models, by 2 to 5 s at the median.

**SSL-AASIST does not need it.** Its genuine live calls show no early dip (mean 0.98 over the
first 10 s), its one warned genuine live call is cautioned 48 s in, and the gate delays its
XTTS-v2 catches and adds one Chatterbox caution (C27).

## What it means

A gate set from dev clips is not transferable to real calls: the dev audio is normalised and
the calls are not, so the same level means different things in the two. The gate, and the
operating point with it, have to be set on a separate set of real development calls, which
is what both papers say. For the final recipe, the first-window diagnostic in
`real_calls_v1.md` (two of three warned genuine live calls cleared over G.711, none of the
early speech windows touched) remains the best case for this kind of fix, and the warned call
C05, whose run includes two speech windows 8-14 s in, is beyond it.

## Limitations

- One gate design (an absolute level over a fixed first stretch), one dev rule, 30 calls.
  A gate relative to the call's own speech level would need the level of speech that has not
  arrived yet for the first window, so it would have to wait one hop; it was not tried.
- The windows were scored with the live path's -50 dBFS gate; the stricter gate is applied on
  top of the stored scores, which is what the live path would do with the same windows.

## Reproduce

```
python -m src.inference.early_warning --dev experiments/calibration_final/dev_windows.csv \
    --windows experiments/results/real_calls_v1 --out experiments/results/early_warning_v1
```
