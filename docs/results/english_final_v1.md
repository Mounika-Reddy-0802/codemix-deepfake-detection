# English with the final recipe (v1)

External review comment 4: what adaptation costs on English. The same fixed 6,004-clip
ASVspoof sample as `english_channel_v1.md` (all 13 attacks, all 67 speakers), clean and
through the G.711 chain, scored with the final recipe's seed-1234 model (248 speakers, both
telephone codecs). Built by `src/reporting/english_channel.py`; numbers in
`experiments/results/english_channel_v1/summary.json`.

| Model | Condition | EER (%) | Genuine called fake at 0.5 (%) | Fakes caught at 0.5 (%) |
|---|---|---:|---:|---:|
| S1 (English only) | clean | 0.51 [0.35, 0.79] | 0.20 | 94.23 |
| S1 (English only) | phone line | 6.50 [5.92, 7.19] | 0.65 | 76.15 |
| Deployed S2 | clean | 2.30 [1.84, 2.85] | 0.05 | 77.35 |
| Deployed S2 | phone line | 24.20 [23.27, 25.30] | 7.45 | 59.97 |
| Final recipe (seed 1234) | clean | 13.44 [12.40, 14.52] | 0.20 | 26.30 |
| Final recipe (seed 1234) | phone line | 25.95 [24.90, 27.05] | 0.90 | 22.95 |

**Reading.** The final recipe is worse on English, but not in the way the deployed model
is. The deployed model rejects genuine English callers over the phone line (7.45%); the
final recipe almost never does (0.90%) and instead lets most English fakes through. More
Hindi-English voices and real codec audio taught it a broader idea of genuine speech, and
English clones now fall inside it. For a Hindi-English service this costs nothing; for
English calls the final recipe should not be trusted, the same conclusion as for the
deployed model, for the opposite reason. One seed; the larger encoder was not scored on
English.
