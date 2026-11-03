# External review: before and after

One page for the viva. The external reviewer (IIT Madras, October 2026) made six comments.
For each: what the project could show when the review was written, what it shows now, and
where the evidence is. The full response, with every check and caveat, is
[`docs/external_review_response.md`](../external_review_response.md).

**The rule throughout:** every choice (threshold, checkpoint, recipe, augmentation) is made
on development data; test sets are scored once per final system. Week 15's three decisions on
spontaneous speech were each written down and committed before their measurement.

| # | Comment | At the review | Now | Evidence |
|---|---|---|---|---|
| 1 | Several seeds and confidence intervals | One training per number, intervals over clips | Five seeds for every system that carries a claim, a paired bootstrap over speakers: deployed adapter 3.27 ± 1.17% over the phone line; on Tortoise 35.59 ± 3.35%, so the published 30.99% was a favourable draw | `external_review_response.md` §1 |
| 2 | Better generalisation to unseen tools, especially Tortoise | Deployed adapter 30.99% on Tortoise over the line; one unseen tool | Three unseen tools. Final recipe (248 speakers, both codecs): new-speaker Tortoise 30.8% → 8.8% over the line, 36.3% → 6.0% over AMR-NB, every seed beating its floor; SSL-AASIST 0.5%. **Open:** IndicF5 16-22%, Chatterbox 12.9-32.0% | `final_v1`, `encoder_spontaneous_v1.md` |
| 3 | More speakers and speaking styles | 21 training speakers; false alarms 3.40%; seen tool on new speakers 10.75% | 248 training speakers: 0.47% and 0.91%; 183 language-control speakers show the gap is the recording domain, not code-mixing. Two spontaneous corpora (HiACC, 200 IndicVoices speakers). **Open:** spontaneous speech. Retraining with it cuts genuine called fake below 2% for every seed but costs 1-11 EER points on read speech, a trade-off by the rule fixed beforehand | `spontaneous_training_v1.md`, `indicvoices_data_v1.md` |
| 4 | The Hinglish-versus-English trade-off | Not measured for English over a phone line | Deployed adapter 2.30% clean, 24.20% over the line (same sample); final recipe 13.44% clean (misses English clones). A language gate chosen on dev audio: English 0.60% clean, 9.80% over the line (routing test sample), at most 0.4 points lost on Hindi-English | `english_final_v1.md`, `routing_v1.md` |
| 5 | How to read the shortcut floor | A floor fitted on Tortoise itself (25.79%) | A floor matched to each detector's own training data (deployed: 37.79%), a paired speaker-level test against it, and the floors above chance marked as weak bars | `external_review_response.md` §5 |
| 6 | Full references and positioning | 10 references, no DOIs | 45 entries, each checked against its publisher; related work rewritten with how this work differs | `external_review_response.md` §6 |

## Beyond the six comments

- **Calibration** (`calibration_v1.md`): at the fixed rule the final systems almost never call
  genuine speech fake; their errors on the newest tools are confident misses.
- **Robustness** (`robustness_v1.md`): MP3, Opus, AAC and a second phone hop raise no EER by
  more than 2 points; a reverberant room and a one-semitone pitch shift are the weak points.
- **The voice check** (`docs/voice_check.md`): any recording judged real or cloned, 99% of
  genuine speech judged real; the retrained detector was measured under a rule committed
  first and not adopted, because one of its two trainings missed the read-speech bar (P-040).
