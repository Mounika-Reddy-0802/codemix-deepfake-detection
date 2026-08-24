# Gap matrix v1 — the English-trained detector on code-mixed Hinglish

**First measurement of the project's central claim.** One Stage-1 checkpoint,
trained on ASVspoof 2019 LA (English) and nothing else, scored on two columns.

Checkpoint: `checkpoints/baseline/best.pt` (wav2vec2-base, epoch 1, dev EER 0.0009)

## Result

| Column | Clips | EER | AUC |
|---|---|---|---|
| **English, unseen attacks** (ASVspoof eval, A07–A19) | 71,237 | **0.87%** | 0.9990 |
| **Code-mixed Hinglish** (MUCS real + XTTS spoof) | 4,434 | **44.65%** | 0.533 |

A **51× degradation**, from near-perfect to approximately chance. 95% CI on the
code-mixed EER is [42.7%, 45.8%], so the effect is far larger than sampling noise.

## The failure mode is not what the EER implies

| Set | mean P(bonafide) | classified "bonafide" |
|---|---|---|
| ASVspoof bonafide (English, real) | 0.9902 | 99.1% |
| ASVspoof spoof (English, fake) | 0.0114 | 0.8% |
| **MUCS bonafide (Hinglish, real)** | **0.1291** | **12.1%** |
| XTTS spoof (Hinglish, fake) | 0.0648 | 5.2% |

The detector does not merely fail to catch the fakes — it **calls 87.9% of
genuine human Hinglish speech "spoof"**, with a median confidence of
P(bonafide) = 0.0014. Both classes collapse toward the spoof end, which is why
AUC sits at 0.533.

For the deployment this project targets, that is the more damaging error: a real
customer on a real call is flagged as a deepfake nearly nine times in ten.

## What this does NOT establish

**Language shift and recording-domain shift are confounded here.** ASVspoof is
studio read speech; MUCS is NPTEL lecture audio — different microphones, rooms
and speaking style. The 51× gap therefore measures *English studio → Hinglish
lecture*, not code-mixing alone, and must not be reported as a pure
code-mixing effect.

Two ways to separate them, both available:

- **Channel-matched columns.** Pushing both corpora through the same 8 kHz
  telephony chain compresses recording differences, so a gap that survives is
  more attributable to language.
- **A monolingual Hindi column** and **HiACC** (a different Indian recording
  domain) bracket the language axis against the domain axis.

## Second caveat: the spoof side is weak

The XTTS corpus scored 1.5/5 sounds-human and 1/5 code-switch-natural from the
team, with measured pitch range 25–29 Hz against 41–42 Hz for real speech
(**P-019**). These are *easy* fakes.

That cuts in an interesting direction. The detector still fails on them — but it
fails by rejecting the real audio, not by being fooled by good fakes. A stronger
attack family (RVC, still blocked) would test the other half of the claim.

## Method

- Both classes drawn from the **same 25 train-pool speakers**, balanced 2,217 /
  2,217, so speaker identity cannot separate the classes. An eval-pool bonafide
  set paired with train-pool spoofs would have made the gap partly a speaker-ID
  artefact.
- No leakage: Stage-1 trained on `asvspoof2019_la` only (P-001), so every MUCS
  and XTTS clip here is unseen. The speaker-pool firewall governs Stage-3
  adaptation, not Stage-1 evaluation.
- Spoof clips restricted to the 3,998 that passed the generation quality screen.

Reproduce: `python -m src.training.evaluate --checkpoint checkpoints/baseline/best.pt
--manifest data/manifests/gap_codemix_clean.csv --device cuda`
