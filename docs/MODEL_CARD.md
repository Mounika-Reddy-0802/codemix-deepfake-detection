# Model card: code-mixed telephone deepfake detector (final recipe)

> **Status: draft, not published.** Releasing the weights needs the team's and the
> mentor's approval (see "Release decision" below). Until then the weights stay private, as
> `docs/release.md` states.

## Model

- **What it is.** A binary detector: given 4 s of 16 kHz speech, it outputs P(genuine).
  Architecture: wav2vec 2.0 base encoder (frozen) with LoRA adapters (rank 8, alpha 16, on
  the attention projections of all 12 layers; 1.13% of the parameters trained) and a small
  classification head. Checkpoint: `final_s1234_best.pt` (one of five training seeds).
- **Recipe.** Starts from the Stage-1 detector trained on ASVspoof 2019 LA (English) only.
  Adapted on `data/manifests/codemix_adapt_train_final.csv`: 248 genuine Hindi-English
  speakers (MUCS 2021 read lectures) and XTTS-v2 clones and RVC conversions of training
  speakers, every clip both over a simulated G.711 telephone line (20 dB) and through the
  real AMR-NB mobile codec (16,036 clips). Config: `configs/train_lora_final.yaml`.
- **Operating point for calls.** Threshold 0.687 on smoothed window scores, warning after 3
  fake-leaning windows and "likely fake" after 5 (`configs/threshold_final.yaml`), chosen on
  the dev set only for at most 5% warnings on genuine 60 s calls.

## Intended use

- Research on deepfake detection for Hindi-English code-mixed speech over telephone
  channels, and reproduction of the paper's numbers.
- A warning shown to the **callee only**, on Hindi-English calls of read or prepared speech,
  as in the project's live system (`live_call/`).

## Out of scope

- **Spontaneous conversational speech.** Every genuine 60 s spontaneous call (HiACC) raises
  a warning (`docs/results/live_final_recipe_v1.md`). Do not deploy it in front of
  spontaneous callers.
- **English calls.** It lets most English clones through (13.44% EER on clean English;
  catches 26% of English fakes at 0.5) (`docs/results/english_final_v1.md`).
- Any decision about a person (blocking, reporting, evidence) without a human in the loop.
- Commercial use (see licence).

## Evaluation (five training seeds unless noted; EER %)

| Test set | Simulated G.711 | Real AMR-NB |
|---|---:|---:|
| New speakers, XTTS-v2 (seen tool) | 0.94 | 0.69 |
| New speakers, Tortoise (unseen) | 8.81 | 6.04 |
| CM04 Tortoise (unseen) | 17.73 | 6.90 |
| IndicF5 (unseen) | 21.68 | 21.32 |
| Chatterbox (unseen, one seed) | 31.98 | - |
| RVC test half | 8.53 | - |
| Genuine read speech called fake | 0.51 | - |

Every seed beats a shortcut floor of eight low-level signal statistics on every set
(`docs/results/floor_final_v1.md`); three of those floors are above chance and so weak bars.
On 60 s calls through the live path it warns no genuine read-speech call and catches every
XTTS, Tortoise and RVC call, half of the IndicF5 calls and one in six Chatterbox calls.

## Limitations

- Trained and tested on read lecture speech; spontaneous speech is out of scope (above).
- The G.711 line is simulated; AMR-NB is a real codec run offline. On a real mobile network
  the evidence is a pilot of 30 calls in the voices of the three authors
  (`docs/results/real_calls_v1.md`): it left nine of twelve genuine live calls unwarned (over G.711
  the three warned are one speaker's), took five of six XTTS-v2 clone calls to "likely cloned" and
  none of six Chatterbox clone calls (one was cautioned). Its false alarms on genuine live calls come early in a call, starting with
  the quiet before the caller speaks, and some continue into the caller's first seconds of
  speech.
- IndicF5 and Chatterbox remain hard (21-32% EER for this adapter), and Chatterbox clones
  passed on real calls.
- All attacks were made by the project with open tools; commercial cloning services were
  not tested.
- A larger encoder (SSL-AASIST, XLS-R 300M) trained on the same data is better on eight of
  eleven sets (`docs/results/ssl_vs_adapter_v1.md`) and keeps up with a live call even on two
  CPU threads (1.14 s per 2 s window hop; 69 ms on a T4 GPU). On spontaneous speech one of its
  trainings calls 5.8% of genuine clips fake and another 46.7%, so it is a lead, not a fix
  (`docs/results/encoder_spontaneous_v1.md`). It is the candidate for the next release.

## Training data and licences

ASVspoof 2019 LA (research licence), MUCS 2021 Hindi-English (CC BY-SA 4.0), and clones made
by the project with XTTS-v2 (Coqui Public Model Licence, non-commercial) and RVC. HiACC
(CC BY 4.0) and every other evaluation corpus never enter training, and HiACC child audio is
never used (`docs/licences.md`, enforced by `tests/test_splits.py`).

## Release decision (to be made by the team and the mentor)

1. Whether to publish the weights at all.
2. Under which account (team or one member's) on Hugging Face.
3. The licence. Proposed: **CC BY-NC-SA 4.0**, because the weights are derived from
   MUCS (share-alike) and from XTTS-v2 output (non-commercial).
4. No audio is released in any case: the clones are cloned speech of identifiable people.

## How to run

```bash
set DFD_CHECKPOINT=<data>/checkpoints/final_s1234_best.pt
set DFD_THRESHOLD_FILE=configs/threshold_final.yaml
uvicorn live_call.server:app --host 0.0.0.0 --port 8000
```

Offline scoring: `python -m src.training.evaluate --checkpoint <ckpt> --manifest <csv>
--encoder wav2vec2-base --data-root <root>`.
