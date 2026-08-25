# Channel-matched v1 — the gap closure does not survive a phone line for free

**Companion to [gap_closure_v1.md](gap_closure_v1.md), and a correction to how that
result should be read.** The 1.34% EER reported there is measured on clean 16 kHz
NPTEL lecture audio. This project targets **live call detection**. Those are not the
same condition, and it turns out they are not close.

`gap_matrix_v1.md` asked for this column for a different reason — to separate
language shift from recording-domain shift by pushing every corpus through the same
narrowband chain. It does that too. But the finding that matters is the deployment
one.

## Protocol

Every clip in `data/manifests/codemix_eval.csv` rendered through the chain already
validated in W3-T2: **16 kHz → 8 kHz → G.711 µ-law → white noise @ 20 dB SNR →
16 kHz** (`src/data/channel_sim.simulate_channel`, applied per manifest by the new
`src/data/channel_bundle`).

Re-verified on this render, against the W3-T2 reference numbers:

| Check | W3-T2 reference | This render |
|---|---|---|
| Energy above 4 kHz | 0.6% → 0.0001% | **0.185% → 0.0001%** |
| Measured SNR (target 20 dB) | 17.6 dB | **18.8 dB** |

Noise is seeded **per clip** from a hash of the clip name. A single shared noise
draw would add the same waveform to all 3,966 clips, which a detector can learn
instead of the speech — the same class of shortcut the duration and speaker-firewall
checks exist to rule out.

## Result

Same 3,966 clips and the same 15 held-out speakers throughout; only the audio
condition and the adapter's training condition change.

| Adapter trained on | Clean eval | Channel eval |
|---|---|---|
| *(none — Stage-1 baseline)* | 53.71% (AUC 0.459) | 54.92% (AUC 0.414) |
| **Clean audio** | **1.34%** (AUC 1.000) | **38.58%** (AUC 0.666) |
| **Channel-matched audio** | 13.92% (AUC 0.924) | **3.89%** (AUC 0.993) |

Two things follow, and the second rescues the first.

**1. The clean-trained adapter mostly collapses over a phone line.** 1.34% → 38.58%
EER. Deployed as-is on the target application it would be close to useless, and
nothing in `gap_closure_v1.md` predicts that, because nothing there is measured
under a codec.

**2. Training channel-matched recovers it.** 54.92% → 3.89% EER, a 14× reduction,
with AUC back to 0.993. The gap *is* closable under telephony — the adapter simply
has to be trained on the condition it will be deployed in. That costs clean
performance (1.34% → 13.92%), which is the expected trade and an argument for
matching the training condition to the deployment rather than for one universal
adapter.

## Why it collapses: the spoof cue lives above 4 kHz

The pooled EER hides the mechanism, exactly as it did in `gap_matrix_v1.md`. The
per-class breakdown does not:

| Model / condition | bonafide called "bonafide" | spoof called "bonafide" |
|---|---|---|
| Clean-trained, clean eval | 99.9% | 2.8% |
| **Clean-trained, channel eval** | 100.0% | **98.9%** |
| Channel-trained, channel eval | 100.0% | 37.3% |

Under the codec the clean-trained adapter calls **98.9% of spoofs genuine**. It has
not become confused — it has become a model that says "real" to everything. Its real
side is untouched; only its ability to recognise a fake is gone.

That points at one thing. The adapter learned to detect XTTS-v2 by a **high-frequency
vocoder signature**, and the 8 kHz round-trip destroys precisely that band (0.185% →
0.0001% of energy above 4 kHz). Remove the artefact and the fakes look real to it.

This sharpens `gap_matrix_v1.md`'s own "the spoof side is weak" caveat into something
measurable: the XTTS corpus is not merely *easy*, its detectability is concentrated in
the part of the spectrum a telephone network throws away. The channel-trained adapter
still finds *something* narrowband to use — 37.3% of spoofs slip through versus
98.9% — but its dev EER (6.51%) never approaches the clean run's 1.01%, so the
narrowband cue is genuinely weaker rather than merely unlearned.

## What this does NOT establish

**One channel condition.** G.711 at 20 dB SNR only. `configs/data/channel_sim.yaml`
defines a sweep — `amr_nb` and SNR 5/10/15 — none of which was run. 20 dB is a good
line; a real call is often worse, and the trend across SNR is unmeasured.

**Still one attack family, still one seed** — the caveats in `gap_closure_v1.md`
carry over unchanged. The channel-trained row has *not* been repeated over seeds.

**Additive white noise is not call noise.** Real calls carry babble, music on hold,
packet loss and handset variation. The chain models bandwidth and codec faithfully
and noise only crudely.

**This does not separate language from domain.** The column was built partly to do
that, but with the ASVspoof side unavailable on this machine (see
`gap_closure_v1.md`), only the code-mixed half exists channel-matched. The
confound-separation the gap matrix asked for still needs the English column.

## Reproduce

```powershell
$env:DATA_ROOT = "D:\dfdata"

# render the eval set (and the adaptation pool, for the channel-matched adapter)
.venv\Scripts\python.exe -m src.data.channel_bundle `
  --manifest data\manifests\codemix_eval.csv `
  --out-dir "D:\dfdata\channel_bundle_20db" `
  --manifest-out data\manifests\codemix_eval_channel20.csv `
  --data-root "D:\dfdata\lora_bundle" --codec g711 --snr-db 20

# score any checkpoint on it
.venv\Scripts\python.exe -m src.training.evaluate `
  --checkpoint checkpoints\lora_codemix\best.pt `
  --manifest data\manifests\codemix_eval_channel20.csv `
  --device cuda --batch-size 32 --num-workers 4 `
  --data-root "D:\dfdata\channel_bundle_20db" `
  --out experiments\lora_codemix_channel20.json

# the channel-matched adapter
.venv\Scripts\python.exe -m src.training.train `
  --config configs\train_lora_codemix_channel.yaml `
  --device cuda --data-root "D:\dfdata\channel_adapt_20db"
```
