# Streamlit demo

A standalone page for showing the detector working on a file, a microphone
recording, or a prepared clip. It is a stand-in for the live-call system while that
is being worked on, and it **imports nothing from `live_call/`**.

```bash
pip install -r requirements.txt -r app/requirements-app.txt
streamlit run app/streamlit_app.py
```

It opens on <http://localhost:8501>. CPU is fine; a GPU is used automatically if
one is visible.

## What it scores with

The same functions the published numbers were measured with — no second
implementation of the pipeline:

| Step | Comes from |
|---|---|
| Resample to 16 kHz mono, level to −23 dBFS | `src.utils.audio_utils` |
| Simulated phone line (8 kHz, G.711 or AMR-NB, noise at SNR, back to 16 kHz) | `src.data.channel_sim` |
| 4 s windows every 2 s, −50 dBFS silence gate | `src.inference.streaming` |
| Checkpoint loading and `window -> P(real)` | `src.inference.predict` |
| Decision threshold | `configs/threshold.yaml` |

The clip verdict is the **lowest-scoring window**, matching `src.inference.predict`:
one confidently synthetic stretch is not averaged away by a calm one.

## Checkpoints

The sidebar offers whichever of these exist on the machine, best first:

| Model | Path |
|---|---|
| S2 LoRA, normalised + RVC, channel-matched (**deployed**) | `$DFD_CHECKPOINT` from `.env` |
| S2 LoRA, channel-matched | `checkpoints/lora_codemix_channel/best.pt` |
| S2 LoRA, clean-trained | `checkpoints/lora_codemix/best.pt` |
| S1 baseline, English-only | `checkpoints/baseline/best.pt` |

The in-repository checkpoints are Git LFS pointers: run `git lfs pull` after
cloning. The deployed adapter is 365 MB and lives outside the repository — point
`DFD_CHECKPOINT` at it. If none is present the app says so and stops, rather than
failing halfway through a demonstration.

## Demo clips

Copy a few clips into `app/demo_clips/` and they appear in the third tab with their
ground truth. See the README in that folder. Audio there is not committed.

## Two things worth showing an examiner

**The channel is the whole story.** Pick *S2 LoRA, clean-trained*, score a genuine
clip (green), then tick **Simulate phone line** and score it again. It degrades
badly. Switch to the channel-matched adapter and it holds. That contrast is the
central finding of the project, and it takes fifteen seconds to demonstrate.

**The unseen tool is missed.** The Tortoise clip usually passes as real. That is
reported openly at 30.99% EER over a telephone line, and showing it is more
convincing than showing only successes.

## Tests

```bash
pytest tests/test_app_helpers.py
```

They cover the pure logic in `app/app_core.py` — model selection, preparation, the
channel, windowing, the worst-window rule, the silence gate and the clip listing —
without starting a Streamlit runtime, so they run in the light CI environment.
