# Demo clips for the Streamlit app

Drop a few short clips here (5-30 seconds each) and they appear in the app's
**Demo clips** tab with their ground truth shown, so a demonstration never depends
on finding a file in a dialog.

The app recognises these names and labels them automatically:

| Filename | Shown as | Ground truth |
|---|---|---|
| `genuine.wav` | Genuine caller | real |
| `rvc_genuine.wav` | Genuine caller (RVC source speaker) | real |
| `xtts_clone.wav` | XTTS-v2 cloned voice | cloned |
| `rvc_conversion.wav` | RVC voice conversion | cloned |
| `tortoise_clone.wav` | Tortoise clone (unseen tool) | cloned |

Any other audio file is listed too, with its ground truth shown as unknown.

**Where to get them.** The prepared demo clips built from held-out audio already
exist outside the repository, at the path in `DEMO_CLIPS_DIR` (see `.env`). Copy
them across:

```bash
cp "$DEMO_CLIPS_DIR"/*.wav app/demo_clips/
```

**These files are not committed.** They are real speech from the evaluation pool,
they are large, and the corpora have their own licence terms - see
`docs/licences.md`. Only this README is tracked.

**Suggested order for a demonstration.** Score a genuine clip first so the panel is
green, then the XTTS clone for the red verdict, then the Tortoise clip, which is an
unseen tool and is usually missed. Showing the miss is the point: it is the
limitation the project reports at 30.99% EER.
