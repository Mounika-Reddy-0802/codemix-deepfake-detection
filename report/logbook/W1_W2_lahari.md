# Log book — Lahari, Weeks 1–2

## Week 1

**W1-T1 — corpus download + verification scripts.** Wrote
`scripts/01_download_data.sh` covering the three corpora we actually keep on disk:
ASVspoof 2019 LA (the only Stage-1 training corpus), MUCS 2021 Hindi-English from
OpenSLR 104 (Bengali-English deliberately skipped), and HiACC from Zenodo record
15551669. The script is dry-run by default — multi-GB pulls only start with
`--run`, so nobody triggers 30 GB by accident. HiACC's file URL and md5 are read
from the Zenodo REST API at runtime rather than hard-coded, and child folders are
moved to `_EXCLUDED_children/` immediately on extraction. ASVspoof 2021 DF was
skipped on storage grounds (34.5 GB).

**W1-T2 — eval-only streaming loaders + licence register.** IndicSynth,
IndicTTS-Deepfake and IndicVoices are Hugging Face streaming only, never bulk
downloaded (`src/data/download.py`). Wrote `docs/licences.md` recording each
corpus's licence and what it permits.

**W1-T7 (ALL, my third) — related-work and ethics templates.**

## Week 2

**W2-T1 — preprocessing + channel simulation.** `src/data/preprocess.py`:
resample to 16 kHz mono, silero VAD trim, RMS loudness normalisation, 2–10 s
segmentation. `src/data/channel_sim.py`: the telephony chain that the whole
channel-matched protocol rests on — 16 kHz → 8 kHz → G.711 mu-law (or AMR-NB via
ffmpeg when available) → additive noise at a controlled SNR → back to 16 kHz. The
lossy parts are pure numpy so they are unit-tested without the audio stack;
`add_noise_at_snr` hits the requested SNR exactly rather than approximately.

## Reflection

The download script's child-quarantine regex bothered me at the time — it assumes
HiACC separates children by folder name, which I could not verify without the
corpus. I flagged it in `Works updates.md` rather than trusting it. That turned out
to be the right call: in Week 3 I found the preprocessing walk would have ingested
the quarantined folder anyway, so the regex was not the only thing standing
between us and child audio in the pipeline — nothing was.

**Blocked on:** downloads not yet run; ethics sign-off not obtained.
