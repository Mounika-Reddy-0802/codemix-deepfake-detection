"""Streamlit demo: score a clip with the deployed code-mixed deepfake detector.

    streamlit run app/streamlit_app.py

A standalone stand-in for the live-call system while that is being worked on. It
imports nothing from ``live_call/`` and changes nothing there. All the real work
happens in :mod:`app.app_core`, which calls the same preprocessing, channel
simulation and scoring the published numbers were measured with.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import app_core as core  # noqa: E402

st.set_page_config(page_title="Code-mixed voice deepfake detector", page_icon="🎙", layout="wide")


# ----------------------------------------------------------------- resources


@st.cache_resource(show_spinner=False)
def get_scorer(checkpoint: str, device: str | None):
    """Load a checkpoint once per session. Heavy imports stay inside."""
    from src.inference.predict import load_detector, make_score_fn

    model, resolved = load_detector(checkpoint, device=device)
    return make_score_fn(model, resolved), resolved


@st.cache_data(show_spinner=False)
def cached_threshold() -> tuple[float, str]:
    return core.load_threshold()


def detected_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 - torch missing is a normal, reportable state
        return "cpu"


# -------------------------------------------------------------------- sidebar

st.sidebar.title("Detector")

env_checkpoint = os.environ.get("DFD_CHECKPOINT") or None
choices = [c for c in core.model_choices(env_checkpoint)]
available = [c for c in choices if c.available]

if not available:
    st.sidebar.error("No checkpoint found on this machine.")
    st.title("Code-mixed voice deepfake detector")
    st.error(core.missing_checkpoint_message(choices[0]))
    st.stop()

labels = {c.label: c for c in available}
picked = st.sidebar.selectbox("Model", list(labels), index=0)
choice = labels[picked]
st.sidebar.caption(choice.blurb)

if len(available) < len(choices):
    missing = ", ".join(c.label for c in choices if not c.available)
    st.sidebar.caption(f"Not on this machine: {missing}")

st.sidebar.divider()
phone_line = st.sidebar.checkbox(
    "Simulate phone line",
    value=False,
    help="8 kHz downsample, codec round-trip, noise at the chosen SNR, back to 16 kHz "
    "-- the same chain the channel-matched results were measured on.",
)
codec = st.sidebar.selectbox("Codec", ["g711", "amr-nb"], disabled=not phone_line)
snr_db = st.sidebar.slider("SNR (dB)", 5.0, 30.0, 20.0, 1.0, disabled=not phone_line)

st.sidebar.divider()
default_threshold, threshold_source = cached_threshold()
threshold = st.sidebar.slider(
    "Decision threshold on P(real)", 0.05, 0.95, float(round(default_threshold, 3)), 0.01
)
st.sidebar.caption(f"Default {default_threshold:.3f} from `{threshold_source}`")

device = detected_device()
st.sidebar.caption(f"Device: **{device}**")


# ------------------------------------------------------------------ plotting


def waveform_figure(audio: np.ndarray):
    import matplotlib.pyplot as plt

    seconds = np.arange(audio.size) / core.SAMPLE_RATE
    fig, ax = plt.subplots(figsize=(9, 1.8))
    ax.plot(seconds, audio, linewidth=0.5, color="#4f8cff")
    ax.set_xlabel("seconds", fontsize=8)
    ax.set_yticks([])
    ax.set_xlim(0, max(seconds[-1] if seconds.size else 1, 1))
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


def timeline_figure(result: core.Result):
    import matplotlib.pyplot as plt

    scored = result.scored_windows
    fig, ax = plt.subplots(figsize=(9, 2.4))
    times = [w.end_seconds for w in scored]
    values = [w.score for w in scored]
    colours = ["#e04b4b" if v < result.threshold else "#1fb46a" for v in values]
    ax.bar(times, values, width=1.4, color=colours)
    ax.axhline(result.threshold, color="#e04b4b", linestyle="--", linewidth=1)
    ax.text(0, result.threshold + 0.03, "threshold", color="#e04b4b", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("P(real)", fontsize=8)
    ax.set_xlabel("window end (seconds)", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def show_result(audio: np.ndarray, result: core.Result) -> None:
    headline, colour, detail = core.verdict_card(result)
    box = {"red": st.error, "green": st.success, "gray": st.warning}[colour]
    box(f"### {headline}\n{detail}")

    left, right = st.columns(2)
    left.metric("P(real), lowest window", "—" if result.score is None else f"{result.score:.3f}")
    right.metric("Clip length", f"{result.seconds:.1f} s")

    st.caption("Waveform of exactly what was scored")
    st.pyplot(waveform_figure(audio), clear_figure=True)

    if len(result.scored_windows) > 1:
        st.caption("Every 4 s window, scored every 2 s — the same windowing as the live system")
        st.pyplot(timeline_figure(result), clear_figure=True)
    skipped = len(result.windows) - len(result.scored_windows)
    if skipped:
        st.caption(f"{skipped} window(s) skipped as silence, below the −50 dBFS gate.")


def run(audio: np.ndarray, sr: int) -> None:
    """The one path every tab uses: prepare, optional phone line, score, show."""
    prepared = core.prepare(audio, sr)
    if phone_line:
        prepared = core.apply_phone_line(prepared, codec=codec, snr_db=snr_db)
    with st.spinner("Scoring…"):
        score_fn, _ = get_scorer(str(choice.path), device)
        result = core.score_audio(prepared, score_fn, threshold)
    if phone_line:
        st.caption(f"Carried over a simulated {codec.upper()} line at {snr_db:.0f} dB SNR.")
    st.audio(prepared, sample_rate=core.SAMPLE_RATE)
    show_result(prepared, result)


def read_upload(upload) -> tuple[np.ndarray, int]:
    """Streamlit hands us bytes; soundfile wants a file."""
    suffix = Path(upload.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(upload.getvalue())
        temporary = handle.name
    try:
        return core.load_audio(temporary)
    finally:
        Path(temporary).unlink(missing_ok=True)


# ---------------------------------------------------------------------- page

st.title("Is this voice real, or cloned?")
st.caption(
    "A wav2vec 2.0 detector with a LoRA adapter, trained on Hindi–English code-mixed "
    "speech and the clones made from it. Scored in 4-second windows, exactly as the "
    "live call system does it."
)

upload_tab, mic_tab, clips_tab = st.tabs(["Upload", "Microphone", "Demo clips"])

with upload_tab:
    upload = st.file_uploader("Audio file", type=["wav", "mp3", "flac", "m4a", "ogg"])
    if upload is not None:
        try:
            audio, sr = read_upload(upload)
        except Exception as err:  # noqa: BLE001 - surfaced to the user, not swallowed
            st.error(f"Could not read that file: {err}")
        else:
            run(audio, sr)

with mic_tab:
    st.caption("Record a few seconds and it goes through the identical path.")
    recorded = getattr(st, "audio_input", None)
    if recorded is None:
        st.info("This Streamlit version has no microphone input. Upgrade to 1.40 or newer.")
    else:
        clip = st.audio_input("Record")
        if clip is not None:
            try:
                audio, sr = read_upload(clip)
            except Exception as err:  # noqa: BLE001
                st.error(f"Could not read the recording: {err}")
            else:
                run(audio, sr)

with clips_tab:
    clips = core.demo_clips()
    if not clips:
        st.info(
            "No clips yet. Copy a few into `app/demo_clips/` — see the README in that "
            "folder for which ones and what to name them."
        )
    else:
        st.caption("Prepared clips with known ground truth, so nothing depends on a file dialog.")
        chosen = None
        for clip in clips:
            columns = st.columns([3, 1.4, 1])
            columns[0].write(f"**{clip.title}**")
            badge = {"real": "ground truth: real", "cloned": "ground truth: cloned"}
            columns[1].write(badge.get(clip.truth, "ground truth: unknown"))
            if columns[2].button("Score", key=f"clip-{clip.path.name}"):
                chosen = clip
        # Rendered after the whole list, so the verdict never appears wedged between
        # two clips and push the rest of them off the screen mid-demonstration.
        if chosen is not None:
            st.divider()
            st.subheader(chosen.title)
            audio, sr = core.load_audio(chosen.path)
            run(audio, sr)

with st.expander("About, and what this does not do"):
    st.markdown(
        """
**What it is.** A wav2vec 2.0 encoder with a LoRA adapter, adapted on Hindi–English
code-mixed speech and the clones built from the same speakers. The clip verdict is
the **lowest-scoring 4-second window**, which is deliberately cautious.

**What it was trained against.** XTTS-v2 clones and RVC voice conversions. Those are
the attacks it has seen.

**Tortoise is an unseen tool and is often missed.** We report that rather than hide
it: 30.99% EER over a telephone line. A demo that only shows successes is not
evidence.

**Every result is read against a shortcut floor** — the error rate eight cheap signal
statistics reach on the same clips. A number better than the floor means the model
found something; a number at the floor means it did not.

**The threshold is a clip-level operating point**, calibrated on development calls
and validated on held-out ones. Move the slider and the verdict moves with it; that
is the honest way to see how much the decision depends on where the line is drawn.

**The channel matters more than you would expect.** Turn "simulate phone line" on
with the clean-trained adapter selected and watch a real voice start failing: that
is the single most important finding in this project.
"""
    )
