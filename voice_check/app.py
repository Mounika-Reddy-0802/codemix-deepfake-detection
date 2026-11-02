"""Voice check: record or upload any audio, get one answer, real or cloned.

A single page, no phone call. The input can be speech recorded on the spot, a phone-call
recording, a voice note, any audio file, or audio played aloud from another device into
the microphone. Every input is judged by ``src.inference.any_audio.judge`` with the
detector named in ``configs/voice_check.yaml``: the same pipeline the accuracy numbers
were measured with.

The page also collects labelled examples for measuring accuracy on real use. A recording
is kept only when the person saving it says what it really was **and** confirms the
speaker agreed to it being kept. Kept recordings go to ``data/field_validation/`` (ignored
by git) with a ``log.csv`` that ``src.inference.any_audio_eval score`` reads as a manifest.

    python -m voice_check.app              # then open http://127.0.0.1:7860
"""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
import threading
from datetime import datetime
from pathlib import Path

from src.inference import any_audio as aa

FIELD_DIR = Path("data/field_validation")
FIELD_COLUMNS = (
    "filepath",
    "label",
    "tool",
    "set",
    "split",
    "condition",
    "saved_at",
    "verdict_at_save",
    "p_genuine_at_save",
    "front_end",
    "checkpoint",
)
#: How the audio reached the microphone or the file; kept so accuracy can be split by it.
CONDITIONS = {
    "Spoken live into this microphone": "mic_live",
    "Played aloud from another device into the microphone": "mic_replay",
    "Phone-call recording": "phone_recording",
    "Audio file from elsewhere": "file",
}
TRUTHS = {"Real human voice": "bonafide", "Cloned or synthetic voice": "spoof"}

COLOURS = {aa.REAL: "#059669", aa.CLONED: "#dc2626", aa.NO_SPEECH: "#6b7280"}
HEADLINES = {
    aa.REAL: "Real voice",
    aa.CLONED: "Cloned voice",
    aa.NO_SPEECH: "Not enough speech to judge",
}


# ------------------------------------------------------------------ detector (lazy)
class Detector:
    """Loads the configured checkpoint on first use, once, behind a lock."""

    def __init__(self, config_file: str = aa.CONFIG_FILE) -> None:
        self.config = aa.DetectorConfig.from_file(config_file)
        self.threshold = self.config.threshold()
        self._score_fn = None
        self._lock = threading.Lock()

    def judge(self, path: str) -> aa.VoiceVerdict:
        with self._lock:
            if self._score_fn is None:
                self._score_fn = aa.load_detector(self.config)
        return aa.judge(
            aa.load_audio(path), self._score_fn, self.threshold, self.config.front_end, path
        )


# ------------------------------------------------------------------ presentation
def verdict_html(result: aa.VoiceVerdict) -> str:
    """The answer as a coloured card with the numbers behind it."""
    colour = COLOURS[result.verdict]
    head = HEADLINES[result.verdict]
    if result.p_genuine is None:
        detail = (
            f"Only {result.speech_seconds:.1f} s of speech was found in "
            f"{result.seconds:.1f} s of audio. Record at least {aa.MIN_SPEECH_SECONDS:.0f} s "
            "of someone speaking."
        )
    else:
        detail = (
            f"P(genuine) {result.p_genuine:.3f} against a threshold of {result.threshold:.3f}. "
            f"{result.windows_agreeing} of {len(result.scored)} four-second windows agree. "
            f"{result.speech_seconds:.1f} s of speech judged."
        )
    return (
        f'<div style="border:2px solid {colour};border-radius:12px;padding:18px 20px;">'
        f'<div style="font-size:28px;font-weight:700;color:{colour};">{head}</div>'
        f'<div style="margin-top:6px;font-size:15px;">{detail}</div></div>'
    )


def window_plot(result: aa.VoiceVerdict):
    """P(genuine) of every window over time, with the threshold drawn across."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 2.6))
    xs = [(w.start_s + w.end_s) / 2 for w in result.windows if w.p_genuine is not None]
    ys = [w.p_genuine for w in result.windows if w.p_genuine is not None]
    if xs:
        ax.plot(xs, ys, marker="o", color="#2563eb", linewidth=1.5)
    ax.axhline(result.threshold, color="#dc2626", linestyle="--", linewidth=1)
    ax.text(
        0.01,
        result.threshold + 0.03,
        "threshold",
        transform=ax.get_yaxis_transform(),
        color="#dc2626",
        fontsize=8,
    )
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("time in the recording (s), centre of each 4 s window")
    ax.set_ylabel("P(genuine)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ field validation
def save_example(
    audio_path: str,
    truth: str,
    condition: str,
    tool: str,
    consent: bool,
    result: aa.VoiceVerdict | None,
    checkpoint: str,
    folder: Path = FIELD_DIR,
) -> str:
    """Keep one labelled recording for the accuracy check; returns a status line."""
    if not audio_path:
        return "Nothing to save: record or upload audio first."
    if not consent:
        return "Not saved: tick the box confirming the speaker agreed to the recording being kept."
    if truth not in TRUTHS or condition not in CONDITIONS:
        return "Not saved: say what the audio really was and how it was captured."
    clips = folder / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    data = Path(audio_path).read_bytes()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = clips / f"{stamp}_{hashlib.sha256(data).hexdigest()[:10]}{Path(audio_path).suffix}"
    shutil.copyfile(audio_path, target)
    row = {
        "filepath": str(target.resolve()).replace("\\", "/"),
        "label": TRUTHS[truth],
        "tool": (tool.strip() or "unknown") if TRUTHS[truth] == "spoof" else "none",
        "set": "field",
        "split": "test",
        "condition": CONDITIONS[condition],
        "saved_at": stamp,
        "verdict_at_save": result.verdict if result else "",
        "p_genuine_at_save": ""
        if result is None or result.p_genuine is None
        else round(result.p_genuine, 4),
        "front_end": result.front_end if result else "",
        "checkpoint": checkpoint,
    }
    log = folder / "log.csv"
    new = not log.is_file()
    with log.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELD_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow(row)
    total = sum(1 for _ in log.open(encoding="utf-8")) - 1
    return f"Saved ({total} labelled recordings so far)."


# ------------------------------------------------------------------ page
def build_ui(detector: Detector):
    import gradio as gr

    with gr.Blocks(title="Voice check") as page:
        gr.Markdown(
            "# Voice check\nRecord someone speaking, or upload any audio file, and the detector "
            "says whether the voice is **real** or **cloned**. Speak or play at least "
            f"{aa.MIN_SPEECH_SECONDS:.0f} seconds; 10 seconds or more gives a steadier answer."
        )
        audio = gr.Audio(
            sources=["microphone", "upload"], type="filepath", label="Record or upload a voice"
        )
        run = gr.Button("Check this voice", variant="primary")
        card = gr.HTML()
        plot = gr.Plot(label="Score of every 4-second window")
        state = gr.State(None)

        with gr.Accordion(
            "Help measure accuracy: save this recording with its true label", open=False
        ):
            truth = gr.Radio(list(TRUTHS), label="What was it really?")
            condition = gr.Dropdown(list(CONDITIONS), label="How was it captured?")
            tool = gr.Textbox(label="If cloned: which tool made it (optional)")
            consent = gr.Checkbox(
                label="The speaker agreed to this recording being kept for testing"
            )
            save = gr.Button("Save labelled recording")
            status = gr.Markdown()

        gr.Markdown(
            f"Detector: `{Path(detector.config.checkpoint).name}` · threshold "
            f"{detector.threshold:.3f} (chosen on development data only) · front end "
            f"`{detector.config.front_end}`."
        )

        def check(path):
            if not path:
                return "<p>Record or upload audio first.</p>", None, None
            try:
                result = detector.judge(path)
            except Exception as exc:
                return f"<p style='color:#dc2626'>Could not judge this audio: {exc}</p>", None, None
            return verdict_html(result), window_plot(result), result

        run.click(check, inputs=audio, outputs=[card, plot, state])
        save.click(
            lambda p, t, c, tl, ok, r: save_example(p, t, c, tl, ok, r, detector.config.checkpoint),
            inputs=[audio, truth, condition, tool, consent, state],
            outputs=status,
        )
    return page


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Voice check: real or cloned?")
    parser.add_argument("--config", default=aa.CONFIG_FILE)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="public Gradio link")
    args = parser.parse_args()
    # Gradio keeps uploads in a temp folder; keep it next to the project, not on C:.
    os.environ.setdefault("GRADIO_TEMP_DIR", str(Path("data/field_validation/.tmp").resolve()))
    detector = Detector(args.config)
    build_ui(detector).launch(server_name="127.0.0.1", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
