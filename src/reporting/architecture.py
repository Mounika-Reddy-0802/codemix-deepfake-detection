"""Draw the live-call system architecture for the report.

Kept as code rather than a drawing tool so the figure can be regenerated when the
system changes, and so what it claims can be checked against the modules it names.
``tests/test_architecture.py`` asserts every module referenced here exists.

    python -m src.reporting.architecture --out paper/figures
"""

from __future__ import annotations

from pathlib import Path

OUT_DIR = "paper/figures"

#: (label, module) for every box, grouped by layer. The module path is checked by a
#: test, so the diagram cannot claim a component the repository does not have.
CARRIERS: tuple[tuple[str, str], ...] = (
    ("Browser call\n(WebRTC)", "live_call/webrtc_harness/rtc_server.py"),
    ("Phone call\n(Twilio Media Streams)", "live_call/media_handler.py"),
)
CORE: tuple[tuple[str, str], ...] = (
    ("Shared detector\nwav2vec 2.0 + LoRA", "live_call/detector.py"),
    ("Streaming scorer\n4 s window / 2 s hop", "src/inference/streaming.py"),
    ("Verdict ladder\n2 low → caution, 4 → warning", "live_call/verdict_engine.py"),
)
DELIVERY: tuple[tuple[str, str], ...] = (
    ("Exchange\ncall state, routing", "live_call/phone.py"),
    ("Alerts\ntone, in-call warning, SMS", "live_call/alerts.py"),
)

INK = "#111827"
MUTED = "#6b7280"
BLUE = "#2563eb"
GREEN = "#059669"
RED = "#dc2626"


def _box(ax, x, y, w, h, label, colour, fontsize=9):
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.2,
            edgecolor=colour,
            facecolor=colour + "14",
        )
    )
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fontsize, color=INK)


def _arrow(ax, start, end, colour=MUTED, style="-|>", label="", dashed=False):
    from matplotlib.patches import FancyArrowPatch

    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=11,
            linewidth=1.1,
            color=colour,
            linestyle="--" if dashed else "-",
            shrinkA=2,
            shrinkB=2,
        )
    )
    if label:
        ax.text(
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2 + 0.018,
            label,
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=colour,
        )


def draw(out_dir: str = OUT_DIR) -> list[str]:
    """Write the architecture figure as PDF (for LaTeX) and PNG (for slides)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # --- carriers -------------------------------------------------------
    ax.text(0.02, 0.965, "CARRIERS", fontsize=8, color=MUTED, weight="bold")
    _box(ax, 0.02, 0.80, 0.21, 0.11, "Browser call\nWebRTC / Opus", BLUE)
    _box(ax, 0.02, 0.63, 0.21, 0.11, "Phone call\nTwilio Media Streams", BLUE)
    ax.text(
        0.125,
        0.60,
        "one contract: 16 kHz mono PCM\n(Twilio leg not yet run on live numbers)",
        ha="center",
        va="top",
        fontsize=7,
        color=MUTED,
        style="italic",
    )

    # --- detection core -------------------------------------------------
    ax.text(
        0.33, 0.965, "DETECTION CORE  (carrier-agnostic)", fontsize=8, color=MUTED, weight="bold"
    )
    _box(
        ax,
        0.33,
        0.79,
        0.24,
        0.12,
        "Streaming scorer\n4 s window every 2 s\nsilence gate, −23 dBFS",
        GREEN,
    )
    _box(
        ax,
        0.33,
        0.60,
        0.24,
        0.12,
        "Shared detector\nwav2vec 2.0 + LoRA\nloaded once at startup",
        GREEN,
    )
    _box(ax, 0.33, 0.41, 0.24, 0.12, "Verdict ladder\n2 low → caution\n4 low → warning", GREEN)

    _arrow(ax, (0.23, 0.855), (0.33, 0.855))
    _arrow(ax, (0.23, 0.685), (0.325, 0.82))
    _arrow(ax, (0.45, 0.79), (0.45, 0.72))
    ax.text(0.462, 0.752, "window", fontsize=7.5, color=MUTED, va="center")
    _arrow(ax, (0.45, 0.60), (0.45, 0.53))
    ax.text(0.462, 0.562, "P(real)", fontsize=7.5, color=MUTED, va="center")

    # --- delivery -------------------------------------------------------
    ax.text(0.68, 0.965, "DELIVERY", fontsize=8, color=MUTED, weight="bold")
    _box(ax, 0.68, 0.79, 0.24, 0.11, "Exchange\ncall state and routing", RED)
    _box(ax, 0.68, 0.58, 0.24, 0.11, "Receiver handset\ntone, banner, SMS", RED)
    _box(ax, 0.68, 0.37, 0.24, 0.11, "Caller handset\ncall state only", MUTED)

    _arrow(ax, (0.57, 0.47), (0.68, 0.61), colour=RED)
    ax.text(0.615, 0.565, "verdict", fontsize=7.5, color=RED, ha="center")
    _arrow(ax, (0.72, 0.79), (0.72, 0.69), colour=MUTED)
    _arrow(ax, (0.955, 0.79), (0.955, 0.48), colour=MUTED)
    ax.text(0.955, 0.455, "call\nstate", fontsize=7, color=MUTED, ha="center", va="top")

    ax.text(
        0.80,
        0.335,
        "no verdict ever reaches the caller:\na detected fraudster must not learn it",
        ha="center",
        va="top",
        fontsize=7.5,
        color=RED,
        style="italic",
    )

    # --- footer ---------------------------------------------------------
    ax.plot([0.02, 0.98], [0.255, 0.255], color="#e5e7eb", linewidth=1)
    ax.text(
        0.02,
        0.20,
        "Latency budget on speech: first caution ≈ 6 s, warning ≈ 10 s.",
        fontsize=8.5,
        color=INK,
    )
    ax.text(
        0.02,
        0.145,
        "The detector is the evaluation protocol, unchanged: the same encoder, windowing, "
        "silence gate\nand operating point that produced the reported error rates.",
        fontsize=8,
        color=MUTED,
        va="top",
    )
    ax.text(
        0.02,
        0.035,
        "Scoring runs off the audio thread, so a slow forward pass cannot stall the call.",
        fontsize=8,
        color=MUTED,
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix, dpi in ((".pdf", None), (".png", 200)):
        path = out / f"system_architecture{suffix}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        written.append(str(path))
    plt.close(fig)
    return written


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Draw the live-call architecture")
    parser.add_argument("--out", default=OUT_DIR)
    args = parser.parse_args()
    for path in draw(args.out):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
