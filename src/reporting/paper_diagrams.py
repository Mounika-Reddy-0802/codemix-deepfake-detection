"""Draw the system architecture and the algorithm flowchart of the paper.

Two figures, kept as code so every number printed on them is written in one place and can
be checked against the modules it describes:

* ``system_architecture_detailed`` - the pipeline from speech to decision in one row, with
  the detector opened to its main parts and tensor sizes, and a training strip beneath.
* ``algorithm_flowchart`` - the training, evaluation and deployment procedure as a
  single-column flowchart with standard shapes.

Numbers come from ``src/models`` (hidden 768, attention 128, head 256, LoRA r=8, alpha=16
on q/k/v/out_proj), ``src/inference/streaming.py`` (4 s / 2 s, -50 dBFS, EMA 0.5),
``live_call/verdict_engine.py`` and ``configs/threshold*.yaml`` (ladders 2/4 and 3/5), and
``configs/train_*.yaml``. ``tests/test_paper_diagrams.py`` checks both files are produced.

    python -m src.reporting.paper_diagrams --out experiments/figures
"""

from __future__ import annotations

from pathlib import Path

OUT_DIR = "experiments/figures"

INK = "#111827"
MUTED = "#6b7280"
GREY = "#9ca3af"
BLUE = "#2563eb"
GREEN = "#059669"
ORANGE = "#d97706"
RED = "#dc2626"
PURPLE = "#7c3aed"

S1_LR, S1_BATCH, S1_EPOCHS = "1e-5", 4, 3
S2_LR, S2_BATCH, S2_EPOCHS = "1e-4", 8, 5
HIDDEN, ATT_DIM, HEAD_HIDDEN, FRAMES = 768, 128, 256, 199
LORA_R, LORA_ALPHA = 8, 16
LADDER = {"deployed": (2, 4, 0.500), "final": (3, 5, 0.687)}


# ----------------------------------------------------------------------------- helpers
def _font(plt) -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
    plt.rcParams["mathtext.fontset"] = "dejavusans"


def _rect(
    ax,
    x,
    y,
    w,
    h,
    colour,
    title=None,
    body=None,
    *,
    ts=6.0,
    bs=5.4,
    fill=True,
    dashed=False,
    lw=0.9,
):
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.35",
            linewidth=lw,
            edgecolor=colour,
            facecolor=(colour + "12") if fill else "none",
            linestyle="--" if dashed else "-",
        )
    )
    if title and body:
        ax.text(
            x + w / 2,
            y + h - 0.35,
            title,
            ha="center",
            va="top",
            fontsize=ts,
            fontweight="bold",
            color=INK,
        )
        ax.text(
            x + w / 2,
            y + (h - 1.4) / 2,
            body,
            ha="center",
            va="center",
            fontsize=bs,
            color=INK,
            linespacing=1.25,
        )
    elif title:
        ax.text(
            x + w / 2,
            y + h / 2,
            title,
            ha="center",
            va="center",
            fontsize=ts,
            fontweight="bold",
            color=INK,
        )
    elif body:
        ax.text(
            x + w / 2,
            y + h / 2,
            body,
            ha="center",
            va="center",
            fontsize=bs,
            color=INK,
            linespacing=1.25,
        )


def _arrow(
    ax,
    p,
    q,
    colour=MUTED,
    label=None,
    *,
    style="-|>",
    lw=0.9,
    dashed=False,
    ls=5.0,
    loff=(0, 0.35),
    ha="center",
    rot=0,
):
    from matplotlib.patches import FancyArrowPatch

    ax.add_patch(
        FancyArrowPatch(
            p,
            q,
            arrowstyle=style,
            mutation_scale=7,
            linewidth=lw,
            color=colour,
            linestyle="--" if dashed else "-",
            shrinkA=0.5,
            shrinkB=0.5,
        )
    )
    if label:
        mx, my = (p[0] + q[0]) / 2 + loff[0], (p[1] + q[1]) / 2 + loff[1]
        ax.text(
            mx,
            my,
            label,
            ha=ha,
            va="bottom",
            fontsize=ls,
            color=colour,
            rotation=rot,
            bbox={"boxstyle": "round,pad=0.1", "fc": "white", "ec": "none"},
        )


def _save(fig, out_dir: str, stem: str) -> list[str]:
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for ext, dpi in (("pdf", 300), ("png", 400)):
        path = out / f"{stem}.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
        written.append(str(path))
    plt.close(fig)
    return written


# -------------------------------------------------------------- 1. system architecture
def draw_system_architecture(out_dir: str = OUT_DIR) -> list[str]:
    """One pipeline row plus a training strip. Units: 0.1 inch."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _font(plt)
    W, H = 70.4, 30.0
    fig, ax = plt.subplots(figsize=(W / 10, H / 10))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- main row
    ry, rh = 14.0, 16.0
    _rect(
        ax,
        0.0,
        ry,
        10.2,
        rh,
        GREY,
        "Input speech",
        "Genuine: MUCS 2021\nHindi–English, 520 spk\n\nClones of the same\nspeakers\nseen: XTTS-v2, RVC\n"
        "unseen: Tortoise,\nIndicF5, Chatterbox",
        bs=4.9,
    )
    _arrow(ax, (10.2, ry + rh / 2), (11.5, ry + rh / 2))
    _rect(
        ax,
        11.5,
        ry,
        8.6,
        rh,
        ORANGE,
        "Pre-processing",
        "16 kHz mono\nVAD trims silence\nkeep 2–10 s\nlevel −23 dBFS",
        bs=4.9,
    )
    _arrow(ax, (20.1, ry + rh / 2), (21.4, ry + rh / 2))
    _rect(
        ax,
        21.4,
        ry,
        10.6,
        rh,
        ORANGE,
        "Telephone channel",
        "simulated line:\nG.711 µ-law, 8 kHz,\n+ noise 20 dB SNR\nor real codec:\nAMR-NB 12.2 kbit/s\n\n→ one 4 s crop\n(64,000 samples)",
        bs=4.9,
    )
    _arrow(ax, (32.0, ry + rh / 2), (33.3, ry + rh / 2))
    ex, ew = 33.3, 17.4
    _rect(ax, ex, ry, ew, rh, BLUE, None, None)
    ax.text(
        ex + ew / 2,
        ry + rh - 0.35,
        "wav2vec 2.0 encoder",
        ha="center",
        va="top",
        fontsize=6.0,
        fontweight="bold",
        color=INK,
    )
    _rect(
        ax,
        ex + 0.5,
        ry + 8.3,
        ew - 1.0,
        4.6,
        GREY,
        None,
        "CNN feature extractor\n7 conv layers, frozen\n→ 199 frames × 512 → 768",
        bs=4.8,
    )
    _arrow(ax, (ex + ew / 2, ry + 8.3), (ex + ew / 2, ry + 7.5), BLUE)
    _rect(ax, ex + 0.5, ry + 0.5, ew - 1.0, 7.0, BLUE, None, None)
    ax.text(
        ex + ew / 2,
        ry + 7.2,
        "12 transformer layers (12 heads, d = 768)",
        ha="center",
        va="top",
        fontsize=4.4,
        fontweight="bold",
        color=INK,
    )
    _rect(
        ax,
        ex + 1.0,
        ry + 0.9,
        ew - 2.0,
        4.0,
        GREEN,
        None,
        f"LoRA on q, k, v, out projections\nW (frozen) + (α/r)·B·A,  r = {LORA_R}, α = {LORA_ALPHA}\n"
        "48 adapters = 0.59 M params; + pooling, head = 1.08 M (1.13 %)",
        bs=4.3,
    )
    _arrow(ax, (ex + ew, ry + rh / 2), (ex + ew + 1.3, ry + rh / 2))
    px, pw = ex + ew + 1.3, 8.2
    _rect(
        ax,
        px,
        ry,
        pw,
        rh,
        GREEN,
        "Attentive\nstatistics\npooling",
        "attention weights\nover the 199 frames;\nweighted mean and\nweighted std\n→ 1536-dim vector",
        bs=4.8,
    )
    _arrow(ax, (px + pw, ry + rh / 2), (px + pw + 1.3, ry + rh / 2))
    hx, hw = px + pw + 1.3, 8.5
    _rect(
        ax,
        hx,
        ry + 7.2,
        hw,
        rh - 7.2,
        GREEN,
        "Classifier",
        f"Linear {2 * HIDDEN} → {HEAD_HIDDEN}\nReLU, dropout 0.3\nLinear {HEAD_HIDDEN} → 2\nsoftmax",
        bs=4.6,
    )
    _arrow(ax, (hx + hw / 2, ry + 7.2), (hx + hw / 2, ry + 6.3), GREEN)
    _rect(
        ax,
        hx,
        ry,
        hw,
        6.3,
        PURPLE,
        "P(genuine)",
        "offline: EER, false\nalarms vs floor\nlive: ladder, alert",
        bs=4.5,
    )

    # ---- training strip
    ty, th = 0.0, 12.0
    _rect(ax, 0.0, ty, hx + hw, th, GREY, None, None, fill=False, dashed=True, lw=0.7)
    ax.text(
        0.6,
        ty + th - 0.3,
        "Training (two stages)",
        fontsize=6.2,
        fontweight="bold",
        va="top",
        color=INK,
    )
    _rect(
        ax,
        0.6,
        ty + 0.6,
        26.4,
        th - 3.0,
        BLUE,
        "Stage 1 → S1  (English detector)",
        "data: ASVspoof 2019 LA, English only\ntrains transformer, pooling and head\n"
        f"AdamW lr {S1_LR}, batch {S1_BATCH}, {S1_EPOCHS} epochs; best dev epoch kept",
        bs=4.6,
    )
    _arrow(
        ax,
        (27.0, ty + 0.6 + (th - 3.0) / 2),
        (28.8, ty + 0.6 + (th - 3.0) / 2),
        BLUE,
        "S1",
        ls=4.3,
        loff=(0, 0.3),
    )
    _rect(
        ax,
        28.8,
        ty + 0.6,
        28.6,
        th - 3.0,
        GREEN,
        "Stage 2 → S2  (code-mixed telephone adapter)",
        "data: genuine + XTTS-v2 + RVC via the channel; trains LoRA, pooling, head\n"
        f"AdamW lr {S2_LR}, batch {S2_BATCH}, {S2_EPOCHS} epochs, five seeds\n"
        "deployed: 21 speakers, G.711  ·  final: 248 speakers, G.711 + AMR-NB",
        bs=4.6,
    )
    kx = 59.0
    ax.text(kx, ty + th - 1.6, "Colour key", fontsize=5.2, fontweight="bold", va="top", color=INK)
    keys = (
        (GREY, "frozen"),
        (BLUE, "trained in Stage 1"),
        (GREEN, "trained in Stage 2"),
        (PURPLE, "decision"),
    )
    for k, (c, t) in enumerate(keys):
        _rect(ax, kx, ty + th - 3.5 - k * 1.6, 1.4, 1.0, c)
        ax.text(kx + 1.9, ty + th - 3.0 - k * 1.6, t, fontsize=4.8, va="center", color=INK)
    _arrow(ax, (13.8, ty + th), (ex + 0.3, ry - 0.1), BLUE, dashed=True)
    _arrow(ax, (43.0, ty + th), (ex + 3.0, ry + 0.3), GREEN, dashed=True)
    _arrow(ax, (50.0, ty + th), (px + pw / 2, ry - 0.1), GREEN, dashed=True)

    return _save(fig, out_dir, "system_architecture_detailed")


# ------------------------------------------------------------ 2. algorithm flowchart
def _proc(ax, cx, y, w, h, text, colour=BLUE, bs=5.0):
    _rect(ax, cx - w / 2, y, w, h, colour, None, text, bs=bs)
    return (cx, y + h), (cx, y), (cx - w / 2, y + h / 2), (cx + w / 2, y + h / 2)


def _term(ax, cx, y, w, h, text, colour=INK):
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2, y),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={h / 2}",
            linewidth=0.9,
            edgecolor=colour,
            facecolor=colour + "10",
        )
    )
    ax.text(
        cx, y + h / 2, text, ha="center", va="center", fontsize=5.6, fontweight="bold", color=colour
    )
    return (cx, y + h), (cx, y), (cx - w / 2, y + h / 2), (cx + w / 2, y + h / 2)


def _decision(ax, cx, cy, w, h, text, colour=RED, bs=4.9):
    from matplotlib.patches import Polygon

    ax.add_patch(
        Polygon(
            [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)],
            closed=True,
            linewidth=0.9,
            edgecolor=colour,
            facecolor=colour + "12",
        )
    )
    ax.text(cx, cy, text, ha="center", va="center", fontsize=bs, color=INK, linespacing=1.2)
    return (cx, cy + h / 2), (cx, cy - h / 2), (cx - w / 2, cy), (cx + w / 2, cy)


def draw_algorithm_flowchart(out_dir: str = OUT_DIR) -> list[str]:
    """Single-column flowchart of the whole method, compact. Units: 0.1 inch."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _font(plt)
    W, H = 38.0, 47.0
    fig, ax = plt.subplots(figsize=(W / 10, H / 10))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")
    cx, bw, gap, bs = 18.0, 29.0, 1.0, 5.2

    y = H - 0.2
    _, bot, _, _ = _term(ax, cx, y - 1.8, 9, 1.8, "Start")
    y -= 1.8 + gap
    p1 = _proc(
        ax,
        cx,
        y - 3.4,
        bw,
        3.4,
        "1  Freeze speaker pools; generate clones (seen: XTTS-v2,\n"
        "RVC · held out: Tortoise, IndicF5, Chatterbox); pre-process\n"
        "(16 kHz, 2–10 s, −23 dBFS); render G.711 and AMR-NB copies",
        GREY,
        bs=bs,
    )
    _arrow(ax, bot, p1[0])
    y -= 3.4 + gap
    p2 = _proc(
        ax,
        cx,
        y - 2.6,
        bw,
        2.6,
        f"2  Stage 1: fine-tune wav2vec 2.0 on ASVspoof 2019 LA\n(English only), {S1_EPOCHS} epochs → S1",
        BLUE,
        bs=bs,
    )
    _arrow(ax, p1[1], p2[0])
    y -= 2.6 + gap
    p3 = _proc(
        ax,
        cx,
        y - 2.6,
        bw,
        2.6,
        f"3  Stage 2: freeze S1, add LoRA (r = {LORA_R}) to the attention\n"
        "projections; train one epoch on code-mixed telephone speech",
        GREEN,
        bs=bs,
    )
    _arrow(ax, p2[1], p3[0])
    y -= 2.6 + gap
    d1 = _decision(ax, cx, y - 1.8, 16.0, 3.6, "dev score better\nthan best so far?", bs=bs)
    _arrow(ax, p3[1], d1[0])
    _rect(ax, cx + 9.6, d1[3][1] - 1.1, 6.2, 2.2, GREEN, None, "keep this\nepoch", bs=4.8)
    _arrow(ax, d1[3], (cx + 9.6, d1[3][1]), GREEN, "yes", ls=4.8, loff=(0, 0.15))
    y -= 3.6 + gap
    d2 = _decision(ax, cx, y - 1.6, 16.0, 3.2, f"{S2_EPOCHS} epochs done?", bs=bs)
    _arrow(ax, d1[1], d2[0], MUTED, "no", ls=4.8, loff=(0.5, -0.15), ha="left")
    _arrow(ax, (cx + 12.7, d1[3][1] - 1.1), (cx + 12.7, d2[3][1]), GREEN, style="-")
    _arrow(ax, (cx + 12.7, d2[3][1]), d2[3], GREEN)
    lx = cx - bw / 2 - 1.2
    _arrow(ax, d2[2], (lx, d2[2][1]), MUTED, "no", ls=4.8, loff=(0, 0.15), style="-")
    _arrow(ax, (lx, d2[2][1]), (lx, p3[2][1]), MUTED, style="-")
    _arrow(ax, (lx, p3[2][1]), p3[2], MUTED)
    ax.text(
        lx - 0.3,
        (d2[2][1] + p3[2][1]) / 2,
        "next epoch",
        rotation=90,
        fontsize=4.6,
        color=MUTED,
        ha="right",
        va="center",
    )
    y -= 3.2 + gap
    p4 = _proc(
        ax,
        cx,
        y - 2.6,
        bw,
        2.6,
        "4  Score each test clip once (4 s crop) → EER, false alarms;\n"
        "fit the matched shortcut floor on the same clips",
        RED,
        bs=bs,
    )
    _arrow(ax, d2[1], p4[0], MUTED, "yes", ls=4.8, loff=(0.5, -0.1), ha="left")
    y -= 2.6 + gap
    d3 = _decision(
        ax, cx, y - 1.9, 17.0, 3.8, "bootstrap 95 % CI of\nEER(det) − EER(floor) < 0?", bs=4.9
    )
    _arrow(ax, p4[1], d3[0])
    _rect(ax, cx + 10.2, d3[3][1] - 1.1, 6.0, 2.2, GREEN, None, "report as\nevidence", bs=4.8)
    _arrow(ax, d3[3], (cx + 10.2, d3[3][1]), GREEN, "yes", ls=4.8, loff=(0, 0.15))
    _rect(ax, cx - 16.2, d3[2][1] - 1.1, 6.0, 2.2, MUTED, None, "report as\ninconclusive", bs=4.6)
    _arrow(ax, d3[2], (cx - 10.2, d3[2][1]), MUTED, "no", ls=4.8, loff=(0, 0.15))
    y -= 3.8 + gap
    d4 = _decision(ax, cx, y - 1.6, 16.0, 3.2, "all 5 seeds done?", bs=bs)
    _arrow(ax, d3[1], d4[0])
    lx2 = cx + bw / 2 + 4.6
    _arrow(ax, d4[3], (lx2, d4[3][1]), MUTED, "no", ls=4.8, loff=(0, 0.15), style="-")
    _arrow(ax, (lx2, d4[3][1]), (lx2, p3[3][1] - 0.6), MUTED, style="-")
    _arrow(ax, (lx2, p3[3][1] - 0.6), (p3[3][0], p3[3][1] - 0.6), MUTED)
    ax.text(
        lx2 + 0.3,
        (d4[3][1] + p3[3][1]) / 2,
        "next seed",
        rotation=90,
        fontsize=4.6,
        color=MUTED,
        ha="left",
        va="center",
    )
    y -= 3.2 + gap
    p5 = _proc(
        ax,
        cx,
        y - 2.6,
        bw,
        2.6,
        "5  Across seeds: mean ± SD; exact Mann–Whitney test\nbetween recipes",
        BLUE,
        bs=bs,
    )
    _arrow(ax, d4[1], p5[0], MUTED, "yes", ls=4.8, loff=(0.5, -0.1), ha="left")
    y -= 2.6 + gap
    p6 = _proc(
        ax,
        cx,
        y - 3.4,
        bw,
        3.4,
        "6  Deploy the chosen checkpoint: 4 s window every 2 s → EMA\n"
        "→ threshold θ → caution after k1, 'likely cloned' after k2\n"
        "consecutive fake windows → alert the receiver",
        PURPLE,
        bs=bs,
    )
    _arrow(ax, p5[1], p6[0])
    y -= 3.4 + gap
    _term(ax, cx, y - 1.8, 9, 1.8, "End")
    _arrow(ax, p6[1], (cx, y))
    return _save(fig, out_dir, "algorithm_flowchart")


def draw(out_dir: str = OUT_DIR) -> list[str]:
    """Draw both figures and return the files written."""
    return draw_system_architecture(out_dir) + draw_algorithm_flowchart(out_dir)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Draw the architecture and algorithm figures")
    parser.add_argument("--out", default=OUT_DIR)
    args = parser.parse_args()
    for path in draw(args.out):
        print(path)


if __name__ == "__main__":
    main()
