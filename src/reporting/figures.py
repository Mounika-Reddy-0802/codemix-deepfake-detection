"""Publication figures, built only from committed measurements (W9-T2, owner M).

The plan asks for "consolidated final tables + publication figures". The hard part
is not drawing them — it is making sure the figure and the number in the paper can
never drift apart. So every value plotted here comes from a **committed artefact**:
a results JSON, a per-clip score CSV, or the :data:`SYSTEM_MATRIX` table below whose
every cell carries the document it was measured in.

Three figures, and each one exists because a reviewer will ask for it:

- :func:`system_matrix_figure` — the headline. Three systems × three conditions in
  one grid, which is the only view that shows the whole finding at once: the
  English-trained baseline collapses on Hinglish, the clean adapter fixes that and
  dies on a phone line, and the channel-matched adapter is the one that holds.
- :func:`det_curves_figure` — DET curves on the 71,237-clip ASVspoof eval partition,
  computed from **per-clip scores**, not from a summary. This is the evidence that
  the English-retention numbers are a real distribution and not three point
  estimates.
- :func:`shortcut_gate_figure` — the low-level-cue gate for CM01 and CM02 against
  chance, because per `lowlevel_cue_check_v1.md` no model number on this corpus can
  be read without its shortcut baseline beside it.

Figures that the plan lists but that have **no data yet** are not drawn and not
faked: the cross-eval (`affectdf_crosseval.json`) and reverse-degradation
(`s3_reverse_degradation.json`) artefacts do not exist, and S3 has never been
trained. :func:`missing_inputs` reports exactly that, and the CLI prints it.

Matplotlib is imported lazily so this module stays importable in CI, which installs
only ruff/pytest/numpy/pandas/pyyaml.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FIGURES_DIR = "experiments/figures"
DPI = 200

#: EER (%) per system and condition, each cell with the document it was measured in.
#: Kept as data rather than read from JSON because these runs live in several files
#: with different shapes -- and a hand-copied number with no provenance is exactly
#: what this project's own results docs keep catching. Every entry is checked by
#: ``tests/test_figures.py`` against the source doc named here.
SYSTEM_MATRIX: dict[str, dict[str, tuple[float, str]]] = {
    "S1 baseline\n(English-trained)": {
        "English\n(ASVspoof eval)": (0.85, "asvspoof_retention_summary.json"),
        "Code-mixed\nclean": (53.71, "gap_closure_v1.md"),
        "Code-mixed\nchannel (G.711)": (54.92, "channel_matched_v1.md"),
    },
    "S2 LoRA\n(clean-trained)": {
        "English\n(ASVspoof eval)": (8.38, "asvspoof_retention_summary.json"),
        "Code-mixed\nclean": (1.34, "gap_closure_v1.md"),
        "Code-mixed\nchannel (G.711)": (38.58, "channel_matched_v1.md"),
    },
    "S2 LoRA\n(channel-matched)": {
        "English\n(ASVspoof eval)": (4.83, "asvspoof_retention_summary.json"),
        "Code-mixed\nclean": (13.92, "channel_matched_v1.md"),
        "Code-mixed\nchannel (G.711)": (3.89, "channel_matched_v1.md"),
    },
}

#: Low-level-cue gate results. Chance is 50%; lower means a stronger shortcut.
SHORTCUT_GATE: tuple[tuple[str, float, str], ...] = (
    ("CM01 clean", 1.39, "lowlevel_cue_check.json"),
    ("CM01 clean\n+ normalised", 5.17, "lowlevel_cue_check.json"),
    ("CM01\nchannel-matched", 9.25, "lowlevel_cue_check_channel20.json"),
    ("CM02 raw", 22.42, "lowlevel_cue_check_cm02_raw.json"),
    ("CM02\nnormalised", 22.28, "lowlevel_cue_check_cm02_normalised.json"),
    ("AffectDF\n(reference)", 53.16, "AffectDF Appendix G"),
)

#: Per-clip score CSVs for the DET figure, in legend order.
DET_SOURCES: tuple[tuple[str, str], ...] = (
    ("S1 baseline", "experiments/asvspoof_retention_stage1_baseline_scores.csv"),
    ("S2 LoRA (clean)", "experiments/asvspoof_retention_lora_clean_scores.csv"),
    ("S2 LoRA (channel-matched)", "experiments/asvspoof_retention_lora_channel_scores.csv"),
)

#: Artefacts the plan's figure list needs that do not exist yet.
PLANNED_BUT_UNMEASURED: tuple[tuple[str, str], ...] = (
    ("cross-eval column", "experiments/results/affectdf_crosseval.json"),
    ("reverse degradation (S3 on English)", "experiments/results/s3_reverse_degradation.json"),
    ("ablations (XTTS-only vs XTTS+RVC)", "experiments/results/ablations.json"),
)


class FigureDataError(RuntimeError):
    """Raised when a figure is asked for and its measured input is absent."""


@dataclass(frozen=True)
class DetCurve:
    """One system's detection-error-tradeoff curve plus its equal-error point."""

    label: str
    false_alarm: np.ndarray  # P(bonafide called spoof is 1 - this); x axis, %
    miss: np.ndarray  # y axis, %
    eer: float
    clips: int


# --------------------------------------------------------------------------- #
# Pure computation (no plotting, so CI exercises it)
# --------------------------------------------------------------------------- #
def det_points(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """False-alarm and miss rates (%) swept over every distinct threshold.

    ``labels`` is 1 for bonafide. The detector emits P(bonafide), so a *miss* is a
    spoof scored high and a *false alarm* is a bonafide scored low -- the same
    convention ``src.training.metrics.compute_eer`` uses, so the EER marked on the
    curve and the EER in the results JSON cannot disagree.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    order = np.argsort(scores, kind="stable")
    sorted_labels = labels[order]

    n_bonafide = int(sorted_labels.sum())
    n_spoof = int(len(sorted_labels) - n_bonafide)
    if n_bonafide == 0 or n_spoof == 0:
        raise FigureDataError("a DET curve needs both classes present")

    # Sweeping the threshold upward: everything at or below it is called spoof.
    false_alarm = np.cumsum(sorted_labels) / n_bonafide  # bonafide called spoof
    miss = 1.0 - (np.cumsum(1 - sorted_labels) / n_spoof)  # spoof called bonafide
    return false_alarm * 100.0, miss * 100.0


def curve_from_scores(label: str, csv_path: str) -> DetCurve:
    """Build one :class:`DetCurve` from a committed per-clip score CSV."""
    path = Path(csv_path)
    if not path.is_file():
        raise FigureDataError(f"score CSV not found: {csv_path}")
    frame = pd.read_csv(path)
    for column in ("label", "score"):
        if column not in frame.columns:
            raise FigureDataError(f"{csv_path} has no {column!r} column")

    labels = (frame["label"].astype(str) == "bonafide").to_numpy().astype(int)
    scores = frame["score"].to_numpy(dtype=float)

    from src.training.metrics import eer as compute

    return DetCurve(
        label=label,
        false_alarm=det_points(scores, labels)[0],
        miss=det_points(scores, labels)[1],
        eer=float(compute(scores, labels)) * 100.0,
        clips=int(len(frame)),
    )


def matrix_frame(matrix: dict[str, dict[str, tuple[float, str]]] | None = None) -> pd.DataFrame:
    """The system × condition EER table, as a frame (rows = systems)."""
    data = matrix or SYSTEM_MATRIX
    return pd.DataFrame(
        {
            condition: {system: cells[condition][0] for system, cells in data.items()}
            for condition in next(iter(data.values()))
        }
    )


def matrix_markdown(matrix: dict[str, dict[str, tuple[float, str]]] | None = None) -> str:
    """The system × condition table as markdown, for pasting into the paper.

    Hand-rolled rather than ``DataFrame.to_markdown`` because that needs
    ``tabulate``, and CI installs only ruff/pytest/numpy/pandas/pyyaml -- a table
    helper is not worth a dependency the test environment would have to grow.
    """
    frame = matrix_frame(matrix)
    flat = lambda text: " ".join(str(text).split())  # noqa: E731 - local formatting shim
    header = "| System | " + " | ".join(flat(c) for c in frame.columns) + " |"
    rule = "|---|" + "---:|" * frame.shape[1]
    rows = [
        "| "
        + flat(system)
        + " | "
        + " | ".join(f"{value:.2f}%" for value in frame.loc[system])
        + " |"
        for system in frame.index
    ]
    return "\n".join([header, rule, *rows])


def missing_inputs() -> list[tuple[str, str]]:
    """Planned figures whose measured input does not exist yet."""
    return [(name, path) for name, path in PLANNED_BUT_UNMEASURED if not Path(path).is_file()]


# --------------------------------------------------------------------------- #
# Plotting (matplotlib imported lazily)
# --------------------------------------------------------------------------- #
def _style():
    import matplotlib

    matplotlib.use("Agg")  # headless: this runs in CI and over SSH
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def system_matrix_figure(out_dir: str = FIGURES_DIR) -> str:
    """Three systems × three conditions, EER heatmap. The headline figure."""
    plt = _style()
    frame = matrix_frame()

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    values = frame.to_numpy(dtype=float)
    # Log-scaled colour: the cells span 0.85% to 54.92%, so a linear map would show
    # four indistinguishable dark squares and one bright one.
    image = ax.imshow(np.log10(values), cmap="RdYlGn_r", aspect="auto")

    ax.set_xticks(range(frame.shape[1]), frame.columns, fontsize=8)
    ax.set_yticks(range(frame.shape[0]), frame.index, fontsize=8)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}%",
                ha="center",
                va="center",
                fontsize=11,
                color="white" if value > 20 else "black",
                fontweight="bold",
            )
    ax.set_title("EER by system and evaluation condition (lower is better)", fontsize=10, pad=10)
    bar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    bar.set_label("log10 EER (%)", fontsize=8)
    bar.ax.tick_params(labelsize=7)

    out = Path(out_dir) / "system_matrix.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return str(out)


def det_curves_figure(out_dir: str = FIGURES_DIR, sources=DET_SOURCES) -> str:
    """DET curves on the ASVspoof eval partition, from per-clip scores."""
    plt = _style()
    curves = [curve_from_scores(label, path) for label, path in sources]

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for curve in curves:
        ax.plot(
            curve.false_alarm, curve.miss, lw=1.6, label=f"{curve.label} — {curve.eer:.2f}% EER"
        )
        ax.plot(curve.eer, curve.eer, "o", ms=4, color=ax.lines[-1].get_color())

    limit = max(1.0, max(c.eer for c in curves) * 4)
    ax.plot([0, limit], [0, limit], ls=":", lw=0.9, color="grey", label="equal error")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("False alarm rate (%) — genuine speech called spoof")
    ax.set_ylabel("Miss rate (%) — spoof called genuine")
    ax.set_title(f"DET on ASVspoof 2019 LA eval ({curves[0].clips:,} clips)", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper right")

    out = Path(out_dir) / "det_asvspoof_eval.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return str(out)


def shortcut_gate_figure(out_dir: str = FIGURES_DIR) -> str:
    """Low-level-cue gate per corpus condition, against chance."""
    plt = _style()
    names = [n for n, _, _ in SHORTCUT_GATE]
    values = [v for _, v, _ in SHORTCUT_GATE]

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    colours = ["#b2182b" if v < 10 else "#ef8a62" if v < 40 else "#4d9221" for v in values]
    bars = ax.bar(names, values, color=colours, width=0.62)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.2,
            f"{value:.2f}%",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
        )

    ax.axhline(50, ls="--", lw=1.1, color="grey")
    # Left-aligned: the right-hand end of this line sits on top of the tallest bar.
    ax.text(-0.42, 51.2, "chance (50%) — a clean corpus", fontsize=7.5, color="grey", ha="left")
    ax.set_ylim(0, 62)
    ax.set_ylabel("Low-level-cue EER (%)")
    ax.set_title(
        "Can eight cheap signal statistics separate the classes? Higher is safer.",
        fontsize=10,
    )
    ax.tick_params(axis="x", labelsize=7.5)

    out = Path(out_dir) / "shortcut_gate.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return str(out)


def build_all(out_dir: str = FIGURES_DIR) -> list[str]:
    """Every figure whose measured input exists. Returns the paths written."""
    return [
        system_matrix_figure(out_dir),
        det_curves_figure(out_dir),
        shortcut_gate_figure(out_dir),
    ]


def main() -> None:
    """CLI: ``python -m src.reporting.figures [--out-dir experiments/figures]``."""
    import argparse

    parser = argparse.ArgumentParser(description="Build the publication figures")
    parser.add_argument("--out-dir", default=FIGURES_DIR)
    parser.add_argument("--table", action="store_true", help="also print the matrix as markdown")
    args = parser.parse_args()

    for path in build_all(args.out_dir):
        size = Path(path).stat().st_size / 1024
        print(f"wrote {path}  ({size:.0f} KB)")

    if args.table:
        print()
        print(matrix_markdown())

    absent = missing_inputs()
    if absent:
        print("\nnot drawn -- no measured input exists yet:")
        for name, path in absent:
            print(f"  - {name}  ({path})")
        print("These are real gaps in the results, not missing plotting code.")


if __name__ == "__main__":
    main()
