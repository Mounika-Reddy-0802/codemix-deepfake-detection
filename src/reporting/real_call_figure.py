"""The call chain figure: why the deployed model fails on real calls.

Each speaker's direct reading, passed through the stages of a phone call
(``src.inference.real_call_codecs``), and the genuine live calls themselves: the mean P(genuine)
of each model at each stage. One line per model, in the colours the overview figure gives the
same models, with every point labelled and a different marker per model, so neither colour nor
the legend alone carries the identity. The dashed lines are each model's live threshold: a mean
below it is a call the model would warn about. Every number is read from
``experiments/results/real_calls_v1``; nothing is typed in.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.reporting.figures import DPI, FIGURES_DIR, _style

R = Path("experiments/results/real_calls_v1")
#: (stage key in codec_chain.csv, axis label); the last stage is the real calls.
STAGES = (
    ("direct", "Direct"),
    ("g711", "G.711\n(simulated)"),
    ("amr_nb", "AMR-NB\n(real codec)"),
    ("amr_nb_rec", "AMR-NB\n+ recorder"),
    ("amr_wb_rec", "AMR-WB\n+ recorder"),
    ("real_calls", "Real calls\n(genuine)"),
)
#: Model -> (label, colour, marker, live threshold file). Colours as in ``final_figure``.
MODELS = {
    "deployed": ("Deployed adapter", "#eb6834", "o", "configs/threshold.yaml"),
    "final": ("Final-recipe adapter", "#1baf7a", "s", "configs/threshold_final.yaml"),
}


def table(root: str | Path = R, plan: pd.DataFrame | None = None) -> pd.DataFrame:
    """Mean P(genuine) per model and stage: the codec chain, then the genuine live calls."""
    from src.inference.real_calls import CALL_PLAN

    root = Path(root)
    plan = CALL_PLAN if plan is None else plan
    chain = pd.read_csv(root / "codec_chain.csv")
    rows = [
        {"model": model, "stage": stage, "mean_p_genuine": float(group["mean_p_genuine"].mean())}
        for (model, stage), group in chain.groupby(["model", "version"])
    ]
    live = set(plan.loc[plan["kind"] == "genuine_live", "call_id"])
    for model in MODELS:
        path = root / f"windows_{model}_g711.csv"
        if not path.is_file():
            continue
        w = pd.read_csv(path).dropna(subset=["score"])
        call = w["filepath"].map(lambda p: Path(str(p)).stem)
        w = w[call.isin(live)]
        if len(w):
            # each call counts once, as each recording does in the chain
            per_call = w.groupby(call[call.isin(live)])["score"].mean()
            rows.append(
                {"model": model, "stage": "real_calls", "mean_p_genuine": float(per_call.mean())}
            )
    order = {key: i for i, (key, _) in enumerate(STAGES)}
    out = pd.DataFrame(rows)
    out = out[out["stage"].isin(order)]
    return out.sort_values(
        ["model", "stage"], key=lambda s: s.map(order) if s.name == "stage" else s
    )


def _placement(y: float, thresholds) -> tuple[tuple[int, int], str]:
    """Where a point's label goes: above it, or below and to the left of it (clear of a falling
    line) unless that would sit on a threshold line or the axis."""
    below = y - 0.075
    if below < 0.05 or y > 0.85 or any(abs(below - t) < 0.05 for t in thresholds):
        return (0, 6), "center"
    return (-3, -10), "right"


def figure(out_dir: str = FIGURES_DIR, root: str | Path = R) -> str:
    """Write ``real_call_chain.png`` and ``.pdf``; returns the PNG path."""
    from live_call.verdict_engine import load_config

    data = table(root)
    plt = _style()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    labels = [label for _, label in STAGES]
    x_of = {key: i for i, (key, _) in enumerate(STAGES)}
    thresholds = {model: load_config(spec[3]).threshold for model, spec in MODELS.items()}
    for model, (name, colour, marker, _) in MODELS.items():
        part = data[data["model"] == model]
        xs = [x_of[s] for s in part["stage"]]
        ys = list(part["mean_p_genuine"])
        ax.plot(xs, ys, color=colour, marker=marker, markersize=4.5, lw=1.6, label=name, zorder=3)
        for x, y in zip(xs, ys, strict=True):
            xytext, ha = _placement(y, thresholds.values())
            ax.annotate(
                f"{y:.2f}",
                (x, y),
                textcoords="offset points",
                xytext=xytext,
                ha=ha,
                fontsize=6,
                color="#333333",
            )
        ax.axhline(thresholds[model], ls="--", lw=0.7, color=colour, zorder=1)
        ax.text(
            len(STAGES) - 0.55,
            thresholds[model] + 0.015,
            f"threshold {thresholds[model]:.2f}",
            fontsize=5.5,
            color="#333333",
            ha="right",
            va="bottom",
        )
    ax.set_xticks(range(len(labels)), labels, fontsize=5.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Mean P(genuine)", fontsize=7)
    ax.tick_params(axis="y", labelsize=6)
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=6, frameon=False, loc="lower left")
    out = Path(out_dir) / "real_call_chain.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return str(out)


if __name__ == "__main__":
    print(table().to_string(index=False))
    print("wrote", figure())
