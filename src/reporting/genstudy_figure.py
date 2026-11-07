"""Figure of the generator study (P-041), drawn from the rule's own output file.

Reads only ``experiments/results/generator_study_v1.json`` (written by
``src.reporting.generator_study``), so the figure cannot disagree with the verdict:

* panels 1-2: each fold's EER on the tool it never heard, seed by seed, against the control
  (one line per seed; held-out speakers only);
* panel 3: every system's mean EER on IndicF5, Chatterbox and new-speaker Tortoise.

    python -m src.reporting.genstudy_figure
"""

from __future__ import annotations

import json
from pathlib import Path

from src.reporting.figures import _style

REPORT = "experiments/results/generator_study_v1.json"
OUT = "experiments/figures/generator_study"
TRAINED_ON = {"fold_a": "IndicF5", "fold_b": "Chatterbox"}
NAMES = {
    "chatterbox": "Chatterbox",
    "indicf5": "IndicF5",
    "new_tortoise": "Tortoise (new speakers)",
}
#: why a fold is not confirmed, one phrase per failed condition of the rule
FAILED = {
    "1_lower_in_4_of_5": lambda f: f"lower in only {f['lower_in_seeds']} of 5 seeds",
    "2_mann_whitney": lambda f: f"p = {f['mann_whitney_p']:.3f}, limit 0.025",
    "3_tortoise_within_1_point": lambda f: (
        f"Tortoise {100 * f['tortoise_mean_rise']:+.2f} points, limit +1.00"
    ),
    "4_false_alarms_within_half_point": lambda f: (
        f"genuine called fake {100 * f['false_alarm_mean_rise']:+.2f} points, limit +0.50"
    ),
}
SYSTEMS = (
    ("control", "#7f7f7f"),
    ("fold_a", "#4393c3"),
    ("fold_b", "#d6604d"),
    ("fold_c", "#5aae61"),
)


def draw(report: dict, out: str = OUT) -> list[Path]:
    """Write the PNG and PDF; refuses a report without the judged folds."""
    if "folds" not in report:
        raise ValueError(f"the study is {report.get('answer')!r}: nothing to draw yet")
    plt = _style()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), gridspec_kw={"width_ratios": [1, 1, 1.5]})
    for ax, (fold, f) in zip(axes[:2], report["folds"].items(), strict=True):
        ctrl = [100 * v for v in f["control"]]
        fold_v = [100 * v for v in f["fold"]]
        for c, v in zip(ctrl, fold_v, strict=True):
            ax.plot([0, 1], [c, v], color="#bbbbbb", lw=1, zorder=1)
        ax.scatter([0] * len(ctrl), ctrl, color="#7f7f7f", zorder=2, s=18)
        ax.scatter([1] * len(fold_v), fold_v, color=dict(SYSTEMS)[fold], zorder=2, s=18)
        ax.set_xticks([0, 1], ["control", f"+ {TRAINED_ON[fold]}"])
        ax.set_xlim(-0.4, 1.4)
        ax.set_ylabel("EER (%)")
        failed = [FAILED[k](f) for k, ok in f["conditions"].items() if not ok]
        ax.set_title(
            f"{NAMES[f['unseen_tool']]}, never heard\n"
            f"lower in {f['lower_in_seeds']}/5 seeds, p = {f['mann_whitney_p']:.3f}: {f['decision']}"
            + "".join(f"\n({why})" for why in failed),
            fontsize=8.5,
        )
    ax = axes[2]
    tools = ("indicf5", "chatterbox", "new_tortoise")
    width = 0.2
    for k, (system, colour) in enumerate(SYSTEMS):
        means = report["means"].get(system, {})
        vals = [100 * means[t]["mean"] if t in means else 0.0 for t in tools]
        ax.bar(
            [i + (k - 1.5) * width for i in range(len(tools))],
            vals,
            width,
            color=colour,
            label=system,
        )
    ax.set_xticks(range(len(tools)), [NAMES[t] for t in tools], fontsize=8)
    ax.set_ylabel("mean EER over seeds (%)")
    ax.set_title(f"All systems, held-out speakers (answer: {report['answer']})", fontsize=8.5)
    ax.legend(fontsize=7, ncols=2)
    fig.tight_layout()
    return _save(plt, fig, out)


def draw_column(report: dict, out: str = OUT + "_column") -> list[Path]:
    """The two judged folds only, one IEEE column wide (3.5 in), for the papers.

    Same numbers as panels 1-2 of :func:`draw`; the reasons a fold is not confirmed go in the
    caption, the decision stays on the panel so the figure cannot be read against the rule.
    """
    if "folds" not in report:
        raise ValueError(f"the study is {report.get('answer')!r}: nothing to draw yet")
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.2), sharey=True)
    for k, (ax, (fold, f)) in enumerate(zip(axes, report["folds"].items(), strict=True)):
        ctrl = [100 * v for v in f["control"]]
        fold_v = [100 * v for v in f["fold"]]
        for c, v in zip(ctrl, fold_v, strict=True):
            ax.plot([0, 1], [c, v], color="#999999", lw=0.8, zorder=1)
        ax.scatter([0] * len(ctrl), ctrl, color="#555555", marker="o", zorder=2, s=10)
        ax.scatter([1] * len(fold_v), fold_v, color=dict(SYSTEMS)[fold], marker="s", zorder=2, s=10)
        ax.set_xticks([0, 1], ["control", f"+{TRAINED_ON[fold]}"], fontsize=6)
        ax.set_xlim(-0.35, 1.35)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_title(
            f"({chr(ord('a') + k)}) {NAMES[f['unseen_tool']]}, never heard\n"
            f"p = {f['mann_whitney_p']:.3f}, {f['decision']}",
            fontsize=6.6,
        )
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("EER (%), held-out speakers", fontsize=6.6)
    axes[0].set_ylim(bottom=0)
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.12, top=0.84, wspace=0.12)
    return _save(plt, fig, out, dpi=300)


def _save(plt, fig, out: str, dpi: int | None = None) -> list[Path]:
    paths = []
    for ext in ("png", "pdf"):
        path = Path(f"{out}.{ext}")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi)
        paths.append(path)
    plt.close(fig)
    return paths


def main() -> None:
    report = json.loads(Path(REPORT).read_text(encoding="utf-8"))
    for path in [*draw(report), *draw_column(report)]:
        print("written", path)


if __name__ == "__main__":
    main()
