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
        ax.set_title(
            f"{NAMES[f['unseen_tool']]}, never heard\n"
            f"lower in {f['lower_in_seeds']}/5 seeds, p = {f['mann_whitney_p']:.3f}: {f['decision']}",
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
    paths = []
    for ext in ("png", "pdf"):
        path = Path(f"{out}.{ext}")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        paths.append(path)
    plt.close(fig)
    return paths


def main() -> None:
    report = json.loads(Path(REPORT).read_text(encoding="utf-8"))
    for path in draw(report):
        print("written", path)


if __name__ == "__main__":
    main()
