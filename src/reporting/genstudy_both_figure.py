"""The generator study on both detectors in one figure (P-041 and P-044), for the papers.

Reads only the two rules' output files, so the figure cannot disagree with either verdict:
``experiments/results/generator_study_v1.json`` (the final recipe's adapter, five seeds) and
``experiments/results/ssl_generator_study_v1.json`` (SSL-AASIST, two seeds). One panel per
never-heard tool; in each, the control and the fold seed by seed (one line per seed, EER on
held-out speakers), the adapter on the left and SSL-AASIST on the right, each pair under its
own decision. One IEEE column wide::

    python -m src.reporting.genstudy_both_figure
"""

from __future__ import annotations

import json
from pathlib import Path

from src.reporting.figures import _style
from src.reporting.genstudy_figure import NAMES, SYSTEMS, TRAINED_ON, _save

ADAPTER = "experiments/results/generator_study_v1.json"
SSL = "experiments/results/ssl_generator_study_v1.json"
OUT = "experiments/figures/generator_study_both_column"
#: (report key, short name, x of the control) for the two detectors
DETECTORS = (("adapter", "adapter", 0.0), ("ssl", "SSL-AASIST", 2.3))


def draw(reports: dict[str, dict], out: str = OUT) -> list[Path]:
    """Write the PNG and PDF; refuses a report whose folds are not judged."""
    for name, report in reports.items():
        if "folds" not in report:
            raise ValueError(f"{name}: the study is {report.get('answer')!r}, nothing to draw")
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.4), sharey=True)
    folds = list(reports["adapter"]["folds"])
    for k, (ax, fold) in enumerate(zip(axes, folds, strict=True)):
        ticks, labels = [], []
        for key, short, x0 in DETECTORS:
            f = reports[key]["folds"][fold]
            ctrl = [100 * v for v in f["control"]]
            fold_v = [100 * v for v in f["fold"]]
            for c, v in zip(ctrl, fold_v, strict=True):
                ax.plot([x0, x0 + 1], [c, v], color="#999999", lw=0.8, zorder=1)
            ax.scatter([x0] * len(ctrl), ctrl, color="#555555", marker="o", zorder=2, s=10)
            ax.scatter(
                [x0 + 1] * len(fold_v),
                fold_v,
                color=dict(SYSTEMS)[fold],
                marker="s",
                zorder=2,
                s=10,
            )
            ticks += [x0, x0 + 1]
            labels += ["control", "fold"]
            ax.text(
                x0 + 0.5,
                -0.17,
                f"{short}, {len(ctrl)} seeds\n{f['decision']}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=5.6,
            )
        ax.set_xticks(ticks, labels, fontsize=5.8)
        ax.set_xlim(-0.4, DETECTORS[-1][2] + 1.4)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_title(
            f"({chr(ord('a') + k)}) {NAMES[f['unseen_tool']]}, never heard\n"
            f"fold = control + {TRAINED_ON[fold]} clones",
            fontsize=6.4,
        )
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("EER (%), held-out speakers", fontsize=6.6)
    axes[0].set_ylim(bottom=0)
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.24, top=0.86, wspace=0.10)
    return _save(plt, fig, out, dpi=300)


def main() -> None:
    reports = {
        "adapter": json.loads(Path(ADAPTER).read_text(encoding="utf-8")),
        "ssl": json.loads(Path(SSL).read_text(encoding="utf-8")),
    }
    for path in draw(reports):
        print("written", path)


if __name__ == "__main__":
    main()
