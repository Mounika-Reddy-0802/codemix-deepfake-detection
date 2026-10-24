"""One overview figure: every main system against every attack tool, with the shortcut floor.

The paper reports the final systems in several tables; this figure puts them side by side.
Two panels share one y-axis (EER, %): the simulated G.711 phone line and the real AMR-NB
mobile codec. In each panel, one group of bars per attack tool, one bar per system:

* S1, the English-only detector (no AMR-NB scores exist for it, so it appears only on the left);
* the deployed adapter (21 speakers, G.711 training);
* the final-recipe adapter (248 speakers, G.711 + AMR-NB training);
* SSL-AASIST, the larger encoder, trained on the final-recipe data.

Bars are means over training seeds with the standard deviation as an error bar (one model
where only one exists, said in the caption). The black tick over each group is the shortcut
floor fitted on the final training data (``src.training.floor_final``): it belongs to the two
final systems. Every number is read from a committed results file; nothing is typed in.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.reporting.figures import DPI, FIGURES_DIR, _style

R = Path("experiments/results")

SYSTEMS = ("S1, English only", "Deployed adapter", "Final-recipe adapter", "SSL-AASIST, final data")
#: Categorical slots 1-4 of the reference palette, validated for colour-blind separation.
COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
HATCHES = ("", "//", "..", "xx")  # a second cue for greyscale print

#: (panel, tool label, field in runs.json, field in floor_final_v1.json)
CELLS = (
    ("G.711 phone line (simulated)", "XTTS-v2\n(seen tool)", "new_xtts", "new_xtts"),
    ("G.711 phone line (simulated)", "Tortoise", "new_tortoise", "new_tortoise"),
    ("G.711 phone line (simulated)", "IndicF5", "cm07", "cm07"),
    ("G.711 phone line (simulated)", "Chatterbox", "cm09", "cm09"),
    ("G.711 phone line (simulated)", "RVC", "rvc_test", "rvc_test"),
    ("AMR-NB codec (real)", "XTTS-v2\n(seen tool)", "new_xtts_amr", "new_xtts_amr"),
    ("AMR-NB codec (real)", "Tortoise", "new_tortoise_amr", "new_tortoise_amr"),
    ("AMR-NB codec (real)", "IndicF5", "cm07_amr", "cm07_amr"),
)


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _per_attack(result: dict, attack: str) -> float:
    for row in result["per_attack"]:
        if row["attack"] == attack:
            return float(row["eer"])
    raise KeyError(attack)


def _s1(field: str, root: Path) -> float | None:
    """S1's EER for a G.711 field, or None where S1 was not scored."""
    sources = {
        "new_xtts": ("ext_scoring_v1/s1__score_ext_channel.json", "xtts_v2"),
        "new_tortoise": ("ext_scoring_v1/s1__score_ext_channel.json", "tortoise"),
        "cm07": ("ext_scoring_v1/s1__score_cm07_channel.json", None),
        "cm09": ("cm09_v1/s1__score_cm09_channel.json", None),
    }
    if field == "rvc_test":
        path = root.parent.parent / "results/s1_norm/stage1__rvc_test_channel.json"
        return float(_load(path)["pooled"]["eer"]) if path.is_file() else None
    if field not in sources:
        return None
    rel, attack = sources[field]
    path = root / rel
    if not path.is_file():
        return None
    result = _load(path)
    return _per_attack(result, attack) if attack else float(result["pooled"]["eer"])


def table(root: str | Path = R) -> pd.DataFrame:
    """One row per (panel, tool, system): mean, sd and n over seeds, and the final-data floor."""
    root = Path(root)
    runs4 = _load(root / "final_v1/runs.json")
    ssl = _load(root / "final_ssl_v1/runs.json") + _load(root / "final_ssl_seeds2_v1/runs.json")
    floors = {
        r["test_set"]: r["floor_eer"]
        for r in _load(root / "floor_final_v1.json")["summary"]
        if r["detector"] == "adapter"
    }
    single = {  # Chatterbox was scored on the laptop, one model each
        ("Deployed adapter", "cm09"): root / "cm09_v1/deployed_s2__score_cm09_channel.json",
        ("Final-recipe adapter", "cm09"): root / "cm09_v1/final_s1234__score_cm09_channel.json",
    }
    rows = []
    for panel, tool, field, floor_key in CELLS:
        for system in SYSTEMS:
            if system == "S1, English only":
                v = _s1(field, root)
                values = [] if v is None else [v]
            elif (system, field) in single:
                values = [float(_load(single[(system, field)])["pooled"]["eer"])]
            elif system == "SSL-AASIST, final data":
                values = [
                    r[field] for r in ssl if r.get("data") == "final" and r.get(field) is not None
                ]
            else:
                name = "deployed" if system == "Deployed adapter" else "final"
                values = [
                    r[field] for r in runs4 if r.get("system") == name and r.get(field) is not None
                ]
            if not values:
                continue
            rows.append(
                {
                    "panel": panel,
                    "tool": tool,
                    "system": system,
                    "eer": 100 * float(np.mean(values)),
                    "sd": 100 * float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "n": len(values),
                    "floor": 100 * float(floors[floor_key]),
                }
            )
    return pd.DataFrame(rows)


def figure(out_dir: str = FIGURES_DIR, root: str | Path = R) -> str:
    """Write ``final_overview.png`` and ``.pdf``; returns the PNG path."""
    data = table(root)
    plt = _style()
    panels = list(dict.fromkeys(data["panel"]))
    widths = [data[data["panel"] == p]["tool"].nunique() for p in panels]
    fig, axes = plt.subplots(
        1, len(panels), figsize=(7.2, 2.9), sharey=True, gridspec_kw={"width_ratios": widths}
    )
    bar = 0.19
    for ax, panel in zip(axes, panels, strict=True):
        part = data[data["panel"] == panel]
        tools = list(dict.fromkeys(part["tool"]))
        for j, system in enumerate(SYSTEMS):
            sub = part[part["system"] == system].set_index("tool")
            xs = [i + (j - 1.5) * bar for i, t in enumerate(tools) if t in sub.index]
            ys = [sub.loc[t, "eer"] for t in tools if t in sub.index]
            # one model has no spread: no error bar rather than a zero one
            es = [
                sub.loc[t, "sd"] if sub.loc[t, "n"] > 1 else np.nan for t in tools if t in sub.index
            ]
            ax.bar(
                xs,
                ys,
                width=bar * 0.9,
                color=COLOURS[j],
                hatch=HATCHES[j],
                edgecolor="white",
                linewidth=0.6,
                yerr=es,
                error_kw={"elinewidth": 0.8, "capsize": 1.5},
                label=system,
                zorder=2,
            )
        for i, t in enumerate(tools):
            floor = part[part["tool"] == t]["floor"].iloc[0]
            ax.plot(
                [i - 2 * bar, i + 2 * bar],
                [floor, floor],
                color="black",
                lw=1.4,
                zorder=3,
                label="Shortcut floor, final data" if i == 0 else None,
            )
        ax.axhline(50, ls="--", lw=0.8, color="grey", zorder=1)
        ax.set_xticks(range(len(tools)), tools, fontsize=7.5)
        ax.set_title(panel, fontsize=8.5)
        ax.set_ylim(0, 75)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("EER (%)")
    axes[0].text(-0.45, 51.5, "chance", fontsize=7, color="grey")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, 1.06),
    )
    out = Path(out_dir) / "final_overview.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return str(out)


if __name__ == "__main__":
    print(table().to_string(index=False))
    print("wrote", figure())
