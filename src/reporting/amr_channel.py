"""Clean, simulated G.711 and real AMR-NB: does the phone-line result survive a mobile codec?

For three models (S1, S2 trained clean, the deployed S2) and each test set, the EER of
every attack on clean audio, over the simulated G.711 chain the paper reports, and
over the AMR-NB codec of a mobile handset (``src.data.amr_channel``), each with a
speaker-bootstrap interval (``src.reporting.new_speakers.attack_eer``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.reporting.new_speakers import attack_eer

MODELS = {
    "s1": "S1 (English only)",
    "s2_rvc_clean": "S2 trained clean",
    "deployed_s2": "Deployed S2 (trained on G.711)",
}
#: column -> (manifest stem, tool, where the clean/G.711 scores live)
ATTACKS = {
    "new XTTS (CM05)": ("score_ext", "xtts_v2", "ext"),
    "new Tortoise (CM06)": ("score_ext", "tortoise", "ext"),
    "IndicF5 (CM07)": ("score_cm07", "indicf5", "ext"),
}
CONDITIONS = ("clean", "G.711", "AMR-NB")


def score_path(model: str, stem: str, condition: str, ext_dir: str, amr_dir: str) -> Path:
    if condition == "AMR-NB":
        return Path(amr_dir) / f"{model}__{stem}_amr_scores.csv"
    cond = "clean" if condition == "clean" else "channel"
    return Path(ext_dir) / f"{model}__{stem}_{cond}_scores.csv"


def analyse(ext_dir: str, amr_dir: str, n_boot: int = 1000) -> list[dict]:
    rows = []
    for model, name in MODELS.items():
        for condition in CONDITIONS:
            row = {"model": name, "tag": model, "condition": condition}
            for column, (stem, tool, _) in ATTACKS.items():
                path = score_path(model, stem, condition, ext_dir, amr_dir)
                if path.is_file():
                    scores = pd.read_csv(path)
                    if (scores["tool"] == tool).any():  # an attack with no clips is skipped
                        row[column] = attack_eer(scores, tool, n_boot)
            if len(row) > 3:
                rows.append(row)
    return rows


def markdown(rows: list[dict]) -> str:
    lines = [
        "# Clean, simulated G.711 and real AMR-NB (v1)",
        "",
        "EER (%) with a 95% speaker-bootstrap interval on the week-13 test sets. G.711 is the",
        "simulated phone line used throughout the paper (8 kHz, mu-law, 20 dB noise); AMR-NB",
        "is the reference 3GPP mobile codec at 12.2 kbit/s, no added noise. Built by",
        "`src/reporting/amr_channel.py`.",
        "",
        "| Model | Audio | " + " | ".join(ATTACKS) + " |",
        "|---|---|" + "---|" * len(ATTACKS),
    ]
    for row in rows:
        cells = []
        for column in ATTACKS:
            r = row.get(column)
            cells.append(
                "-"
                if r is None
                else f"{100 * r['eer']:.1f} [{100 * r['ci_low']:.1f}, {100 * r['ci_high']:.1f}]"
            )
        lines.append(f"| {row['model']} | {row['condition']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean vs G.711 vs AMR-NB")
    parser.add_argument("--ext", default="experiments/results/ext_scoring_v1")
    parser.add_argument("--amr", default="experiments/results/amr_v1")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--out", default="experiments/results/amr_v1/summary.json")
    parser.add_argument("--doc", default="docs/results/amr_channel_v1.md")
    args = parser.parse_args()
    rows = analyse(args.ext, args.amr, args.n_boot)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))
    Path(args.doc).write_text(markdown(rows), encoding="utf-8")
    print(markdown(rows))


if __name__ == "__main__":
    main()
