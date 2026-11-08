"""Voice check v2 on clean wideband files, beside the detector before it (reported, not judged).

Both runs are ``python -m src.inference.any_audio_eval score`` on the clean copies of the dev and
test clips, through the page's own path with the telephone front end: the detector before
8 November at 0.082 (``experiments/any_audio/clean_scored.csv``) and voice check v2 at 0.073
(``clean_scored_vcv2.csv``). They are compared on the recordings both scored, and the 31
``gen_train`` speakers, whose clones and genuine clips v2 trained on, are left out of both. The
clean list has no speaker column, so each file's speaker is read from the project's manifests
by file name; a file found under two speakers stops the table.

A recording too short to judge ("no speech") counts as not judged correctly, as in
``any_audio_eval.report``; the share of such recordings is shown separately.

    python -m src.reporting.voice_check_v2_clean --out experiments/results/voice_check_v2_clean.md
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import generator_study as gs
from src.inference.any_audio_eval import _correct, _eer

BEFORE = "experiments/any_audio/clean_scored.csv"
V2 = "experiments/any_audio/clean_scored_vcv2.csv"
MODELS = {"before 8 Nov (0.082)": BEFORE, "v2 (0.073)": V2}
#: summary rows: (label, sets, kind)
SUMMARY = (
    ("Genuine, read and HiACC spontaneous", None, "genuine"),
    ("XTTS-v2 clones", None, "xtts_v2"),
    ("Tortoise clones (in no training set)", None, "tortoise"),
    ("RVC conversions (held-out speakers)", None, "rvc"),
    ("IndicF5 clones (held-out speakers)", None, "indicf5"),
    ("Chatterbox clones (held-out speakers)", None, "chatterbox"),
)


def speaker_map(pattern: str = "data/manifests/*.csv") -> dict[str, str]:
    """File name -> speaker, from every manifest with both columns."""
    out: dict[str, str] = {}
    for path in sorted(glob.glob(pattern)):
        try:
            d = pd.read_csv(path, usecols=lambda c: c in ("filepath", "speaker"), dtype=str)
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError):
            continue
        if {"filepath", "speaker"} - set(d.columns):
            continue
        for name, spk in zip(d["filepath"].str.split("/").str[-1], d["speaker"], strict=True):
            if out.setdefault(name, spk) != spk:
                raise ValueError(f"{name} belongs to two speakers ({out[name]}, {spk})")
    return out


def load(path: str, speakers: dict[str, str], exclude: set[str]) -> pd.DataFrame:
    """One run's G.711 rows, keyed by (file name, set), without the excluded speakers."""
    d = pd.read_csv(path)
    d = d[(d["front_end"] == "g711") & ~d["verdict"].astype(str).str.startswith("error")].copy()
    d["name"] = d["filepath"].str.split("/").str[-1]
    d["speaker"] = d["name"].map(speakers)
    if d["speaker"].isna().any():
        raise ValueError(f"{path}: {int(d['speaker'].isna().sum())} files have no known speaker")
    d["kind"] = np.where(d["label"].str.lower() == "bonafide", "genuine", d["tool"])
    return d[~d["speaker"].isin(exclude)].drop_duplicates(["name", "set"])


def paired(runs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Each run restricted to the (file, set) rows every run scored."""
    keys = None
    for d in runs.values():
        k = set(zip(d["name"], d["set"], strict=True))
        keys = k if keys is None else keys & k
    return {
        m: d[[(n, s) in keys for n, s in zip(d["name"], d["set"], strict=True)]]
        for m, d in runs.items()
    }


def cell(d: pd.DataFrame) -> str:
    """Percent judged correctly, the share too short to judge, and genuine ones called cloned."""
    if d.empty:
        return "-"
    short = (d["verdict"] == "no speech").mean()
    text = f"{100 * _correct(d).mean():.1f} % of {len(d)}"
    notes = [f"{100 * short:.0f} % too short"] if short else []
    if (d["kind"] == "genuine").all():
        notes.append(f"{int((d['verdict'] == 'cloned').sum())} called cloned")
    return text + (f" ({'; '.join(notes)})" if notes else "")


def table(runs: dict[str, pd.DataFrame], excluded: int) -> str:
    models = list(runs)
    test = {m: d[d["split"] != "dev"] for m, d in runs.items()}
    lines = [
        "# Voice check v2 on clean wideband files",
        "",
        f"Telephone front end, each detector at its frozen point; the {excluded} `gen_train` "
        "speakers left out of both; only recordings both runs scored. Percent judged correctly; "
        "recordings with under 2 s of speech count as not judged correctly.",
        "",
        "| Clean test recordings | " + " | ".join(models) + " |",
        "|---|" + "---|" * len(models),
    ]
    for label, _, kind in SUMMARY:
        lines.append(
            f"| {label} | "
            + " | ".join(cell(test[m][test[m]["kind"] == kind]) for m in models)
            + " |"
        )
    lines += ["", "Per set (dev rows are the development clips):", ""]
    lines += ["| Set | Kind | " + " | ".join(models) + " |", "|---|---|" + "---|" * len(models)]
    first = runs[models[0]]
    for (name, kind), _ in first.groupby(["set", "kind"], sort=False):
        cells = [
            cell(runs[m][(runs[m]["set"] == name) & (runs[m]["kind"] == kind)]) for m in models
        ]
        lines.append(f"| {name} | {kind} | " + " | ".join(cells) + " |")
    lines += ["", "Equal error rate per set (threshold-free), %:", ""]
    lines += ["| Set | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
    for name in first["set"].unique():
        cells = []
        for m in models:
            e = _eer(runs[m][runs[m]["set"] == name])
            cells.append("-" if e is None else f"{e:.2f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    """CLI: print the table and optionally write it."""
    import argparse

    parser = argparse.ArgumentParser(description="Voice check v2 on clean files")
    parser.add_argument("--pools", default=f"data/manifests/{gs.POOLS_OUT}")
    parser.add_argument("--out")
    args = parser.parse_args()
    exclude = gs.train_speakers(args.pools)
    speakers = speaker_map()
    runs = paired({m: load(p, speakers, exclude) for m, p in MODELS.items()})
    text = table(runs, len(exclude))
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
