"""Week-13 scoring sets: the genuine clips of the test speakers plus the new attacks.

A generation run on Kaggle returns a zip whose ``gen_bundle/`` holds ``clean/clips``,
``channel/clips`` and one manifest per attack (``cm05.csv``, ``cm05_channel20.csv``,
...). ``unpack`` copies the clips into the extension bundle beside the genuine clips;
every clip name starts with its attack id, so nothing collides.

``score_set`` stacks attacks under the genuine set of the *same* speakers. The
evaluator scores each tool against that shared genuine set, so one manifest per
genuine pool is enough:

* ``score_ext``: the 60 new test speakers (``eval_ext_genuine``) + CM05 (XTTS) + CM06
  (Tortoise);
* ``score_cm07``: the IndicF5 speakers' all-Hindi genuine clips + CM07 (IndicF5).

A spoof speaker missing from the genuine set is an error, not a warning: the EER
would then compare different people. CM08 is training data and never enters a
scoring set.
"""

from __future__ import annotations

import argparse
import io
import shutil
import zipfile
from pathlib import Path

import pandas as pd

TRAINING_ATTACKS = {"cm08"}
SETS = {
    "score_ext": ("eval_ext_genuine", ("cm05", "cm06")),
    "score_cm07": ("indicf5_genuine", ("cm07",)),
}
CONDITIONS = {"": "clean", "_channel20": "channel"}


def unpack(zip_path: str, clean_root: str, channel_root: str) -> dict[str, int]:
    """Copy the zip's clean and channel clips into the two bundle roots."""
    counts = {"clean": 0, "channel": 0}
    roots = {"clean": Path(clean_root), "channel": Path(channel_root)}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            parts = name.split("/")
            if len(parts) != 4 or parts[0] != "gen_bundle" or parts[2] != "clips":
                continue
            kind = parts[1]
            if kind not in roots or not parts[3]:
                continue
            target = roots[kind] / "clips" / parts[3]
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                with zf.open(name) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            counts[kind] += 1
    return counts


def attack_manifests(zip_path: str) -> dict[str, pd.DataFrame]:
    """Every manifest in the zip, keyed by file stem (``cm05``, ``cm05_channel20``)."""
    out = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.startswith("gen_bundle/manifests/") and name.endswith(".csv"):
                out[Path(name).stem] = pd.read_csv(
                    io.BytesIO(zf.read(name)), dtype={"speaker": str}
                )
    return out


def score_set(genuine: pd.DataFrame, attacks: list[pd.DataFrame]) -> pd.DataFrame:
    """Genuine rows first, then each attack; refuses speakers the genuine set lacks."""
    if set(genuine["label"]) != {"bonafide"}:
        raise ValueError("the genuine set holds non-bonafide rows")
    speakers = set(genuine["speaker"].astype(str))
    for attack in attacks:
        if set(attack["label"]) != {"spoof"}:
            raise ValueError("an attack manifest holds non-spoof rows")
        sources = set(attack["source"])
        if sources & TRAINING_ATTACKS:
            raise ValueError(
                f"training attack {sorted(sources & TRAINING_ATTACKS)} in a scoring set"
            )
        missing = set(attack["speaker"].astype(str)) - speakers
        if missing:
            raise ValueError(f"{sorted(sources)}: {len(missing)} speaker(s) have no genuine clips")
    scored = pd.concat([genuine, *attacks], ignore_index=True)
    if scored["filepath"].duplicated().any():
        raise ValueError("a clip appears twice")
    return scored


def build(manifests_dir: str, attacks: dict[str, pd.DataFrame]) -> dict[str, int]:
    """Write every attack manifest, then every scoring set whose attacks are all present."""
    mdir = Path(manifests_dir)
    for stem, frame in attacks.items():
        frame.to_csv(mdir / f"{stem}.csv", index=False)
    written = {}
    for name, (genuine_stem, attack_ids) in SETS.items():
        for suffix in CONDITIONS:
            stems = [f"{a}{suffix}" for a in attack_ids]
            if not all((mdir / f"{s}.csv").is_file() for s in stems):
                continue
            genuine = pd.read_csv(mdir / f"{genuine_stem}{suffix}.csv", dtype={"speaker": str})
            parts = [pd.read_csv(mdir / f"{s}.csv", dtype={"speaker": str}) for s in stems]
            scored = score_set(genuine, parts)
            scored.to_csv(mdir / f"{name}{suffix}.csv", index=False)
            written[f"{name}{suffix}"] = len(scored)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unpack generation zips and build the scoring sets"
    )
    parser.add_argument("--zips", nargs="+", required=True)
    parser.add_argument("--clean-root", required=True)
    parser.add_argument("--channel-root", required=True)
    parser.add_argument("--manifests", default="data/manifests")
    args = parser.parse_args()

    attacks: dict[str, pd.DataFrame] = {}
    for zip_path in args.zips:
        print(zip_path, unpack(zip_path, args.clean_root, args.channel_root))
        attacks.update(attack_manifests(zip_path))
    for name, rows in build(args.manifests, attacks).items():
        print(f"{name}: {rows} clips")


if __name__ == "__main__":
    main()
