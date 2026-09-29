"""Dataset inventory: clips and hours, real and fake, per corpus and per attack tool.

The datasheet in ``docs/datasheet.md`` is the written record. This module is the
arithmetic behind it: it counts the manifests and *measures* the audio rather than
restating figures, so a number that has drifted since the datasheet was written
shows up as a difference instead of being repeated.

Durations come from two places, in this order:

1. ``data/manifests/clip_index.csv``, which carries ``duration_seconds`` for every
   indexed bonafide code-mixed clip.
2. The audio itself, read header-only through ``soundfile``, for generated clones
   and for ASVspoof. Results are cached, because ASVspoof alone is 122k files.

Audio that is not on this machine -- CM01 lives in a private archive -- is reported
as counted but unmeasured. It is never estimated.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

CLIP_INDEX = "data/manifests/clip_index.csv"
CACHE = "experiments/results/_duration_cache.csv"
OUT_JSON = "experiments/results/dataset_inventory.json"

#: Directories searched for generated and bundled code-mixed audio, keyed by stem.
AUDIO_DIRS: tuple[str, ...] = (
    "lora_bundle_norm_ch20/clips",
    "lora_bundle_norm/clips",
    "lora_bundle/clips",
    "colab_bundle/clips",
    "cm04",
    "rvc_cm02",
    "generated/xtts_v2",
    # gap_codemix_clean references whole MUCS source recordings, not segmented clips
    "raw/mucs2021/train",
    "raw/mucs2021/test",
)

ASVSPOOF_SPLITS: dict[str, str] = {
    "train": "raw/asvspoof2019_LA/LA/ASVspoof2019_LA_train/flac",
    "dev": "raw/asvspoof2019_LA/LA/ASVspoof2019_LA_dev/flac",
    "eval": "raw/asvspoof2019_LA/LA/ASVspoof2019_LA_eval/flac",
}

#: Manifests reported individually, in the order they appear in the datasheet.
MANIFESTS: tuple[str, ...] = (
    "codemix_adapt_train",
    "codemix_adapt_train_rvc",
    "codemix_adapt_dev",
    "codemix_eval",
    "score_cm02",
    "score_cm04_norm",
    "rvc_holdout_train",
    "rvc_holdout_test",
    "gap_codemix_clean",
    "asvspoof_train",
    "asvspoof_dev",
    "asvspoof_eval",
)

#: Generated attack pools as they exist on disk, independent of which clips ended
#: up in a manifest. "Generated" and "used in an experiment" are different numbers
#: and the datasheet reports both.
ATTACK_POOLS: dict[str, tuple[str, str]] = {
    "CM01/CM03 XTTS-v2 outputs": ("generated/xtts_v2", ""),
    "CM02 RVC v2, train pool": ("rvc_cm02", ""),
    "CM04 Tortoise-TTS, held out": ("cm04", ""),
}

#: How a manifest's ``tool`` column maps onto the attack IDs in the datasheet.
TOOL_LABELS: dict[str, str] = {
    "none": "bonafide (no tool)",
    "xtts_v2": "XTTS-v2 (CM01 / CM03)",
    "xtts": "XTTS-v2 (CM01 / CM03)",
    "rvc": "RVC v2 (CM02)",
    "tortoise": "Tortoise-TTS (CM04, held out)",
}


@dataclass
class Tally:
    """Clips and seconds, with the clips whose audio could not be measured."""

    clips: int = 0
    seconds: float = 0.0
    unmeasured: int = 0
    speakers: set[str] = field(default_factory=set)

    def add(self, seconds: float | None, speaker: str = "") -> None:
        self.clips += 1
        if seconds is None:
            self.unmeasured += 1
        else:
            self.seconds += seconds
        if speaker:
            self.speakers.add(speaker)

    def as_dict(self) -> dict:
        return {
            "clips": self.clips,
            "hours": round(self.seconds / 3600.0, 2),
            "speakers": len(self.speakers),
            "unmeasured_clips": self.unmeasured,
        }


def _stem(path: str) -> str:
    return Path(path.replace("\\", "/")).stem


def load_cache(path: str | Path = CACHE) -> dict[str, float]:
    file = Path(path)
    if not file.is_file():
        return {}
    with open(file, encoding="utf-8", newline="") as handle:
        return {row["stem"]: float(row["seconds"]) for row in csv.DictReader(handle)}


def save_cache(durations: dict[str, float], path: str | Path = CACHE) -> None:
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    with open(file, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stem", "seconds"])
        writer.writerows(sorted(durations.items()))


def measure(paths: list[Path], durations: dict[str, float], progress: bool = False) -> int:
    """Read header durations for anything not already cached. Returns how many."""
    import soundfile as sf

    added = 0
    for index, path in enumerate(paths):
        stem = path.stem
        if stem in durations:
            continue
        try:
            durations[stem] = float(sf.info(str(path)).duration)
            added += 1
        except Exception:  # noqa: BLE001 - an unreadable file is reported, not fatal
            continue
        if progress and added % 5000 == 0:
            print(f"  measured {added} ({index + 1}/{len(paths)})", file=sys.stderr)
    return added


def index_durations(
    data_root: str | Path, root: str | Path = ".", progress: bool = False
) -> dict[str, float]:
    """Stem -> seconds, from the clip index first and then from the audio itself."""
    durations = load_cache(Path(root) / CACHE)

    index = Path(root) / CLIP_INDEX
    if index.is_file():
        with open(index, encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                raw = (row.get("duration_seconds") or "").strip()
                if raw and row["utt_id"] not in durations:
                    durations[row["utt_id"]] = float(raw)

    base = Path(data_root)
    targets: list[Path] = []
    for folder in AUDIO_DIRS:
        directory = base / folder
        if directory.is_dir():
            targets += [p for p in directory.rglob("*") if p.suffix.lower() in {".wav", ".flac"}]
    for folder in ASVSPOOF_SPLITS.values():
        directory = base / folder
        if directory.is_dir():
            targets += sorted(directory.glob("*.flac"))

    if targets:
        added = measure(targets, durations, progress=progress)
        if added:
            save_cache(durations, Path(root) / CACHE)
    return durations


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarise(root: str | Path = ".", data_root: str | Path | None = None, progress: bool = False):
    """The whole inventory: per corpus, per tool, per manifest."""
    import os

    base = Path(root)
    data = Path(data_root or os.environ.get("DATA_ROOT", "."))
    durations = index_durations(data, root=base, progress=progress)

    corpora: dict[str, Tally] = defaultdict(Tally)
    tools: dict[str, Tally] = defaultdict(Tally)
    asvspoof: dict[str, Tally] = defaultdict(Tally)
    manifests: dict[str, dict] = {}

    # bonafide code-mixed corpora, straight from the clip index
    index = base / CLIP_INDEX
    if index.is_file():
        for row in read_manifest(index):
            raw = (row.get("duration_seconds") or "").strip()
            corpora[row["source"]].add(float(raw) if raw else None, row.get("speaker", ""))

    for name in MANIFESTS:
        path = base / "data" / "manifests" / f"{name}.csv"
        if not path.is_file():
            continue
        real, fake = Tally(), Tally()
        by_tool: dict[str, Tally] = defaultdict(Tally)
        for row in read_manifest(path):
            stem = row.get("utt_id") or _stem(row["filepath"])
            seconds = durations.get(stem)
            speaker = row.get("speaker", "")
            bonafide = row["label"].strip().lower() == "bonafide"
            (real if bonafide else fake).add(seconds, speaker)
            tool = (row.get("tool") or "none").strip().lower()
            by_tool[tool].add(seconds, speaker)
            if name.startswith("asvspoof"):
                # A01..A19 are ASVspoof's own attacks; they are English and belong in
                # their own section, not beside the clones this project generated.
                asvspoof[f"{name.split('_')[1]} {'bonafide' if bonafide else 'spoof'}"].add(
                    seconds, speaker
                )
            elif not bonafide:
                tools[tool].add(seconds, speaker)
        manifests[name] = {
            "clips": real.clips + fake.clips,
            "bonafide": real.as_dict(),
            "spoof": fake.as_dict(),
            "spoof_per_real": round(fake.clips / real.clips, 2) if real.clips else None,
            "by_tool": {TOOL_LABELS.get(k, k): v.as_dict() for k, v in sorted(by_tool.items())},
        }

    pools: dict[str, dict] = {}
    for label, (folder, caveat) in ATTACK_POOLS.items():
        directory = data / folder
        tally = Tally()
        if directory.is_dir():
            for path in sorted(directory.rglob("*")):
                if path.suffix.lower() in {".wav", ".flac"}:
                    tally.add(durations.get(path.stem))
        pools[label] = {**tally.as_dict(), "on_disk": directory.is_dir(), "note": caveat}

    return {
        "note": (
            "hours are measured from audio headers or the clip index; clips whose audio "
            "is not on this machine are counted under unmeasured_clips and never estimated"
        ),
        "data_root": str(data),
        "corpora": {k: v.as_dict() for k, v in sorted(corpora.items())},
        "attack_tools": {TOOL_LABELS.get(k, k): v.as_dict() for k, v in sorted(tools.items())},
        "generated_pools": pools,
        "asvspoof_splits": {k: v.as_dict() for k, v in sorted(asvspoof.items())},
        "manifests": manifests,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Count clips and hours per dataset")
    parser.add_argument("--out", default=OUT_JSON)
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()

    report = summarise(data_root=args.data_root, progress=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["corpora"], indent=2))
    print(json.dumps(report["attack_tools"], indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
