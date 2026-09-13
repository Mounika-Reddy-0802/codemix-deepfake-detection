"""Level-normalise an existing portable bundle -- the W5-T4 fix, without re-cutting.

``portable_bundle.build`` produced every bundle this project has scored on, and it
went straight from ``load_wav`` to ``save_wav`` with no level normalisation, though
``preprocess.py`` and ``configs/data/channel_sim.yaml`` both specify -23 dBFS. The
shortcut gate caught it: XTTS arrives peak-normalised where MUCS spans do not, so
level alone separated the classes (``docs/results/lowlevel_cue_check_v1.md``).

``build`` only loads, resamples to 16 kHz and saves, so normalising the clips it
already wrote gives exactly what a fixed rebuild would -- without re-cutting 6,626
clips from sources that are not all on one machine. ``build`` itself is now fixed
too, so any future rebuild normalises from the start.

It also counts how often ``rms_normalize``'s peak guard fires, **per class**. The
guard divides by the peak when scaling would clip, which lands that clip at peak
exactly 1.0. XTTS output sits near full scale already, so if the guard fires far
more often on spoof than on bonafide, normalisation has re-created a peak cue --
one reason it moved the gate only from 1.39% to 5.17%. The count says whether
that is happening.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.paths import resolve

EPS = 1e-8


def trips_peak_guard(audio: np.ndarray, target_dbfs: float = -23.0) -> bool:
    """Whether ``rms_normalize`` would have to fall back to peak-scaling this clip."""
    s = np.asarray(audio, dtype=np.float64)
    if s.size == 0:
        return False
    rms = float(np.sqrt(np.mean(s**2)))
    if rms < EPS:
        return False
    gain = 10.0 ** (target_dbfs / 20.0) / rms
    return float(np.max(np.abs(s))) * gain > 1.0


def normalise_clips(
    manifest: pd.DataFrame, src_root: str, out_root: str, target_dbfs: float = -23.0
) -> dict:
    """Write a normalised copy of every clip the manifest names under ``out_root``.

    Same relative path, so the manifest resolves against either root unchanged.
    Resumable: a clip already written is skipped. Returns per-label counts of
    clips and peak-guard trips.
    """
    from src.utils.audio_utils import load_wav, rms_normalize, save_wav

    stats: dict[str, dict[str, int]] = {}
    seen: set[str] = set()
    for row in manifest.to_dict(orient="records"):
        rel = str(row["filepath"])
        if rel in seen:
            continue
        seen.add(rel)
        label = str(row.get("label", "unknown"))
        bucket = stats.setdefault(label, {"clips": 0, "guard_trips": 0, "written": 0})
        dst = Path(resolve(rel, out_root))
        audio, sr = load_wav(str(resolve(rel, src_root)))
        bucket["clips"] += 1
        if trips_peak_guard(audio, target_dbfs):
            bucket["guard_trips"] += 1
        if not dst.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            save_wav(str(dst), rms_normalize(audio, target_dbfs), sr)
            bucket["written"] += 1
    for bucket in stats.values():
        bucket["guard_trip_rate"] = round(bucket["guard_trips"] / max(bucket["clips"], 1), 4)
    return stats


def main() -> None:
    """CLI: ``python -m src.data.normalise_bundle --manifest M.csv --src-root A --out-root B``."""
    import argparse

    parser = argparse.ArgumentParser(description="Level-normalise an existing portable bundle")
    parser.add_argument("--manifest", action="append", required=True, help="repeatable")
    parser.add_argument("--src-root", required=True, help="root the manifest resolves to now")
    parser.add_argument("--out-root", required=True, help="where the normalised copy goes")
    parser.add_argument("--target-dbfs", type=float, default=-23.0)
    parser.add_argument("--stats-out", default=None, help="write the per-class stats as JSON")
    args = parser.parse_args()

    manifest = pd.concat([pd.read_csv(m) for m in args.manifest], ignore_index=True)
    stats = normalise_clips(manifest, args.src_root, args.out_root, args.target_dbfs)
    print(json.dumps(stats, indent=2))
    if args.stats_out:
        Path(args.stats_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.stats_out).write_text(json.dumps(stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
