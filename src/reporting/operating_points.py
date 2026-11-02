"""Decision D1 part (a): each model's operating point before and after the 20 IndicVoices
dev speakers joined its dev set (P-038), measured on the same held-out sets.

``data/indicvoices/choose_operating_points.sh`` runs the unchanged rule of P-031
(``src.inference.calibrate choose``) on the old dev windows and on the old dev windows plus
the ``iv_dev`` windows; the old run reproduces the tracked threshold file exactly and the new
one is written beside it as ``configs/<name>_spont.yaml``. This module reads those two files
and the stored window scores, and chooses nothing. For every set and both points it reports:

- the live path: each clip replayed through the verdict engine, and the clips of one label
  stitched into 60 s calls (``calibrate.replay_calls``, ``calibrate.stitch_calls``); a real
  call that raises any alert is a false alarm, a fake call that does is a detection;
- the voice check: one decision per file, real when the mean P(genuine) of its scored windows
  is at least the threshold, as ``src.inference.any_audio.judge`` decides.

``dev`` and ``iv_dev`` are where the points were chosen; every other set is held out. The 30
real calls are whole calls, so they are replayed one call at a time and never stitched::

    python -m src.reporting.operating_points --out experiments/results/operating_points_spont_v1.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from live_call.verdict_engine import EngineConfig
from src.inference import calibrate as cal

#: calibration folder -> (tracked threshold file, the 30 real calls' windows over G.711)
MODELS = {
    "final": ("threshold_final", "experiments/results/real_calls_v1/windows_final_g711.csv"),
    "ssl": (
        "threshold_ssl_aasist",
        "experiments/results/real_calls_v1/windows_ssl_aasist_g711.csv",
    ),
    "ssl_s2025": (
        "threshold_ssl_aasist_s2025",
        "experiments/results/real_calls_v1/ssl_s2025/windows_ssl_aasist_g711.csv",
    ),
    "ssl_avg": (
        "threshold_ssl_aasist_avg",
        "experiments/results/real_calls_v1/ssl_avg/windows_ssl_aasist_g711.csv",
    ),
}
#: window sets in each calibration folder, where present; the first two chose the points
SETS = (
    "dev",
    "iv_dev",
    "hiacc",
    "iv_test_all",
    "cm04",
    "cm07",
    "cm09",
    "eval_pool",
    "new_speakers",
    "rvc_test",
)
CHOSEN_ON = ("dev", "iv_dev")
#: genuine spontaneous speech, and the clone sets it is separated from (threshold-free)
SPONTANEOUS = ("hiacc", "iv_test_all")
CLONE_SETS = {
    "iv_test_all": "xtts_v2_same_speakers",
    "cm04": "tortoise",
    "cm07": "indicf5",
    "cm09": "chatterbox",
}


def engine_config(path: str | Path) -> EngineConfig:
    """The verdict engine settings a threshold file holds."""
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return EngineConfig(**{k: values[k] for k in EngineConfig.__dataclass_fields__ if k in values})


def _pct(flags: pd.Series) -> float | None:
    return None if flags.empty else round(100 * float(flags.mean()), 2)


def file_verdicts(windows: pd.DataFrame, threshold: float) -> dict:
    """The voice check's decision per file: real when the mean window P(genuine) >= threshold."""
    per_file = (
        windows.dropna(subset=["score"])
        .groupby("filepath")
        .agg(label=("label", "first"), p=("score", "mean"))
    )
    real = per_file.loc[per_file["label"] == "bonafide", "p"]
    fake = per_file.loc[per_file["label"] == "spoof", "p"]
    return {
        "genuine_files": int(real.size),
        "genuine_called_cloned_pct": _pct(real < threshold),
        "cloned_files": int(fake.size),
        "cloned_called_real_pct": _pct(fake >= threshold),
    }


def measure(windows: pd.DataFrame, config: EngineConfig, stitch: bool = True) -> dict:
    """Live path per clip (and as 60 s calls) and the voice check, at one operating point."""
    out = {"live_per_clip": cal.call_metrics(cal.replay_calls(windows, config))}
    if stitch:  # None when neither label has a minute of windows
        calls = cal.stitch_calls(windows, 60)
        out["live_calls_60s"] = (
            cal.call_metrics(cal.replay_calls(calls, config)) if len(calls) else None
        )
    out["voice_check"] = file_verdicts(windows, config.threshold)
    return out


def _file_means(windows: pd.DataFrame, label: str) -> pd.Series:
    scored = windows.dropna(subset=["score"])
    return scored[scored["label"] == label].groupby("filepath")["score"].mean()


def separation(folder: Path) -> dict:
    """Threshold-free: EER (%) of the voice check's file score, genuine spontaneous files
    (HiACC and the iv_test speakers) against each set's clones. No operating point moves
    this; a high value means no threshold can serve both."""
    from src.training.metrics import compute_eer

    genuine = []
    for s in SPONTANEOUS:
        if (folder / f"{s}_windows.csv").is_file():
            genuine.append(_file_means(pd.read_csv(folder / f"{s}_windows.csv"), "bonafide"))
    if not genuine:
        return {}
    real = pd.concat(genuine).to_numpy(float)
    out = {"genuine_files": int(real.size)}
    for s, tool in CLONE_SETS.items():
        if not (folder / f"{s}_windows.csv").is_file():
            continue
        fake = _file_means(pd.read_csv(folder / f"{s}_windows.csv"), "spoof").to_numpy(float)
        if fake.size:
            scores = np.concatenate([real, fake])
            labels = np.concatenate([np.ones(real.size), np.zeros(fake.size)])
            eer, _ = compute_eer(scores, labels)
            out[tool] = {"clone_files": int(fake.size), "eer_pct": round(100 * float(eer), 2)}
    return out


def compare(model: str, root: str = ".") -> dict:
    """Old and new operating point of one model on every window set it has."""
    base = Path(root)
    name, real_calls = MODELS[model]
    points = {
        "old": engine_config(base / f"configs/{name}.yaml"),
        "new": engine_config(base / f"configs/{name}_spont.yaml"),
    }
    folder = base / f"experiments/calibration_{model}"
    sets = {s: folder / f"{s}_windows.csv" for s in SETS if (folder / f"{s}_windows.csv").is_file()}
    out = {
        "points": {
            k: {f: getattr(c, f) for f in ("threshold", "suspicious_after", "fake_after")}
            for k, c in points.items()
        },
        "chosen_on": [s for s in CHOSEN_ON if s in sets],
        "sets": {},
    }
    for set_name, path in sets.items():
        windows = pd.read_csv(path)
        out["sets"][set_name] = {k: measure(windows, c) for k, c in points.items()}
    if (base / real_calls).is_file():
        windows = pd.read_csv(base / real_calls)
        out["sets"]["real_calls_g711"] = {
            k: measure(windows, c, stitch=False) for k, c in points.items()
        }
    out["spontaneous_separation"] = separation(folder)
    return out


def main() -> None:
    """CLI: compare every model whose new threshold file exists; write one JSON."""
    import argparse

    parser = argparse.ArgumentParser(description="Old and new operating points (decision D1 a)")
    parser.add_argument("--out", default="experiments/results/operating_points_spont_v1.json")
    parser.add_argument("--models", nargs="*", default=list(MODELS))
    args = parser.parse_args()
    report = {
        m: compare(m) for m in args.models if Path(f"configs/{MODELS[m][0]}_spont.yaml").is_file()
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for model, r in report.items():
        old, new = r["points"]["old"]["threshold"], r["points"]["new"]["threshold"]
        print(f"\n{model}: threshold {old:.4f} -> {new:.4f}")
        for set_name, v in r["sets"].items():
            cells = []
            for k in ("old", "new"):
                vc, live = v[k]["voice_check"], v[k].get("live_calls_60s") or v[k]["live_per_clip"]
                cells.append(
                    f"{k}: voice real->cloned {vc['genuine_called_cloned_pct']} "
                    f"cloned->real {vc['cloned_called_real_pct']} | live real alerted "
                    f"{live['real_calls_alerted_pct']} fake alerted {live['fake_calls_alerted_pct']}"
                )
            print(f"  {set_name:16s} " + "  ||  ".join(cells))


if __name__ == "__main__":
    main()
