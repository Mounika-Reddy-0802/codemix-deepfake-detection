"""How accurate is each checkpoint on every kind of audio already scored?

Chooses the voice check's detector. Reads stored scores only, so no model is loaded:

* the Kaggle result zips (one deterministic 4 s head crop per clip, as in the paper;
  SSL-AASIST scores are log-odds and are turned into P(genuine) with a sigmoid);
* the live-path window files (``experiments/calibration*``; window 0 is the head crop);
* the 30 real calls and the direct-microphone chain (``data/real_calls/windows``), judged by
  the MEAN P(genuine) over all their windows, as ``src.inference.any_audio`` judges a file.

Each audio item gets one decision at the checkpoint's FROZEN threshold (chosen on
development data only, never on these sets). The SSL-AASIST seed-2025 checkpoint has no
dev-chosen threshold yet, so its column uses 0.5 and is marked provisional.

    python -m src.reporting.any_audio_accuracy --out experiments/results/any_audio_accuracy.md

The input paths are this machine's (the zips and checkpoints are outside git).
"""

from __future__ import annotations

import functools
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path("C:/Users/kuchu/codemix-deepfake-detection")
WT = Path("D:/wt_krishna")
RC = PROJ / "data/real_calls/windows"

THRESHOLDS = {  # frozen operating points (configs/threshold*.yaml), chosen on dev only
    "deployed": 0.49997557675233106,
    "final": 0.6868554928153753,
    "ssl_s1234": 0.5795788615942001,
    # the same checkpoint at the point chosen with the 20 iv_dev speakers in dev (decision D1
    # part a, configs/threshold_ssl_aasist_spont.yaml): the voice check's operating point
    "ssl_s1234_spont": 0.08170506833586838,
    "ssl_s2025": 0.3340942859649658,
}
#: columns that reuse another column's stored scores at their own threshold
SCORES_OF = {"ssl_s1234_spont": "ssl_s1234"}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def zip_scores(zpath: Path, run: str, logodds: bool) -> dict[str, pd.DataFrame]:
    z = zipfile.ZipFile(zpath)
    out = {}
    for name in z.namelist():
        base = name.split("/")[-1]
        if base.startswith(run + "__") and base.endswith("_scores.csv"):
            d = pd.read_csv(io.BytesIO(z.read(name)))
            d["p"] = sigmoid(d["score"]) if logodds else d["score"].astype(float)
            out[base[len(run) + 2 : -len("_scores.csv")]] = d
    return out


def window0(path: Path) -> pd.DataFrame:
    """First 4 s window of each clip == the deterministic head crop."""
    d = pd.read_csv(path)
    d = d[d["window"] == 0].copy()
    d["p"] = d["score"]
    return d


@functools.cache
def sources() -> dict:
    """Every stored score this table reads, loaded on first use (not at import)."""
    return {
        "deployed": {
            "g711": {
                s: window0(WT / f"experiments/calibration/{s}_windows.csv")
                for s in ("new_speakers", "eval_pool", "cm04", "cm07", "cm09", "rvc_test", "hiacc")
            }
        },
        "final": {
            "zip": zip_scores(PROJ / "final_results.zip", "final_s1234", False),
            "g711": {
                s: window0(WT / f"experiments/calibration_final/{s}_windows.csv")
                for s in ("cm09", "hiacc")
            },
        },
        "ssl_s1234": {
            "zip": zip_scores(
                Path("D:/kaggle_realcall/ssl_out/ssl_checkpoint_results.zip"), "final_s1234", True
            ),
            "g711": {
                s: window0(WT / f"experiments/calibration_ssl/{s}_windows.csv") for s in ("hiacc",)
            },
        },
        "ssl_s2025": {
            "zip": zip_scores(
                Path("D:/kaggle_realcall/ssl_out_s2025/ssl_checkpoint_s2025_results.zip"),
                "final_s2025",
                True,
            )
        },
    }


def get(model: str, key: str) -> pd.DataFrame | None:
    src = sources()[SCORES_OF.get(model, model)]
    if key in src.get("zip", {}):
        return src["zip"][key]
    return src.get("g711", {}).get(key)


# (condition label, set key, which rows, expected label)
CONDITIONS = [
    ("Simulated G.711 line", None, None, None),
    ("Genuine, read lectures (60 new speakers)", "new_speakers", ("bonafide", None), "bonafide"),
    ("Genuine, 183 language-control speakers", "control", ("bonafide", None), "bonafide"),
    ("Genuine, SPONTANEOUS (HiACC, 24 speakers)", "hiacc", ("bonafide", None), "bonafide"),
    ("XTTS-v2 clones (seen tool)", "new_speakers", ("spoof", "xtts_v2"), "spoof"),
    ("Tortoise clones (unseen)", "new_speakers", ("spoof", "tortoise"), "spoof"),
    ("Tortoise clones, CM04 (unseen)", "cm04", ("spoof", None), "spoof"),
    ("IndicF5 clones (unseen)", "cm07", ("spoof", None), "spoof"),
    ("Chatterbox clones (unseen)", "cm09", ("spoof", None), "spoof"),
    ("RVC voice conversion", "rvc_test", ("spoof", None), "spoof"),
    ("Real AMR-NB mobile codec", None, None, None),
    (
        "Genuine, read lectures (60 new speakers)",
        "new_speakers_amr",
        ("bonafide", None),
        "bonafide",
    ),
    ("XTTS-v2 clones", "new_speakers_amr", ("spoof", "xtts_v2"), "spoof"),
    ("Tortoise clones", "new_speakers_amr", ("spoof", "tortoise"), "spoof"),
    ("IndicF5 clones", "cm07_amr", ("spoof", None), "spoof"),
]


def rate(model: str, key: str, rows, expected: str) -> str:
    d = get(model, key)
    if d is None:
        return "-"
    lab, tool = rows
    d = d[d["label"].str.lower() == lab]
    if tool:
        d = d[d["tool"] == tool]
    if d.empty:
        return "-"
    thr = THRESHOLDS[model]
    correct = (d["p"] >= thr) if expected == "bonafide" else (d["p"] < thr)
    return f"{100 * correct.mean():.1f} (n={len(d)})"


def real_calls() -> list[tuple[str, dict]]:
    rows = []
    for kind, tool, text in (
        ("genuine_live", "none", "Genuine live call (12 calls)"),
        ("genuine_replay", "none", "Genuine recording replayed into the phone (6)"),
        ("clone_replay", "xtts_v2", "XTTS-v2 clone replayed into the phone (6)"),
        ("clone_replay", "chatterbox", "Chatterbox clone replayed into the phone (6)"),
    ):
        res = {}
        for model, fname in (
            ("deployed", "deployed"),
            ("final", "final"),
            ("ssl_s1234", "ssl_aasist"),
            ("ssl_s1234_spont", "ssl_aasist"),
        ):
            c = pd.read_csv(RC / f"calls_{fname}_recorded.csv")
            c = c[(c["kind"] == kind) & (c["tool"] == tool)]
            thr = THRESHOLDS[model]
            ok = (
                (c["mean_score"] >= thr)
                if c["label"].iloc[0] == "bonafide"
                else (c["mean_score"] < thr)
            )
            res[model] = f"{int(ok.sum())}/{len(c)}"
        rows.append((text, res))
    return rows


def direct_mic() -> tuple[str, dict]:
    cc = pd.read_csv(RC / "codec_chain.csv")
    cc = cc[cc["version"] == "direct"]
    res = {}
    for model in ("deployed", "final"):
        d = cc[cc["model"] == model]
        res[model] = f"{int((d['mean_p_genuine'] >= THRESHOLDS[model]).sum())}/{len(d)}"
    return ("Genuine, read into a phone microphone, no network (6)", res)


def table() -> str:
    models = list(THRESHOLDS)
    head = "| Condition | " + " | ".join(models) + " |"
    lines = [head, "|" + "---|" * (len(models) + 1)]
    for label, key, rows, expected in CONDITIONS:
        if key is None:
            lines.append(f"| **{label}** |" + " |" * len(models))
            continue
        lines.append(
            f"| {label} | " + " | ".join(rate(m, key, rows, expected) for m in models) + " |"
        )
    lines.append(
        "| **Real recordings (as captured on the receiving phone)** |" + " |" * len(models)
    )
    for text, res in real_calls() + [direct_mic()]:
        lines.append(f"| {text} | " + " | ".join(res.get(m, "-") for m in models) + " |")
    lines.append("")
    lines.append(
        "Thresholds: "
        + ", ".join(f"{k} {v:.3f}" for k, v in THRESHOLDS.items())
        + " (each chosen on dev only; ssl_s1234_spont: the dev set includes the 20 iv_dev"
        " spontaneous speakers, decision D1 part a)."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Accuracy per audio condition, stored scores")
    parser.add_argument("--out")
    args = parser.parse_args()
    text = table()
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
