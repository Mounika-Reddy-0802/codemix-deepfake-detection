"""Build the demo call arsenal from held-out evaluation audio, and verify each one (W8-T5).

The evaluation clips last 7-15 s, too short for the verdict ladder to show anything a
viewer can follow. A demo call joins several clips of **one held-out speaker** (0.4 s
pauses between them) into a call of about 45 s, in the phone-line condition the
deployed model runs in:

| id | What it is | Source set |
|---|---|---|
| ``genuine`` | a real code-mixed speaker | eval pool, bonafide |
| ``xtts_clone`` | that family of attack: XTTS-v2 clone of an eval speaker | eval pool, spoof |
| ``rvc_conversion`` | RVC voice conversion, speaker never trained on | RVC test half, spoof |
| ``rvc_genuine`` | the real recordings those conversions were made from | RVC test half, bonafide |
| ``tortoise_clone`` | a cloning tool the model never saw | CM04, spoof |

Every call is then replayed through the exact live path (``StreamingScorer`` plus
``VerdictEngine`` at the frozen threshold) and the **observed** peak verdict is written
beside the expected one. A call the model gets wrong is kept and labelled, not
swapped for one it gets right: the Tortoise call is there to show a known limit.

Audio is written outside git, to ``--out`` (a data directory, never the repo), with a
``demo_clips.json`` index the server reads through ``DEMO_CLIPS_DIR``.

Run::

    python scripts/prepare_demo_clips.py --checkpoint <data>/checkpoints/lora_norm_rvc_channel_best.pt \\
        --data-root <data>/lora_bundle_norm_ch20 --out <data>/demo_clips
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd

from live_call.replay import FileSource, replay
from live_call.verdict_engine import State, load_config
from src.utils.audio_utils import load_wav, save_wav
from src.utils.paths import resolve

SR = 16_000
GAP_SECONDS = 0.4

SPECS = [
    {
        "id": "genuine",
        "manifest": "codemix_eval_channel20",
        "label": "bonafide",
        "tool": None,
        "title": "Genuine caller",
        "description": "a real Hinglish speaker, never trained on",
    },
    {
        "id": "xtts_clone",
        "manifest": "codemix_eval_channel20",
        "label": "spoof",
        "tool": "xtts_v2",
        "title": "XTTS-v2 cloned voice",
        "description": "voice clone of a held-out speaker",
    },
    {
        "id": "rvc_conversion",
        "manifest": "rvc_holdout_test_channel20",
        "label": "spoof",
        "tool": "rvc",
        "title": "RVC voice conversion",
        "description": "real speech converted to another voice, unseen speakers",
    },
    {
        "id": "rvc_genuine",
        "manifest": "rvc_holdout_test_channel20",
        "label": "bonafide",
        "tool": None,
        "title": "Genuine caller (RVC source)",
        "description": "real recordings the RVC conversions start from",
    },
    {
        "id": "tortoise_clone",
        "manifest": "score_cm04_norm_channel20",
        "label": "spoof",
        "tool": "tortoise",
        "title": "Tortoise cloned voice (unseen tool)",
        "description": "a cloning tool the model never saw: a known limit",
    },
]


def pick_speaker_clips(frame: pd.DataFrame, target_seconds: float, data_root: str, seed: int = 0):
    """Clips of the one speaker with the most material, until ``target_seconds`` is reached."""
    by_speaker = frame.groupby("speaker")["filepath"].count().sort_values(ascending=False)
    speaker = str(by_speaker.index[0])
    rows = frame[frame["speaker"].astype(str) == speaker].sample(frac=1.0, random_state=seed)
    audio, used, total = [], [], 0.0
    for fp in rows["filepath"]:
        clip, _ = load_wav(resolve(fp, data_root), target_sr=SR)
        audio += [clip, np.zeros(int(GAP_SECONDS * SR), dtype=np.float32)]
        used.append(fp)
        total += clip.size / SR + GAP_SECONDS
        if total >= target_seconds:
            break
    return speaker, np.concatenate(audio).astype(np.float32), used


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and verify the demo calls")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True, help="data directory for the demo audio")
    parser.add_argument("--seconds", type=float, default=45.0)
    parser.add_argument("--threshold-file", default="configs/threshold.yaml")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from src.inference.predict import load_detector, make_score_fn

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = load_config(args.threshold_file)
    model, device = load_detector(args.checkpoint, device=args.device)
    score_fn = make_score_fn(model, device)

    index = []
    for spec in SPECS:
        frame = pd.read_csv(f"data/manifests/{spec['manifest']}.csv")
        frame = frame[frame["label"] == spec["label"]]
        if spec["tool"]:
            frame = frame[frame["tool"].astype(str).str.lower() == spec["tool"]]
        speaker, audio, used = pick_speaker_clips(frame, args.seconds, args.data_root)
        name = f"{spec['id']}.wav"
        save_wav(str(out / name), audio, SR)

        result = asyncio.run(replay(FileSource(audio), score_fn, config))
        summary = result.engine.summary
        expected = "alert" if spec["label"] == "spoof" else "no alert"
        alerted = summary.peak_state in (State.SUSPICIOUS, State.LIKELY_FAKE)
        observed = "alert" if alerted else "no alert"
        entry = {
            "id": spec["id"],
            "file": name,
            "title": spec["title"],
            "description": spec["description"],
            "label": spec["label"],
            "tool": spec["tool"] or "none",
            "speaker": speaker,
            "seconds": round(audio.size / SR, 1),
            "source_clips": len(used),
            "source": used,
            "expected": expected,
            "observed": observed,
            "peak_state": summary.peak_state.value,
            "fake_leaning_windows": summary.fake_leaning_windows,
            "scored_windows": summary.scored_windows,
            "correct": expected == observed,
        }
        index.append(entry)
        print(
            f"{spec['id']:16s} {entry['seconds']:5.1f}s  expected {expected:8s} observed {observed:8s} "
            f"peak {entry['peak_state']:11s} fake-leaning {summary.fake_leaning_windows}/{summary.scored_windows}"
        )

    (out / "demo_clips.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    report = [{k: v for k, v in e.items() if k != "source"} for e in index]
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path("experiments/results/demo_clips_verification.json").write_text(
        json.dumps(
            {
                "checkpoint": Path(args.checkpoint).name,
                "threshold": config.threshold,
                "calls": report,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out / 'demo_clips.json'}")


if __name__ == "__main__":
    main()
