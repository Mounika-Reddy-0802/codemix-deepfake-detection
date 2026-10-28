"""Render one genuine corpus call through each stage of a phone call, and score every stage.

The real-network scene lets the audience *hear* what a phone line does to a voice and
see what it does to each detector, without playing anyone from the team: one held-out
corpus speaker's clean recording is passed through the same chain as the team's direct
readings (``src.inference.real_call_codecs``):

    direct -> simulated G.711 -> real AMR-NB -> AMR-NB + call recorder -> AMR-WB + recorder

Every version is scored by the deployed adapter and the final recipe through the live
streaming path (4 s windows every 2 s, -23 dBFS, EMA 0.5), exactly as the codec-chain
results were. The audio and a ``stage_player.json`` index go to ``--out``, a data
directory outside git; the scene-4 page plays them through ``STAGE_PLAYER_DIR``. The scores
alone are also written to ``experiments/results/stage_player_v1.json``, the committed record
the page reads its numbers from.

Run::

    python scripts/prepare_stage_player.py --clean-dir C:/dfdata/processed/clean/mucs2021 \\
        --recording mSiZUO1cgwb8nNpA --speaker 596505 \\
        --deployed C:/dfdata/checkpoints/lora_norm_rvc_channel_best.pt \\
        --final C:/dfdata/checkpoints/final_s1234_best.pt --out C:/dfdata/stage_player
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SR = 16_000
GAP_SECONDS = 0.4
RECORD = Path(__file__).resolve().parents[1] / "experiments" / "results" / "stage_player_v1.json"
LABELS = {
    "direct": "Direct recording",
    "g711": "Simulated G.711 line",
    "amr_nb": "Real AMR-NB codec, 12.2 kbit/s",
    "amr_nb_rec": "AMR-NB, then the call recorder",
    "amr_wb_rec": "AMR-WB (4G), then the recorder",
}


def build_call(clean_dir: Path, recording: str, seconds: float) -> tuple[np.ndarray, list[str]]:
    """The recording's clean segments in order, joined with short pauses, to ``seconds``."""
    from src.utils.audio_utils import load_wav

    segments = sorted(clean_dir.glob(f"{recording}_seg*.wav"))
    if not segments:
        sys.exit(f"no segments {recording}_seg*.wav in {clean_dir}")
    parts, used, total = [], [], 0.0
    for path in segments:
        audio, _ = load_wav(str(path), target_sr=SR)
        parts += [audio, np.zeros(int(GAP_SECONDS * SR), dtype=np.float32)]
        used.append(path.name)
        total += audio.size / SR + GAP_SECONDS
        if total >= seconds:
            break
    return np.concatenate(parts).astype(np.float32), used


def score(audio: np.ndarray, score_fn, threshold: float) -> dict:
    """Mean window P(genuine) and the share of fake-leaning windows, on the live path."""
    from src.inference.streaming import StreamingScorer

    windows = [w for w in StreamingScorer(score_fn).push(audio) if not w.skipped]
    scores = np.array([w.score for w in windows])
    smoothed = np.array([w.smoothed for w in windows])
    return {
        "mean_p_genuine": round(float(scores.mean()), 4),
        "fake_leaning_pct": round(100 * float((smoothed < threshold).mean()), 1),
        "windows": int(len(windows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the scene-4 stage player")
    parser.add_argument("--clean-dir", required=True)
    parser.add_argument(
        "--recording", required=True, help="MUCS recording id of a held-out speaker"
    )
    parser.add_argument("--speaker", required=True)
    parser.add_argument("--seconds", type=float, default=45.0)
    parser.add_argument("--deployed", required=True)
    parser.add_argument("--final", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from live_call.verdict_engine import load_config
    from src.inference.predict import load_detector, make_score_fn
    from src.inference.real_call_codecs import VERSIONS, chain_versions
    from src.inference.streaming import normalise
    from src.utils.audio_utils import save_wav

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    call, used = build_call(Path(args.clean_dir), args.recording, args.seconds)
    versions = chain_versions(call)
    models = {}
    for name, ckpt, file in (
        ("deployed", args.deployed, "configs/threshold.yaml"),
        ("final", args.final, "configs/threshold_final.yaml"),
    ):
        models[name] = (
            make_score_fn(*load_detector(ckpt, device=args.device)),
            load_config(file).threshold,
            file,
        )

    stages = []
    for key in VERSIONS:
        audio = versions[key]
        file = f"{key}.wav"
        # played at one level, so the audience hears the codec, not a volume change
        save_wav(str(out / file), normalise(audio[: int(args.seconds * SR) + SR]), SR)
        entry = {"key": key, "label": LABELS[key], "file": file}
        for name, (score_fn, threshold, _) in models.items():
            entry[name] = score(audio, score_fn, threshold)
        stages.append(entry)
        print(
            f"{key:11s} deployed {entry['deployed']['mean_p_genuine']:.3f}  final {entry['final']['mean_p_genuine']:.3f}"
        )

    index = {
        "about": "One held-out corpus speaker (genuine, never trained on) through each stage of a "
        "phone call, scored on the live path. Not a team member's voice.",
        "speaker": args.speaker,
        "recording": args.recording,
        "segments": used,
        "seconds": round(call.size / SR, 1),
        "thresholds": {name: {"value": t, "file": f} for name, (_, t, f) in models.items()},
        "stages": stages,
    }
    text = json.dumps(index, indent=2) + "\n"
    (out / "stage_player.json").write_text(text, encoding="utf-8")
    # the scores (never the audio) are a result like any other: committed, and the page
    # shows them from this file, not from the data directory
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(text, encoding="utf-8")
    print("wrote", out / "stage_player.json", "and", RECORD)


if __name__ == "__main__":
    main()
