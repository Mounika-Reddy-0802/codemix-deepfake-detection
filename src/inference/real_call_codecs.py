"""Which part of a real call breaks a detector? The call chain, rebuilt on direct recordings.

The real-call test (``src.inference.real_calls``) scores what the receiving phone recorded:
the speaker's voice after the phone's microphone processing, the mobile network's speech
codec and the call recorder. The direct recordings each speaker made for cloning share the
voice and the microphone but none of the rest, so passing them through the chain one stage
at a time shows which stage moves the score:

- ``direct``: the recording as made, on a phone, with no network;
- ``g711``: the paper's simulated phone line (8 kHz G.711 round trip);
- ``amr_nb``: the real 2G/3G mobile codec, AMR-NB at 12.2 kbit/s (reference encoder);
- ``amr_nb_rec``: AMR-NB, then an 8 kHz AAC encode at 48 kbit/s, the format the receiving
  phone's call recorder wrote for every real call;
- ``amr_wb_rec``: the 4G voice codec, AMR-WB at 12.65 kbit/s, then the same recorder.

Run::

    python -m src.inference.real_call_codecs --references data/real_calls/references \\
        --deployed CKPT --final CKPT --out experiments/results/real_calls_v1/codec_chain.csv
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

VERSIONS = ("direct", "g711", "amr_nb", "amr_nb_rec", "amr_wb_rec")
#: The scripted readings: the same text as the scripted real calls of each speaker.
READINGS = ("read_A", "read_B")
SAMPLE_RATE = 16_000
RECORDER_SR = 8_000
RECORDER_BITRATE = "48k"
AMR_WB_BITRATE = "12.65k"


def _ffmpeg(*args: str) -> None:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError("ffmpeg is needed for the codec chain")
    subprocess.run([exe, "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def recorder(src: str, dst: str) -> None:
    """The call recorder: 8 kHz mono AAC at 48 kbit/s, decoded back to 16 kHz PCM."""
    with tempfile.TemporaryDirectory() as tmp:
        aac = str(Path(tmp) / "recording.m4a")
        _ffmpeg(
            "-i",
            src,
            "-ac",
            "1",
            "-ar",
            str(RECORDER_SR),
            "-c:a",
            "aac",
            "-b:a",
            RECORDER_BITRATE,
            aac,
        )
        _ffmpeg("-i", aac, "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", dst)


def amr_wb(src: str, dst: str) -> None:
    """The 4G voice codec: AMR-WB at 12.65 kbit/s and back to 16 kHz PCM."""
    with tempfile.TemporaryDirectory() as tmp:
        coded = str(Path(tmp) / "clip.3gp")
        _ffmpeg(
            "-i",
            src,
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-c:a",
            "libvo_amrwbenc",
            "-b:a",
            AMR_WB_BITRATE,
            coded,
        )
        _ffmpeg("-i", coded, "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", dst)


def chain_versions(audio: np.ndarray) -> dict[str, np.ndarray]:
    """Every version of one 16 kHz recording, in ``VERSIONS`` order."""
    from src.data.amr_channel import encode_decode
    from src.inference.real_calls import telephone_leg
    from src.utils.audio_utils import load_any, save_wav

    out = {"direct": audio, "g711": telephone_leg(audio)}
    with tempfile.TemporaryDirectory() as tmp:
        direct, nb, nb_rec = (str(Path(tmp) / n) for n in ("direct.wav", "nb.wav", "nb_rec.wav"))
        wb, wb_rec = str(Path(tmp) / "wb.wav"), str(Path(tmp) / "wb_rec.wav")
        save_wav(direct, audio, SAMPLE_RATE)
        encode_decode(direct, nb)
        recorder(nb, nb_rec)
        amr_wb(direct, wb)
        recorder(wb, wb_rec)
        for name, path in (("amr_nb", nb), ("amr_nb_rec", nb_rec), ("amr_wb_rec", wb_rec)):
            out[name] = load_any(path, SAMPLE_RATE)
    return out


def score_chain(recordings: dict[str, np.ndarray], models: dict[str, tuple]) -> pd.DataFrame:
    """Mean window score and fake-leaning share of every version, per model.

    ``models`` maps a name to ``(score_fn, threshold)``; scoring uses the live streaming path.
    """
    from src.inference.calibrate import window_rows

    versions = {name: chain_versions(audio) for name, audio in recordings.items()}
    rows = []
    for model, (score_fn, threshold) in models.items():
        for name, by_version in versions.items():
            for version in VERSIONS:
                w = pd.DataFrame(
                    window_rows(name, "bonafide", "none", by_version[version], score_fn)
                ).dropna(subset=["smoothed"])
                rows.append(
                    {
                        "recording": name,
                        "version": version,
                        "model": model,
                        "mean_p_genuine": round(float(w["score"].mean()), 4),
                        "fake_leaning_pct": round(
                            100 * float((w["smoothed"] < threshold).mean()), 1
                        ),
                        "windows": int(len(w)),
                    }
                )
    return pd.DataFrame(rows)


def load_readings(folder: Path, speakers: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Each speaker's scripted readings at 16 kHz, silence trimmed, keyed ``P2_read_A``."""
    from src.data.real_call_pack import find_recording
    from src.utils.audio_utils import load_any, trim_silence

    out = {}
    for speaker in speakers:
        for reading in READINGS:
            path = find_recording(folder, f"{speaker}_{reading}")
            out[f"{speaker}_{reading}"] = trim_silence(
                load_any(str(path), SAMPLE_RATE), SAMPLE_RATE
            )
    return out


def main() -> None:
    """CLI: score every speaker's readings through each stage of the call chain."""
    import argparse

    from live_call.verdict_engine import load_config
    from src.inference.predict import load_detector, make_score_fn

    parser = argparse.ArgumentParser(description="Score direct recordings through the call chain")
    parser.add_argument("--references", required=True)
    parser.add_argument("--speakers", default="P1,P2,P3", help="speakers whose readings exist")
    parser.add_argument("--deployed", required=True)
    parser.add_argument("--final", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    recordings = load_readings(Path(args.references), tuple(args.speakers.split(",")))
    models = {}
    for name, ckpt, thresholds in (
        ("deployed", args.deployed, "configs/threshold.yaml"),
        ("final", args.final, "configs/threshold_final.yaml"),
    ):
        models[name] = (
            make_score_fn(*load_detector(ckpt, device=args.device)),
            load_config(thresholds).threshold,
        )
    table = score_chain(recordings, models)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    summary = table.pivot_table(index="version", columns="model", values="mean_p_genuine").reindex(
        list(VERSIONS)
    )
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
