"""A real mobile codec: render a manifest through AMR-NB with ffmpeg.

The phone-line condition used throughout the paper (``src.data.channel_bundle``) is a
simulated narrowband chain: 8 kHz, G.711 mu-law, noise at 20 dB. G.711 is the
fixed-line codec and is nearly transparent at 64 kbit/s; mobile calls in India go
through AMR-NB, a speech codec that models the voice (ACELP) at 4.75-12.2 kbit/s and
discards what its model does not keep. A detector that learned the G.711 chain may
not survive that, so this module renders the test sets through the real codec:

    16 kHz clip -> 8 kHz -> AMR-NB encode (libopencore_amrnb) -> decode -> 16 kHz

12.2 kbit/s (mode MR122, the GSM-EFR rate) is the default; 4.75 kbit/s is the
codec's harshest mode. No noise is added, so the condition isolates the codec. The
encoder is the reference 3GPP implementation shipped with ffmpeg, so the audio is what
a handset would actually transmit, not a simulation of it.

Rendering is resumable (an existing output is kept) and the manifest's ``condition``
column records the codec and rate.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from src.data.portable_bundle import DATA_ROOT_TOKEN

BITRATES = ("4.75k", "5.15k", "5.9k", "6.7k", "7.4k", "7.95k", "10.2k", "12.2k")
DEFAULT_BITRATE = "12.2k"


def condition_name(bitrate: str) -> str:
    return "amr_nb_" + bitrate.replace(".", "k").rstrip("k")


def ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError("ffmpeg is not on PATH")
    return exe


def encode_decode(src: str, dst: str, bitrate: str = DEFAULT_BITRATE) -> None:
    """One clip through AMR-NB and back to 16 kHz mono PCM."""
    if bitrate not in BITRATES:
        raise ValueError(f"AMR-NB has no {bitrate} mode; choose one of {BITRATES}")
    with tempfile.TemporaryDirectory() as tmp:
        coded = str(Path(tmp) / "clip.amr")
        common = [ffmpeg(), "-hide_banner", "-loglevel", "error", "-y"]
        subprocess.run(
            [
                *common,
                "-i",
                src,
                "-ac",
                "1",
                "-ar",
                "8000",
                "-c:a",
                "libopencore_amrnb",
                "-b:a",
                bitrate,
                coded,
            ],
            check=True,
        )
        subprocess.run(
            [*common, "-i", coded, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", dst], check=True
        )


def render(
    manifest: pd.DataFrame, out_dir: str, data_root: str, bitrate: str = DEFAULT_BITRATE
) -> pd.DataFrame:
    """Every clip of ``manifest`` through AMR-NB into ``out_dir/clips``; new manifest."""
    clips = Path(out_dir) / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    out = manifest.copy()
    names = []
    for i, path in enumerate(manifest["filepath"]):
        name = Path(str(path)).name
        target = clips / name
        if not target.exists():
            source = str(path).replace(DATA_ROOT_TOKEN, data_root)
            partial = target.with_suffix(".part.wav")
            encode_decode(source, str(partial), bitrate)
            partial.replace(target)
        names.append(f"{DATA_ROOT_TOKEN}/clips/{name}")
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(manifest)} clips", flush=True)
    out["filepath"] = names
    out["condition"] = condition_name(bitrate)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a manifest through AMR-NB")
    parser.add_argument("--manifest", required=True, help="clean manifest to render")
    parser.add_argument("--data-root", required=True, help="where its clips are")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--bitrate", default=DEFAULT_BITRATE, choices=BITRATES)
    args = parser.parse_args()
    df = pd.read_csv(args.manifest, dtype={"speaker": str})
    rendered = render(df, args.out_dir, args.data_root, args.bitrate)
    rendered.to_csv(args.manifest_out, index=False)
    print(f"{args.manifest_out}: {len(rendered)} clips, condition {rendered['condition'].iloc[0]}")


if __name__ == "__main__":
    main()
