"""What the telephone channel does to speech, as spectrograms.

One genuine clip and one XTTS-v2 clone of the same speaker, each shown three ways: clean
(16 kHz), over the simulated G.711 line, and through the real AMR-NB codec. Both channels
cut everything above 4 kHz from genuine and cloned speech alike, which is why a detector
trained on wideband English studio audio meets a different recording domain on a phone.

Clips are looked up by file name in the clean, G.711 and AMR-NB folders of the project's
bundles. Without names, the first speaker of the new-speaker pool that has both a genuine
clip and an XTTS-v2 clone in all three versions is used. Speaker IDs are not printed.

    python -m src.reporting.channel_spectrogram --out Paper_Journal/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SR = 16_000
SECONDS = 3.0
N_FFT, HOP = 512, 128
TOP_DB = 80.0
ROOTS = {
    "clean": "D:/dfdata/ext_bundle/clean/clips",
    "g711": "D:/dfdata/ext_bundle/channel/clips",
    "amr": "D:/dfdata/w13_amr/amr",
}
MANIFEST = "D:/kaggle_realcall/ssl_manifests/new_speakers.csv"
COLUMNS = (
    ("clean", "(a) Clean, 16 kHz"),
    ("g711", "(b) Simulated G.711 line"),
    ("amr", "(c) Real AMR-NB codec"),
)
ROWS = (("genuine", "Genuine"), ("clone", "XTTS-v2 clone"))
SHORT_COLUMNS = {"clean": "(a) Clean", "g711": "(b) G.711 line", "amr": "(c) AMR-NB codec"}


def log_spectrogram(audio: np.ndarray, sr: int = SR) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Magnitude STFT in dB (Hann window, 512-point FFT, 8 ms hop): (dB, freqs Hz, times s)."""
    a = np.asarray(audio, dtype=np.float64).reshape(-1)
    if a.size < N_FFT:
        a = np.pad(a, (0, N_FFT - a.size))
    window = np.hanning(N_FFT)
    frames = 1 + (a.size - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(frames)[:, None]
    spec = np.abs(np.fft.rfft(a[idx] * window, axis=1)).T
    db = 20.0 * np.log10(np.maximum(spec, 1e-10))
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    times = (np.arange(frames) * HOP + N_FFT / 2) / sr
    return db, freqs, times


def pick_pair(manifest: pd.DataFrame, roots: dict[str, str]) -> tuple[str, str]:
    """(genuine name, clone name): the first speaker with both, present in every version."""
    names = manifest["filepath"].astype(str).str.split("/").str[-1]
    df = manifest.assign(name=names)

    def present(name: str) -> bool:
        return all((Path(root) / name).is_file() for root in roots.values())

    genuine = df[df["tool"] == "none"]
    clones = df[df["tool"] == "xtts_v2"]
    for name in genuine["name"]:
        speaker = name.split("_")[0]
        same = [c for c in clones["name"] if c.split("_")[3] == speaker]
        if present(name):
            for clone in same:
                if present(clone):
                    return name, clone
    raise LookupError("no speaker has a genuine clip and an XTTS-v2 clone in every version")


def _load(path: Path, seconds: float) -> np.ndarray:
    from src.utils.audio_utils import load_wav

    audio, _ = load_wav(str(path), target_sr=SR)
    return np.asarray(audio, dtype=np.float32)[: int(seconds * SR)]


def figure(
    paths: dict[str, dict[str, Path]],
    out_dir: str | Path,
    stem: str = "channel_spectrogram",
    seconds: float = SECONDS,
    column: bool = False,
) -> list[str]:
    """Draw the 2 x 3 grid; ``paths[row][column]`` names each clip. Returns files written.

    ``column=True`` draws it one IEEE column wide (3.5 in), with short titles and no colour
    bar, for the six-page paper; the spectrograms are identical.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
    size = (3.5, 1.75) if column else (7.16, 3.2)
    title, label, tick = (5.8, 5.6, 5.0) if column else (7.5, 7.0, 6.5)
    titles = SHORT_COLUMNS if column else dict(COLUMNS)
    fig, axes = plt.subplots(2, 3, figsize=size, sharex=True, sharey=True)
    peak = max(
        log_spectrogram(_load(paths[r][c], seconds))[0].max() for r, _ in ROWS for c, _ in COLUMNS
    )
    image = None
    for i, (row, row_label) in enumerate(ROWS):
        for j, (col, _) in enumerate(COLUMNS):
            ax = axes[i, j]
            db, freqs, times = log_spectrogram(_load(paths[row][col], seconds))
            image = ax.imshow(
                db,
                origin="lower",
                aspect="auto",
                cmap="magma",
                vmin=peak - TOP_DB,
                vmax=peak,
                extent=(0, times[-1] if times.size else seconds, 0, freqs[-1] / 1000),
            )
            ax.axhline(4.0, color="white", lw=0.8, ls="--")
            if i == 0:
                ax.set_title(titles[col], fontsize=title, pad=2 if column else 4)
            if j == 0:
                ax.set_ylabel(
                    f"{row_label}\nkHz" if column else f"{row_label}\nfrequency (kHz)",
                    fontsize=label,
                )
            if i == 1:
                ax.set_xlabel("time (s)", fontsize=label)
            ax.tick_params(labelsize=tick, length=2 if column else 3.5, pad=1.5 if column else 3.5)
    axes[0, 0].text(0.08, 4.25, "4 kHz", color="white", fontsize=tick)
    if column:
        fig.subplots_adjust(left=0.13, right=0.995, bottom=0.17, top=0.9, wspace=0.08, hspace=0.12)
    else:
        cbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.015)
        cbar.set_label("dB (relative)", fontsize=7)
        cbar.ax.tick_params(labelsize=6.5)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    # a spectrogram is a continuous-tone colour image: 300 dpi is the IEEE rate for those
    for ext, dpi in (("pdf", 300), ("png", 300)):
        path = out / f"{stem}.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
        written.append(str(path))
    plt.close(fig)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Spectrograms: clean, G.711 and AMR-NB")
    parser.add_argument("--out", default="experiments/figures")
    parser.add_argument("--genuine", help="file name of the genuine clip")
    parser.add_argument("--clone", help="file name of the XTTS-v2 clone")
    parser.add_argument("--manifest", default=MANIFEST)
    args = parser.parse_args()
    if args.genuine and args.clone:
        genuine, clone = args.genuine, args.clone
    else:
        genuine, clone = pick_pair(pd.read_csv(args.manifest), ROOTS)
    paths = {
        "genuine": {c: Path(ROOTS[c]) / genuine for c, _ in COLUMNS},
        "clone": {c: Path(ROOTS[c]) / clone for c, _ in COLUMNS},
    }
    print(f"genuine {genuine}, clone {clone}")
    for path in figure(paths, args.out) + figure(
        paths, args.out, stem="channel_spectrogram_column", column=True
    ):
        print(path)


if __name__ == "__main__":
    main()
