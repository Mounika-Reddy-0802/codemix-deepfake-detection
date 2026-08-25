"""Render a manifest's audio through the telephony chain into a matched bundle.

Why this exists. ``gap_matrix_v1.md`` states its own central caveat plainly: the
code-mixed gap it measured confounds **language shift** with **recording-domain
shift**, because ASVspoof is studio read speech and MUCS is NPTEL lecture audio.
The remedy it names is a channel-matched column -- push every corpus through the
same 8 kHz chain, which compresses the recording differences, so a gap that
survives is more attributable to language than to microphones.

The second reason is the deployment. This project targets live call detection, so
a number measured on clean 16 kHz lecture audio is not the number that matters; a
detector that only works before the codec is not deployable. This module produces
the audio those two questions are asked on.

:mod:`src.data.channel_sim` already holds the verified chain (16 kHz -> 8 kHz ->
codec -> noise@SNR -> 16 kHz; g711 at 20 dB was validated objectively in W3-T2 --
20/20 clips band-limited, measured SNR 17.6 dB, correlation 0.99 -- and passed a
three-rater listening test at 4.0/5 intelligible). What was missing is a
manifest-level applier: the only caller was ``listening_test``, which renders one
pair at a time.

**Noise is seeded per clip, never globally.** One shared noise draw across a whole
manifest would add a bit-identical waveform to every clip, and a detector can
learn that instead of the speech -- the same class of shortcut the duration and
speaker-firewall checks exist to rule out. The seed is derived from the clip's own
name, so the rendering is deterministic and resumable without being uniform.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from src.data.portable_bundle import DATA_ROOT_TOKEN

#: The condition validated in W3-T2 and the one the gap matrix's channel-matched
#: column is defined against. Anything else is an explicit experiment.
DEFAULT_CODEC = "g711"
DEFAULT_SNR_DB = 20.0


def condition_name(codec: str, snr_db: float) -> str:
    """Label written into the manifest's ``condition`` column.

    Carried per row rather than assumed, so a clean and a channel manifest can be
    concatenated and still sliced apart -- ``evaluate.per_attack_eer`` groups by
    ``tool``, and any per-condition breakdown needs the same for ``condition``.
    """
    return f"channel_{codec}_{snr_db:g}db"


def clip_seed(name: str, base_seed: int) -> int:
    """A stable per-clip noise seed derived from the clip's filename.

    ``hash()`` is salted per process in Python 3, so it cannot be used here: the
    bundle has to come out identical on a re-run and on another machine.
    """
    digest = hashlib.sha256(f"{base_seed}|{name}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def render(
    manifest: pd.DataFrame,
    out_dir: str | Path,
    clips_subdir: str = "clips",
    codec: str = DEFAULT_CODEC,
    snr_db: float = DEFAULT_SNR_DB,
    target_sr: int = 16_000,
    data_root: str | None = None,
    base_seed: int = 1234,
) -> pd.DataFrame:
    """Render every clip through the channel and return a portable manifest.

    ``manifest`` is expected to be portable already (``${DATA_ROOT}/clips/x.wav``),
    i.e. the output of :func:`src.data.portable_bundle.build` -- the spans have
    been cut, so each row is one standalone clip.

    Resumable: a clip whose target already exists is left alone, so an interrupted
    render costs only the clips it had not reached, matching how spoof generation
    and score caching behave elsewhere in the project.
    """
    from src.data.channel_sim import ChannelConfig, simulate_channel
    from src.utils.audio_utils import load_wav, save_wav
    from src.utils.paths import resolve as resolve_path

    root = Path(out_dir)
    clips = root / clips_subdir
    clips.mkdir(parents=True, exist_ok=True)
    condition = condition_name(codec, snr_db)

    rows: list[dict] = []
    written = reused = failed = 0
    for row in manifest.to_dict(orient="records"):
        name = Path(str(row["filepath"])).name
        target = clips / name
        if not target.is_file():
            try:
                audio, sample_rate = load_wav(
                    str(resolve_path(row["filepath"], data_root)), target_sr=target_sr
                )
                rendered = simulate_channel(
                    audio,
                    ChannelConfig(
                        codec=codec,
                        snr_db=snr_db,
                        in_sr=sample_rate,
                        seed=clip_seed(name, base_seed),
                    ),
                )
                save_wav(str(target), rendered, sample_rate)
                written += 1
            except Exception as exc:  # noqa: BLE001 - one bad clip must not stop the bundle
                print(f"  [skip] {name}: {type(exc).__name__}: {exc}")
                failed += 1
                continue
        else:
            reused += 1
        out = dict(row)
        out["filepath"] = f"{DATA_ROOT_TOKEN}/{clips_subdir}/{name}"
        out["condition"] = condition
        rows.append(out)

    print(f"  clips written {written}, reused {reused}, failed {failed}")
    return pd.DataFrame(rows)


def verify(
    clean: pd.DataFrame,
    channel: pd.DataFrame,
    clean_root: str | None,
    channel_root: str | None,
    sample: int = 20,
    seed: int = 1234,
) -> dict[str, float]:
    """Objective check that the chain actually did something, on ``sample`` clips.

    W3-T2 established what a correct render looks like: energy above 4 kHz is
    destroyed by the 8 kHz round-trip, and the measured SNR lands near the target.
    Re-measuring here means a silently mis-wired chain -- a codec that no-ops, a
    resample that never happened -- is caught before it becomes a result table.
    """
    import numpy as np

    from src.data.channel_sim import measured_snr_db
    from src.utils.audio_utils import load_wav
    from src.utils.paths import resolve as resolve_path

    merged = clean.copy()
    merged["_name"] = merged["filepath"].map(lambda p: Path(str(p)).name)
    lookup = {Path(str(p)).name: p for p in channel["filepath"]}
    picks = merged.sample(n=min(sample, len(merged)), random_state=seed)

    hf_clean: list[float] = []
    hf_channel: list[float] = []
    snrs: list[float] = []
    for _, row in picks.iterrows():
        if row["_name"] not in lookup:
            continue
        a, sr = load_wav(str(resolve_path(row["filepath"], clean_root)))
        b, _ = load_wav(str(resolve_path(lookup[row["_name"]], channel_root)))
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        if n == 0:
            continue
        hf_clean.append(_hf_energy_ratio(a, sr))
        hf_channel.append(_hf_energy_ratio(b, sr))
        snrs.append(measured_snr_db(a, b))

    return {
        "clips": float(len(snrs)),
        "hf_ratio_clean_pct": 100.0 * float(np.mean(hf_clean)) if hf_clean else float("nan"),
        "hf_ratio_channel_pct": 100.0 * float(np.mean(hf_channel)) if hf_channel else float("nan"),
        "measured_snr_db": float(np.mean(snrs)) if snrs else float("nan"),
    }


def _hf_energy_ratio(signal, sr: int, cutoff_hz: float = 4_000.0) -> float:
    """Share of spectral energy above ``cutoff_hz`` (the narrowband ceiling)."""
    import numpy as np

    spectrum = np.abs(np.fft.rfft(np.asarray(signal, dtype=np.float64)))
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sr)
    total = float(np.sum(spectrum**2))
    if total <= 0.0:
        return 0.0
    return float(np.sum(spectrum[freqs >= cutoff_hz] ** 2) / total)


def main() -> None:
    """CLI: ``python -m src.data.channel_bundle --manifest ... --out-dir ...``."""
    import argparse

    parser = argparse.ArgumentParser(description="Render a manifest through the telephony chain")
    parser.add_argument("--manifest", required=True, help="portable manifest to render")
    parser.add_argument("--out-dir", required=True, help="channel bundle root")
    parser.add_argument("--manifest-out", required=True, help="where to write the new manifest")
    parser.add_argument("--codec", default=DEFAULT_CODEC, help="g711 | amr_nb | none")
    parser.add_argument("--snr-db", type=float, default=DEFAULT_SNR_DB)
    parser.add_argument(
        "--data-root", default=None, help="root the input manifest resolves against"
    )
    parser.add_argument("--target-sr", type=int, default=16_000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--verify-sample", type=int, default=20, help="0 to skip the check")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    print(f"rendering {len(manifest)} rows through {args.codec} @ {args.snr_db:g} dB")
    rendered = render(
        manifest,
        args.out_dir,
        codec=args.codec,
        snr_db=args.snr_db,
        target_sr=args.target_sr,
        data_root=args.data_root,
        base_seed=args.seed,
    )
    Path(args.manifest_out).parent.mkdir(parents=True, exist_ok=True)
    rendered.to_csv(args.manifest_out, index=False)

    unique = rendered["filepath"].nunique()
    print(f"\nwrote {args.manifest_out}: {len(rendered)} rows, {unique} unique clips")
    if unique < len(rendered):
        print(f"  WARNING: {len(rendered) - unique} duplicate clip(s) -- rows share audio")
    for label, group in rendered.groupby("label"):
        print(f"  {label:9s} {len(group):5d} rows, {group['filepath'].nunique():5d} unique")

    if args.verify_sample:
        stats = verify(manifest, rendered, args.data_root, args.out_dir, sample=args.verify_sample)
        print(
            f"\nverified on {stats['clips']:.0f} clips: "
            f"energy above 4 kHz {stats['hf_ratio_clean_pct']:.3f}% clean -> "
            f"{stats['hf_ratio_channel_pct']:.4f}% channel, "
            f"measured SNR {stats['measured_snr_db']:.1f} dB"
        )


if __name__ == "__main__":
    main()
