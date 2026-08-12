"""Build the channel-simulation listening test (W3-T2, owner L).

The channel-matched protocol is a load-bearing claim in the paper: every eval clip
is passed through an 8 kHz telephony chain so the detector is tested in the
condition it is deployed in. Before that claim is made, a human has to confirm the
output actually *sounds* like a phone call rather than like broken audio -- a
mis-tuned SNR or a botched resample would otherwise silently degrade every eval
number while the code reports success.

This module lays out the test: sample 20 clips, render each one clean and
channel-matched under the same settings the eval pipeline uses, and write a rating
sheet for the team to fill in. It does not score anything. The verdict comes from
ears, and the filled-in sheet is the artefact.

Sampling and sheet construction are pure pandas (tested in CI); only
:func:`render_pairs` touches audio.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

#: Columns the team fills in. One row per clip, one sheet per rater.
RATING_COLUMNS = [
    "pair_id",
    "clean_path",
    "channel_path",
    "source",
    "rater",
    "sounds_like_telephony_1_5",
    "intelligible_1_5",
    "artefacts_noted",
    "notes",
]

DEFAULT_N_CLIPS = 20


@dataclass
class ListeningTestConfig:
    """Settings for the listening set. Mirrors the eval channel configuration."""

    n_clips: int = DEFAULT_N_CLIPS
    seed: int = 1234
    codec: str = "g711"
    snr_db: float = 20.0
    raters: tuple[str, ...] = ("L", "M", "SK")


def sample_clips(clips: pd.DataFrame, config: ListeningTestConfig | None = None) -> pd.DataFrame:
    """Pick the clips to rate, spread across sources.

    Stratified by ``source`` so the sheet cannot end up as 20 MUCS clips and zero
    HiACC ones -- the two corpora have different recording conditions and the
    channel chain has to be judged on both.
    """
    cfg = config or ListeningTestConfig()
    if clips.empty:
        return clips.head(0)

    if "source" not in clips.columns:
        return clips.sample(n=min(cfg.n_clips, len(clips)), random_state=cfg.seed).reset_index(
            drop=True
        )

    sources = sorted(clips["source"].astype(str).unique())
    per_source = max(1, cfg.n_clips // len(sources))
    picked = [
        group.sample(n=min(per_source, len(group)), random_state=cfg.seed)
        for _, group in clips.groupby("source", sort=True)
    ]
    out = pd.concat(picked, ignore_index=True)
    if len(out) < cfg.n_clips:  # top up from whatever is left
        remaining = clips[~clips["filepath"].isin(out["filepath"])]
        if len(remaining):
            extra = remaining.sample(
                n=min(cfg.n_clips - len(out), len(remaining)), random_state=cfg.seed
            )
            out = pd.concat([out, extra], ignore_index=True)
    return out.head(cfg.n_clips).reset_index(drop=True)


def channel_filename(filepath: str, pair_id: int) -> str:
    """Output name for the channel-matched copy of a clip."""
    from pathlib import PurePosixPath

    stem = PurePosixPath(str(filepath).replace("\\", "/")).stem
    return f"pair{pair_id:02d}_{stem}.channel.wav"


def clean_filename(filepath: str, pair_id: int) -> str:
    """Output name for the clean copy of a clip."""
    from pathlib import PurePosixPath

    stem = PurePosixPath(str(filepath).replace("\\", "/")).stem
    return f"pair{pair_id:02d}_{stem}.clean.wav"


def build_rating_sheet(
    sampled: pd.DataFrame, out_dir: str, config: ListeningTestConfig | None = None
) -> pd.DataFrame:
    """One row per (clip, rater), ready for the team to fill in."""
    cfg = config or ListeningTestConfig()
    rows: list[dict[str, object]] = []
    for pair_id, row in enumerate(sampled.itertuples(index=False), start=1):
        filepath = getattr(row, "filepath", "")
        for rater in cfg.raters:
            rows.append(
                {
                    "pair_id": pair_id,
                    "clean_path": f"{out_dir}/{clean_filename(filepath, pair_id)}",
                    "channel_path": f"{out_dir}/{channel_filename(filepath, pair_id)}",
                    "source": getattr(row, "source", ""),
                    "rater": rater,
                    "sounds_like_telephony_1_5": "",
                    "intelligible_1_5": "",
                    "artefacts_noted": "",
                    "notes": "",
                }
            )
    return pd.DataFrame(rows, columns=RATING_COLUMNS)


def summarise_ratings(sheet: pd.DataFrame) -> dict:
    """Summarise a filled-in sheet. Blank rows are counted, never guessed at."""
    filled = sheet[sheet["sounds_like_telephony_1_5"].astype(str).str.strip() != ""]
    scores = pd.to_numeric(filled["sounds_like_telephony_1_5"], errors="coerce").dropna()
    intelligible = pd.to_numeric(filled["intelligible_1_5"], errors="coerce").dropna()
    return {
        "rows": int(len(sheet)),
        "rated": int(len(scores)),
        "unrated": int(len(sheet) - len(scores)),
        "mean_telephony": round(float(scores.mean()), 2) if len(scores) else None,
        "mean_intelligible": round(float(intelligible.mean()), 2) if len(intelligible) else None,
        "complete": bool(len(scores) == len(sheet)) if len(sheet) else False,
    }


def render_pairs(
    sampled: pd.DataFrame, out_dir: str, config: ListeningTestConfig | None = None
) -> int:
    """Write the clean and channel-matched wav pairs. Returns the pair count."""
    from pathlib import Path

    from src.data.channel_sim import ChannelConfig, simulate_channel
    from src.utils.audio_utils import TARGET_SR, load_wav, save_wav

    cfg = config or ListeningTestConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    written = 0
    for pair_id, row in enumerate(sampled.itertuples(index=False), start=1):
        filepath = getattr(row, "filepath", "")
        try:
            audio, sr = load_wav(str(filepath), target_sr=TARGET_SR)
        except Exception as exc:  # noqa: BLE001 - a bad clip must not stop the set
            print(f"  [skip] {filepath}: {type(exc).__name__}: {exc}")
            continue
        channel = simulate_channel(
            audio,
            ChannelConfig(codec=cfg.codec, snr_db=cfg.snr_db, in_sr=sr, seed=cfg.seed + pair_id),
        )
        save_wav(str(out / clean_filename(filepath, pair_id)), audio, sr)
        save_wav(str(out / channel_filename(filepath, pair_id)), channel, sr)
        written += 1
    return written


def main() -> None:
    """CLI: ``python -m src.data.listening_test --index data/manifests/clip_index.csv``."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Build the channel-sim listening test")
    parser.add_argument("--index", default="data/manifests/clip_index.csv")
    parser.add_argument("--out-dir", default="data/processed/listening_test")
    parser.add_argument("--sheet", default="docs/qa/channel_sim_listening_sheet.csv")
    parser.add_argument("--n-clips", type=int, default=DEFAULT_N_CLIPS)
    parser.add_argument("--codec", default="g711", help="g711 | amr_nb")
    parser.add_argument("--snr-db", type=float, default=20.0)
    parser.add_argument(
        "--sheet-only", action="store_true", help="write the sheet without rendering audio"
    )
    args = parser.parse_args()

    cfg = ListeningTestConfig(n_clips=args.n_clips, codec=args.codec, snr_db=args.snr_db)
    clips = pd.read_csv(args.index)
    sampled = sample_clips(clips, cfg)

    sheet = build_rating_sheet(sampled, args.out_dir, cfg)
    Path(args.sheet).parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(args.sheet, index=False)
    print(f"sampled {len(sampled)} clips -> {args.sheet}")

    if not args.sheet_only:
        written = render_pairs(sampled, args.out_dir, cfg)
        print(f"rendered {written} clean/channel pairs -> {args.out_dir}")
    print("next: each of L, M, SK listens to all pairs and fills in their rows")


if __name__ == "__main__":
    main()
