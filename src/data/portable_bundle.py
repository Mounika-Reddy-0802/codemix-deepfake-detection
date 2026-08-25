r"""Materialise manifest audio into standalone clips and a portable manifest.

Two problems, one operation.

**Correctness.** A MUCS row in ``clip_index.csv`` names a *span inside* a long
recording -- ``wav_path`` plus ``start_seconds``/``end_seconds`` -- because MUCS
ships 521 lecture recordings of 8-10 minutes each, not per-utterance files. A
manifest built from ``wav_path`` alone therefore points 2,217 rows at 25 files,
and the loader, which crops to the first N seconds, silently returns the same 25
snippets over and over. That is exactly how the first gap-matrix run produced a
bonafide column of 25 repeated clips wearing 2,217 different row numbers.
Materialising the span fixes it at the source.

**Portability.** Colab cannot see ``C:\dfdata``, and uploading 17 GB of MUCS to
reach 350 MB of clips is absurd. Writing only the spans a manifest actually
references gives a bundle that fits in Drive, and rewriting paths to the
``${DATA_ROOT}`` token means the same manifest resolves on either machine
without being edited.

Deterministic: clip filenames come from ``utt_id`` where present, else from a
hash of the source path and span, so two runs produce identical bundles.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

#: Manifest paths are written with this token so they resolve against whatever
#: DATA_ROOT the reading machine sets (``src.utils.paths.resolve`` expands it).
DATA_ROOT_TOKEN = "${DATA_ROOT}"


def clip_name(row: dict) -> str:
    """A stable filename for one clip."""
    # NaN is truthy, so `row.get("utt_id") or ""` would name every span-less row
    # "nan.wav" and collapse them onto one file. Test for missing explicitly.
    raw = row.get("utt_id")
    utt = "" if raw is None or pd.isna(raw) else str(raw).strip()
    if utt and utt.lower() != "nan":
        return f"{utt}.wav"
    key = f"{row.get('filepath', '')}|{row.get('start_seconds', '')}|{row.get('end_seconds', '')}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:20] + ".wav"


def attach_spans(manifest: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """Recover ``start``/``end``/``utt_id`` for rows that name a long recording.

    A manifest row carries only a filepath. When that path appears in the corpus
    index as a *span source*, the row is really one utterance inside it, and the
    span has to come back before the audio is cut -- otherwise every row sharing
    that recording yields the same opening seconds.
    """
    out = manifest.copy()
    for column in ("utt_id", "start_seconds", "end_seconds"):
        if column not in out.columns:
            out[column] = pd.NA
    if index is None or index.empty:
        return out

    spans = index.dropna(subset=["wav_path"]).copy()
    spans["wav_path"] = spans["wav_path"].astype(str)
    # A recording holding more than one utterance must be cut per utterance.
    grouped = {path: group for path, group in spans.groupby("wav_path")}

    used: dict[str, int] = {}
    for i, row in out.iterrows():
        if pd.notna(row.get("utt_id")):
            continue
        candidates = grouped.get(str(row["filepath"]))
        if candidates is None or candidates.empty:
            continue
        # Rows are assigned distinct spans in order, so N rows pointing at one
        # recording become N different utterances rather than N copies of one.
        position = used.get(str(row["filepath"]), 0)
        if position >= len(candidates):
            continue
        pick = candidates.iloc[position]
        used[str(row["filepath"])] = position + 1
        out.at[i, "utt_id"] = pick["utt_id"]
        out.at[i, "start_seconds"] = pick["start_seconds"]
        out.at[i, "end_seconds"] = pick["end_seconds"]
    return out


#: Loudness every bundled clip is normalised to, matching ``preprocess.PreprocessConfig``
#: and ``configs/data/channel_sim.yaml``.
TARGET_DBFS = -23.0


def build(
    manifest: pd.DataFrame,
    out_dir: str | Path,
    clips_subdir: str = "clips",
    target_sr: int = 16_000,
    data_root: str | None = None,
    normalise: bool = True,
    target_dbfs: float = TARGET_DBFS,
) -> pd.DataFrame:
    """Write every referenced clip into ``out_dir`` and return a portable manifest.

    Clips are RMS-normalised to ``target_dbfs`` unless ``normalise=False``.

    **This was missing and it cost a result.** ``preprocess.py`` normalises to
    -23 dBFS and ``configs/data/channel_sim.yaml`` specifies it, but this function
    went from ``load_wav`` straight to ``save_wav``. XTTS output arrives effectively
    peak-normalised (mean peak 0.997, tightly clustered); raw MUCS spans do not
    (0.940). Level alone was therefore a label, and the W5-T4 low-level-cue check
    scored 1.39% EER on eight signal statistics -- matching the adapted model's
    1.34% and leaving it no margin at all. See ``docs/results/lowlevel_cue_check_v1.md``.

    Normalising is necessary, not sufficient: it moves that check to 5.17%. The
    remainder is lecture audio against a vocoder, which no gain change fixes.
    """
    from src.data.corpora import load_clip
    from src.utils.audio_utils import load_wav, rms_normalize, save_wav
    from src.utils.paths import resolve as resolve_path

    root = Path(out_dir)
    clips = root / clips_subdir
    clips.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    written = reused = failed = 0
    for row in manifest.to_dict(orient="records"):
        name = clip_name(row)
        target = clips / name
        if not target.is_file():
            try:
                # ``data_root`` was accepted and then ignored here, so a portable
                # manifest -- the output of a previous build, carrying
                # ``${DATA_ROOT}/clips/x.wav`` -- could not be fed back in: every
                # row was handed the literal token and failed to open.
                source_path = str(resolve_path(row["filepath"], data_root))
                start, end = row.get("start_seconds"), row.get("end_seconds")
                if pd.notna(start) and pd.notna(end):
                    # A span inside a longer recording (MUCS): cut it out.
                    source = dict(row)
                    source["wav_path"] = source_path
                    audio, sample_rate = load_clip(source, target_sr=target_sr)
                else:
                    # Already a standalone clip (generated audio, HiACC): copy it
                    # through. Requiring a span here is what made the first run
                    # drop every spoof row.
                    audio, sample_rate = load_wav(source_path, target_sr=target_sr)
                if normalise:
                    audio = rms_normalize(audio, target_dbfs)
                save_wav(str(target), audio, sample_rate)
                written += 1
            except Exception as exc:  # noqa: BLE001 - one bad clip must not stop the bundle
                print(f"  [skip] {name}: {type(exc).__name__}: {exc}")
                failed += 1
                continue
        else:
            reused += 1
        portable = {k: v for k, v in row.items() if k not in ("start_seconds", "end_seconds")}
        portable["filepath"] = f"{DATA_ROOT_TOKEN}/{clips_subdir}/{name}"
        rows.append(portable)

    print(f"  clips written {written}, reused {reused}, failed {failed}")
    return pd.DataFrame(rows)


def main() -> None:
    """CLI: ``python -m src.data.portable_bundle --manifest ... --out-dir ...``."""
    import argparse

    parser = argparse.ArgumentParser(description="Bundle manifest audio into portable clips")
    parser.add_argument("--manifest", required=True, help="manifest to materialise")
    parser.add_argument("--out-dir", required=True, help="bundle root (upload this)")
    parser.add_argument("--index", default="data/manifests/clip_index.csv")
    parser.add_argument("--manifest-out", required=True, help="where to write the portable copy")
    parser.add_argument("--target-sr", type=int, default=16_000)
    parser.add_argument(
        "--data-root", default=None, help="root the INPUT manifest resolves against"
    )
    parser.add_argument(
        "--no-normalise",
        action="store_true",
        help="skip RMS normalisation (reproduces the pre-fix bundles; see W5-T4)",
    )
    parser.add_argument("--target-dbfs", type=float, default=TARGET_DBFS)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    index = pd.read_csv(args.index) if Path(args.index).is_file() else pd.DataFrame()
    prepared = attach_spans(manifest, index)

    spanned = int(prepared["start_seconds"].notna().sum())
    print(f"manifest {len(prepared)} rows; {spanned} recovered a span from the corpus index")

    portable = build(
        prepared,
        args.out_dir,
        target_sr=args.target_sr,
        data_root=args.data_root,
        normalise=not args.no_normalise,
        target_dbfs=args.target_dbfs,
    )
    Path(args.manifest_out).parent.mkdir(parents=True, exist_ok=True)
    portable.to_csv(args.manifest_out, index=False)

    unique = portable["filepath"].nunique()
    print(f"\nwrote {args.manifest_out}: {len(portable)} rows, {unique} unique clips")
    if unique < len(portable):
        print(f"  WARNING: {len(portable) - unique} duplicate clip(s) -- rows share audio")
    for label, group in portable.groupby("label"):
        print(f"  {label:9s} {len(group):5d} rows, {group['filepath'].nunique():5d} unique")


if __name__ == "__main__":
    main()
