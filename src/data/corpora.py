"""Corpus-aware clip indexing for MUCS and HiACC (W3-T2/W3-T4, owner L).

Speaker identity is the axis the whole project turns on: pools must be
speaker-disjoint, and a clone of speaker X has to carry X's id. Guessing the
speaker from a file path -- the parent-directory convention assumed by
``speaker_selection.speaker_from_path`` -- is wrong for **both** corpora we
actually downloaded:

- **MUCS 2021** is a Kaldi-style corpus. 521 long recordings sit flat in
  ``train/``, named by recording id. The 52,825 utterances and their 520 speakers
  live in ``transcripts/{segments,utt2spk,wav.scp,text}``. Path-guessing yields
  the speaker ``"train"`` for every file.
- **HiACC** stores audio as ``Corpus/adult/audio/<split>/AD09001.wav``, where the
  speaker is the filename prefix ``AD09`` and the metadata lives in
  ``metadata/speaker_info.csv``. Path-guessing yields ``"train_split"``.

So this module reads each corpus the way its authors intended. Everything here is
pandas over small metadata files -- no audio is decoded, so it is fast and
unit-testable.

Two data-quality facts found on the real 12 Aug 2026 download, both surfaced
rather than silently smoothed over:

- HiACC ``speaker_info.csv`` lists **AD65**, which has no audio at all.
- **AD63** has 119 clips but no row in ``speaker_info.csv``. It sits under
  ``adult/``, and the corpus readme states the adult/child split is by top-level
  folder, so it is treated as adult and flagged ``has_metadata=False``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

#: Columns every corpus index produces. Extra corpus-specific columns may follow.
INDEX_COLUMNS = [
    "utt_id",
    "speaker",
    "source",
    "wav_path",
    "start_seconds",
    "end_seconds",
    "duration_seconds",
    "transcript",
]

MUCS_SOURCE = "mucs2021"
HIACC_SOURCE = "hiacc"


class CorpusError(RuntimeError):
    """Raised when a corpus is missing the metadata needed to identify speakers."""


# --------------------------------------------------------------------------- #
# MUCS 2021 -- Kaldi-style
# --------------------------------------------------------------------------- #
def read_kaldi_table(path: str, max_split: int = 1) -> dict[str, str]:
    """Read a Kaldi two-column table: first field is the key, the rest is the value.

    ``max_split=1`` keeps the value intact for tables like ``text`` whose value is
    a whole transcript containing spaces.
    """
    table: dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, max_split)
            if len(parts) < 2:
                continue
            table[parts[0]] = parts[1]
    return table


def read_segments(path: str) -> dict[str, tuple[str, float, float]]:
    """Read Kaldi ``segments``: ``utt_id recording_id start end``."""
    segments: dict[str, tuple[str, float, float]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 4:
                continue
            utt, rec, start, end = parts
            try:
                segments[utt] = (rec, float(start), float(end))
            except ValueError:
                continue
    return segments


def index_mucs(root: str, split: str = "train") -> pd.DataFrame:
    """Index one MUCS split into per-utterance rows with real speaker ids.

    Each row is one *utterance* -- a time span inside a long recording -- not a
    whole file. That is what makes a MUCS clip attributable to a single speaker.
    """
    base = Path(root) / split
    transcripts = base / "transcripts"
    required = ["segments", "utt2spk", "wav.scp"]
    missing = [name for name in required if not (transcripts / name).is_file()]
    if missing:
        raise CorpusError(f"{transcripts}: missing Kaldi tables {missing}")

    segments = read_segments(str(transcripts / "segments"))
    utt2spk = read_kaldi_table(str(transcripts / "utt2spk"))
    wav_scp = read_kaldi_table(str(transcripts / "wav.scp"))
    text_path = transcripts / "text"
    text = read_kaldi_table(str(text_path)) if text_path.is_file() else {}

    rows: list[dict[str, object]] = []
    for utt, (recording, start, end) in segments.items():
        speaker = utt2spk.get(utt)
        wav_name = wav_scp.get(recording)
        if speaker is None or wav_name is None:
            continue  # an utterance we cannot attribute is not usable
        rows.append(
            {
                "utt_id": utt,
                "speaker": speaker,
                "source": MUCS_SOURCE,
                "wav_path": str(base / wav_name),
                "start_seconds": start,
                "end_seconds": end,
                "duration_seconds": round(end - start, 3),
                "transcript": text.get(utt, ""),
                "split": split,
            }
        )
    return pd.DataFrame(rows, columns=[*INDEX_COLUMNS, "split"])


# --------------------------------------------------------------------------- #
# HiACC
# --------------------------------------------------------------------------- #
def hiacc_speaker(filename: str, known_pids: set[str]) -> str | None:
    """Speaker id from a HiACC filename (``AD09001.wav`` -> ``AD09``).

    Matches the longest known PID first so a hypothetical ``AD1`` cannot shadow
    ``AD13``. Returns ``None`` when the prefix matches no known speaker; the
    caller decides whether that is fatal.
    """
    stem = Path(filename).name
    for pid in sorted(known_pids, key=len, reverse=True):
        if stem.startswith(pid):
            return pid
    return None


def _hiacc_fallback_pid(filename: str) -> str | None:
    """Derive a PID for a clip whose speaker is absent from ``speaker_info.csv``.

    HiACC ids are ``<2 letters><2-digit speaker><3-digit utterance>``, so the
    first four characters are the speaker. Used only to surface undocumented
    speakers (AD63 on the real corpus) rather than dropping their audio silently.
    """
    stem = Path(filename).stem
    if len(stem) >= 4 and stem[:2].isalpha() and stem[2:4].isdigit():
        return stem[:4]
    return None


def index_hiacc(root: str, category: str = "adult") -> pd.DataFrame:
    """Index the HiACC adult subset, joined to its speaker and sentence metadata.

    ``category`` defaults to ``adult`` and should never be changed: the child
    subset is quarantined and must not be indexed. The audio path is checked to
    ensure it does not sit inside a quarantine directory.
    """
    from src.data.preprocess import is_excluded

    if category != "adult":
        raise CorpusError(
            f"refusing to index HiACC category {category!r}; only 'adult' may be used"
        )

    base = Path(root) / "Corpus" / category
    audio_dir = base / "audio"
    if not audio_dir.is_dir():
        raise CorpusError(f"{audio_dir} not found -- extract HiACC first")

    info_path = base / "metadata" / "speaker_info.csv"
    stats_path = base / "metadata" / "sentence_stats.csv"
    speaker_info = pd.read_csv(info_path) if info_path.is_file() else pd.DataFrame(columns=["PID"])
    known_pids = set(speaker_info["PID"].astype(str)) if len(speaker_info) else set()

    stats = pd.read_csv(stats_path) if stats_path.is_file() else pd.DataFrame()
    stats_by_audio = (
        {str(r["audio"]): r for _, r in stats.iterrows()} if "audio" in stats.columns else {}
    )

    rows: list[dict[str, object]] = []
    for wav in sorted(audio_dir.rglob("*.wav")):
        if is_excluded(wav):  # belt and braces: never index quarantined audio
            continue
        name = wav.name
        speaker = hiacc_speaker(name, known_pids)
        documented = speaker is not None
        if speaker is None:
            speaker = _hiacc_fallback_pid(name)
        if speaker is None:
            continue

        stat = stats_by_audio.get(name)
        rows.append(
            {
                "utt_id": wav.stem,
                "speaker": speaker,
                "source": HIACC_SOURCE,
                "wav_path": str(wav),
                "start_seconds": 0.0,
                "end_seconds": float(stat["duration_sec"]) if stat is not None else 0.0,
                "duration_seconds": float(stat["duration_sec"]) if stat is not None else 0.0,
                "transcript": str(stat["sentence"]) if stat is not None else "",
                "split": wav.parent.name.replace("_split", ""),
                "cmi": float(stat["CMI"]) if stat is not None else None,
                "code_switch_count": int(stat["code_switch_count"]) if stat is not None else None,
                "has_metadata": documented,
            }
        )

    columns = [*INDEX_COLUMNS, "split", "cmi", "code_switch_count", "has_metadata"]
    return pd.DataFrame(rows, columns=columns)


# --------------------------------------------------------------------------- #
# Clip loading
# --------------------------------------------------------------------------- #
def load_clip(row, target_sr: int | None = None):
    """Decode one indexed clip, honouring its time span.

    A MUCS row names a span inside a long recording, so only that span is read --
    ``soundfile`` seeks rather than loading a 30-minute file to keep 5 seconds.
    A HiACC row spans a whole file and is read entirely.

    Returns ``(audio, sample_rate)`` as float32 mono.
    """
    import soundfile as sf

    from src.utils.audio_utils import resample, to_mono

    path = row["wav_path"] if isinstance(row, dict) else row.wav_path
    start = float(row["start_seconds"] if isinstance(row, dict) else row.start_seconds)
    end = float(row["end_seconds"] if isinstance(row, dict) else row.end_seconds)

    with sf.SoundFile(path) as handle:
        sample_rate = handle.samplerate
        if end > start > -1 and end > 0:
            handle.seek(int(start * sample_rate))
            frames = int((end - start) * sample_rate)
            audio = handle.read(frames, dtype="float32", always_2d=False)
        else:
            audio = handle.read(dtype="float32", always_2d=False)

    audio = to_mono(audio)
    if target_sr is not None and sample_rate != target_sr:
        audio = resample(audio, sample_rate, target_sr)
        sample_rate = target_sr
    return audio, sample_rate


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def index_summary(frame: pd.DataFrame) -> dict:
    """Counts a human needs before trusting an index."""
    if frame.empty:
        return {"clips": 0, "speakers": 0, "hours": 0.0}
    summary = {
        "clips": int(len(frame)),
        "speakers": int(frame["speaker"].nunique()),
        "hours": round(float(frame["duration_seconds"].sum()) / 3600.0, 2),
        "median_clip_seconds": round(float(frame["duration_seconds"].median()), 2),
        "with_transcript": int((frame["transcript"].astype(str).str.strip() != "").sum()),
    }
    if "has_metadata" in frame.columns:
        undocumented = sorted(frame.loc[~frame["has_metadata"], "speaker"].unique())
        summary["undocumented_speakers"] = undocumented
    return summary


def main() -> None:
    """CLI: ``python -m src.data.corpora --data-root C:/dfdata``."""
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser(description="Index MUCS and HiACC by real speaker id")
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "data"))
    parser.add_argument("--out", default="data/manifests/clip_index.csv")
    parser.add_argument("--mucs-split", default="train")
    args = parser.parse_args()

    raw = Path(args.data_root) / "raw"
    frames: list[pd.DataFrame] = []

    mucs_root = raw / "mucs2021"
    if mucs_root.is_dir():
        mucs = index_mucs(str(mucs_root), args.mucs_split)
        print(f"MUCS  {json.dumps(index_summary(mucs))}")
        frames.append(mucs)

    hiacc_root = raw / "hiacc"
    if hiacc_root.is_dir():
        hiacc = index_hiacc(str(hiacc_root))
        print(f"HiACC {json.dumps(index_summary(hiacc), default=str)}")
        frames.append(hiacc)

    if not frames:
        raise SystemExit(f"no corpora found under {raw}")

    combined = pd.concat(frames, ignore_index=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False)
    print(f"\nwrote {len(combined)} rows across {combined['speaker'].nunique()} speakers -> {out}")


if __name__ == "__main__":
    main()
