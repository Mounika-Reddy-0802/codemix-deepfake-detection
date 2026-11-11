"""Playback files for the corpus replay calls: clones of held-out speakers played into a phone.

The 30 real calls replayed only the team's own clones, six per tool, and every model misses
the replayed Chatterbox clones (0 of 6). This pack widens that test without recording anyone
new: test-set clips of corpus speakers outside ``gen_train`` (no model trained on them), one
speaker per file, for every tool the project measures, with genuine clips of the same
speaker pool through the same loudspeaker as the control.

Nothing is chosen by score. Within each kind the speakers are drawn at random (generator
seed 20261110) among the held-out speakers with at least ``MIN_SPEAKER_S`` seconds of that
kind (so no file is too short to judge after a phone call; lengths only, never scores), then up to ``MAX_CLIPS`` of that speaker's clips
in random order are joined with short gaps until the file reaches ``TARGET_S`` seconds.
Every file is resampled to 16 kHz and set to the same loudness. Files get neutral names
(``R01.wav`` ...), shuffled, so whoever plays them does not see the label; the plan holds it::

    python -m src.data.corpus_replay_pack --clean <clean list> --out D:/replay_pack
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data import generator_study as gs
from src.reporting.voice_check_v2_clean import speaker_map

SEED = 20261110
SR = 16_000
TARGET_S = 25.0
MIN_SPEAKER_S = 12.0  # a speaker needs this much audio of the kind to be drawn
MAX_CLIPS = 6
GAP_S = 0.6
TARGET_RMS_DBFS = -26.0
#: (kind, files, set in the clean list, label, tool); one speaker per file
KINDS = (
    ("genuine", 6, ("cm07", "cm09"), "bonafide", "none"),
    ("indicf5", 6, ("cm07",), "spoof", "indicf5"),
    ("chatterbox", 6, ("cm09",), "spoof", "chatterbox"),
    ("rvc", 6, ("rvc_test",), "spoof", "rvc"),
    ("tortoise", 3, ("cm04",), "spoof", "tortoise"),
    ("xtts_v2", 3, ("new_speakers",), "spoof", "xtts_v2"),
)


def eligible(
    clean: pd.DataFrame, speakers: dict[str, str], train: set[str], duration=None
) -> pd.DataFrame:
    """Test-split clips with a known speaker outside ``train``, with their length in seconds."""
    if duration is None:
        import soundfile as sf

        def duration(path: str) -> float:
            return float(sf.info(path).duration)

    d = clean[clean["split"] == "test"].copy()
    d["speaker"] = d["filepath"].str.replace("\\", "/").str.split("/").str[-1].map(speakers)
    d = d[d["speaker"].notna() & ~d["speaker"].isin(train)].reset_index(drop=True)
    d["seconds"] = [duration(p) for p in d["filepath"]]
    return d


def choose(clips: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """One row per playback file: kind, speaker and its clips, in a shuffled neutral order."""
    rng = np.random.default_rng(seed)
    rows = []
    for kind, n, sets, label, tool in KINDS:
        pool = clips[clips["set"].isin(sets) & (clips["label"] == label) & (clips["tool"] == tool)]
        total = pool.groupby("speaker")["seconds"].sum()
        names = sorted(total[total >= MIN_SPEAKER_S].index)
        if len(names) < n:
            raise ValueError(f"{kind}: only {len(names)} held-out speakers long enough, {n} needed")
        for spk in rng.choice(names, size=n, replace=False):
            files = sorted(pool.loc[pool["speaker"] == spk, "filepath"])
            order = rng.permutation(len(files))[:MAX_CLIPS]
            rows.append(
                {
                    "kind": kind,
                    "label": label,
                    "tool": tool,
                    "speaker": spk,
                    "clips": ";".join(files[i] for i in order),
                }
            )
    plan = pd.DataFrame(rows).iloc[rng.permutation(len(rows))].reset_index(drop=True)
    plan.insert(0, "file", [f"R{i:02d}.wav" for i in range(1, len(plan) + 1)])
    return plan


def join(paths: list[str], read) -> np.ndarray:
    """The clips at 16 kHz, joined with gaps until ``TARGET_S``, at the target loudness."""
    from scipy.signal import resample_poly

    gap = np.zeros(int(GAP_S * SR), dtype=np.float64)
    parts: list[np.ndarray] = []
    total = 0.0
    for p in paths:
        audio, sr = read(p)
        audio = np.asarray(audio, dtype=np.float64)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SR:
            g = np.gcd(int(sr), SR)
            audio = resample_poly(audio, SR // g, int(sr) // g)
        parts += [audio, gap]
        total += len(audio) / SR + GAP_S
        if total >= TARGET_S:
            break
    out = np.concatenate(parts[:-1])
    rms = np.sqrt(np.mean(out**2)) or 1.0
    out = out * (10 ** (TARGET_RMS_DBFS / 20) / rms)
    peak = np.abs(out).max()
    return out / peak * 0.99 if peak > 0.99 else out


def write(plan: pd.DataFrame, out_dir: Path, read=None, write_wav=None) -> pd.DataFrame:
    """Write every playback file and the plan with each file's length."""
    import soundfile as sf

    read = read or sf.read
    write_wav = write_wav or (lambda path, audio: sf.write(path, audio, SR, subtype="PCM_16"))
    out_dir.mkdir(parents=True, exist_ok=True)
    seconds = []
    for row in plan.itertuples(index=False):
        audio = join(row.clips.split(";"), read)
        write_wav(out_dir / row.file, audio)
        seconds.append(round(len(audio) / SR, 1))
    plan = plan.assign(seconds=seconds)
    plan.to_csv(out_dir / "plan.csv", index=False)
    return plan


def main() -> None:
    """CLI: choose the files and write them with the plan."""
    import argparse

    parser = argparse.ArgumentParser(description="Corpus replay playback files")
    parser.add_argument("--clean", required=True, help="the clean clip list (filepath, set...)")
    parser.add_argument("--pools", default=f"data/manifests/{gs.POOLS_OUT}")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    clips = eligible(pd.read_csv(args.clean), speaker_map(), gs.train_speakers(args.pools))
    plan = write(choose(clips), Path(args.out))
    print(plan[["file", "kind", "speaker", "seconds"]].to_string(index=False))
    print(plan.groupby("kind").size().to_string())


if __name__ == "__main__":
    main()
