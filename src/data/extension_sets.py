"""Genuine-clip manifests for the extension pools, and the IndicF5 job table.

Three sets are drawn from ``speaker_pools_ext.csv`` (and, for IndicF5, also the
frozen eval pool), each a seeded sample of clips 2-10 s long:

- ``eval_ext_genuine`` -- the genuine side of the larger test set: up to 40 clips per
  ``eval_ext`` speaker. Their clones (XTTS seen tool, Tortoise unseen tool) come from
  the generation run;
- ``adapt_ext_genuine`` -- extra *genuine* adaptation speech: up to 15 clips per
  ``adapt_ext`` speaker, for the run that tests whether more voices stop the detector
  learning "what MUCS lectures sound like";
- ``indicf5_genuine`` -- all-Hindi genuine clips of the speakers IndicF5 clones (the
  frozen eval pool plus ``eval_ext``), so the second unseen tool is tested against
  genuine speech in the same language: IndicF5 reads Devanagari, so its clones speak
  all-Hindi sentences.

``indicf5_jobs`` builds the IndicF5 table the same way the XTTS tables are built
(``src.data.pool_jobs``): one reference clip per speaker with its own transcript
(IndicF5 needs both), and all-Hindi target sentences taken from *other* speakers of
the same pools, cross-paired by a coprime stride so no speaker reads its own words.
"""

from __future__ import annotations

from math import gcd

import numpy as np
import pandas as pd

from src.data.codemixing import MAX_SECONDS, MIN_SECONDS, MIN_WORDS, annotate

MANIFEST_COLUMNS = [
    "filepath",
    "label",
    "language",
    "speaker",
    "source",
    "tool",
    "condition",
    "split",
    "utt_id",
    "start_seconds",
    "end_seconds",
]
#: Reference clips for cloning: long enough to carry a voice, short enough to be one turn.
REF_MIN_SECONDS, REF_MAX_SECONDS = 6.0, 10.0


def _usable(clip_index: pd.DataFrame, speakers: set[str]) -> pd.DataFrame:
    clips = clip_index[clip_index["source"].astype(str) == "mucs2021"].copy()
    clips["speaker"] = clips["speaker"].astype(str)
    clips = clips[clips["speaker"].isin(speakers)]
    return clips[clips["duration_seconds"].between(MIN_SECONDS, MAX_SECONDS)]


def sample_per_speaker(clips: pd.DataFrame, per_speaker: int, seed: int) -> pd.DataFrame:
    """Up to ``per_speaker`` clips from every speaker, seeded and order-independent."""
    rng = np.random.default_rng(seed)
    picked = []
    for _speaker, group in clips.sort_values("utt_id").groupby("speaker", sort=True):
        n = min(per_speaker, len(group))
        picked.append(group.iloc[np.sort(rng.choice(len(group), n, replace=False))])
    return pd.concat(picked, ignore_index=True) if picked else clips.head(0)


def genuine_manifest(clips: pd.DataFrame, split: str) -> pd.DataFrame:
    """The project's manifest shape for genuine MUCS clips (spans attached later)."""
    return pd.DataFrame(
        {
            "filepath": clips["wav_path"],
            "label": "bonafide",
            "language": "hi-en",
            "speaker": clips["speaker"].astype(str),
            "source": "mucs2021",
            "tool": "none",
            "condition": "clean",
            "split": split,
            "utt_id": clips["utt_id"],
            "start_seconds": clips["start_seconds"],
            "end_seconds": clips["end_seconds"],
        },
        columns=MANIFEST_COLUMNS,
    )


def pool_speakers(pools: pd.DataFrame, *names: str) -> set[str]:
    return set(pools.loc[pools["pool"].isin(names), "speaker"].astype(str))


def all_hindi(clips: pd.DataFrame) -> pd.DataFrame:
    """Clips whose transcript is entirely Devanagari, with at least MIN_WORDS words."""
    tagged = annotate(clips)
    return tagged[(tagged["english_share"] == 0.0) & (tagged["n_words"] >= MIN_WORDS)]


def indicf5_jobs(
    clip_index: pd.DataFrame, speakers: set[str], per_speaker: int = 10, seed: int = 9000
) -> pd.DataFrame:
    """One IndicF5 job per (speaker, all-Hindi sentence of another speaker)."""
    clips = _usable(clip_index, speakers)
    refs = clips[clips["duration_seconds"].between(REF_MIN_SECONDS, REF_MAX_SECONDS)]
    refs = annotate(refs)
    refs = refs[refs["english_share"] == 0.0]  # the reference text must be readable too
    refs = refs.sort_values(
        ["speaker", "duration_seconds", "utt_id"], ascending=[True, False, True]
    )
    refs = refs.groupby("speaker", sort=True).head(1).set_index("speaker")

    texts = all_hindi(clips).drop_duplicates("transcript").sort_values("utt_id")
    speaker_list = sorted(refs.index)
    n_texts = len(texts)
    while n_texts > 1 and gcd(len(speaker_list), n_texts) != 1:
        n_texts -= 1
    texts = texts.head(n_texts).reset_index(drop=True)

    rows, i = [], 0
    target = len(speaker_list) * per_speaker
    while len(rows) < target and i < len(speaker_list) * n_texts:
        speaker = speaker_list[i % len(speaker_list)]
        text = texts.iloc[i % n_texts]
        i += 1
        if str(text["speaker"]) == speaker:
            continue  # nobody reads their own sentence
        job_id = len(rows) + 1
        ref = refs.loc[speaker]
        rows.append(
            {
                "job_id": job_id,
                "speaker": speaker,
                "reference_wav": f"refs/{speaker}.wav",
                "reference_text": str(ref["transcript"]),
                "reference_utt_id": str(ref["utt_id"]),
                "transcript": str(text["transcript"]),
                "transcript_utt_id": str(text["utt_id"]),
                "language": "hi",
                "tool": "indicf5",
                "output_path": f"outputs/indicf5_{speaker}_{job_id:05d}.wav",
                "seed": seed + job_id,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """CLI: write the three genuine source manifests and the IndicF5 job table."""
    import argparse

    from src.data.speaker_pools_ext import EXT_PATH

    parser = argparse.ArgumentParser(description="Extension-pool genuine sets and IndicF5 jobs")
    parser.add_argument("--clip-index", default="data/manifests/clip_index.csv")
    parser.add_argument("--pools", default="data/manifests/speaker_pools.csv")
    parser.add_argument("--pools-ext", default=EXT_PATH)
    parser.add_argument("--out-dir", default="data/manifests")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--pack-dir", default=None, help="also cut IndicF5 reference wavs here")
    args = parser.parse_args()

    index = pd.read_csv(args.clip_index)
    ext = pd.read_csv(args.pools_ext)
    frozen = pd.read_csv(args.pools)
    out = args.out_dir

    eval_ext = pool_speakers(ext, "eval_ext")
    adapt_ext = pool_speakers(ext, "adapt_ext")
    f5_speakers = eval_ext | pool_speakers(frozen, "eval")

    sets = {
        "eval_ext_genuine_source.csv": genuine_manifest(
            sample_per_speaker(_usable(index, eval_ext), 40, args.seed), "eval"
        ),
        "adapt_ext_genuine_source.csv": genuine_manifest(
            sample_per_speaker(_usable(index, adapt_ext), 15, args.seed), "train"
        ),
        "indicf5_genuine_source.csv": genuine_manifest(
            sample_per_speaker(all_hindi(_usable(index, f5_speakers)), 20, args.seed), "eval"
        ),
    }
    for name, frame in sets.items():
        frame.to_csv(f"{out}/{name}", index=False)
        print(f"{name}: {len(frame)} clips, {frame['speaker'].nunique()} speakers")
    jobs = indicf5_jobs(index, f5_speakers)
    jobs.to_csv(f"{out}/indicf5_generation_jobs.csv", index=False)
    print(f"indicf5_generation_jobs.csv: {len(jobs)} jobs, {jobs['speaker'].nunique()} speakers")
    if args.pack_dir:
        write_references(index, jobs, args.pack_dir)


def write_references(clip_index: pd.DataFrame, jobs: pd.DataFrame, pack_dir: str) -> None:
    """Cut each job's reference clip out of its lecture recording into ``pack_dir/refs``."""
    from pathlib import Path

    from src.data.corpora import load_clip
    from src.utils.audio_utils import TARGET_SR, save_wav

    rows = clip_index.set_index(clip_index["utt_id"].astype(str))
    refs = Path(pack_dir) / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    for speaker, utt in jobs.groupby("speaker")["reference_utt_id"].first().items():
        target = refs / f"{speaker}.wav"
        if not target.exists():
            audio, sample_rate = load_clip(rows.loc[utt].to_dict(), target_sr=TARGET_SR)
            save_wav(str(target), audio, sample_rate)
    jobs.to_csv(Path(pack_dir) / "generation_jobs.csv", index=False)
    print(f"IndicF5 pack ready at {pack_dir} ({len(list(refs.glob('*.wav')))} references)")


if __name__ == "__main__":
    main()
