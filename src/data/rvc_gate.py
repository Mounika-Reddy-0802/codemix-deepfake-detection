"""The W5-T4 shortcut gate, asked of CM02: does voice conversion leave a cheap tell?

``lowlevel_cue`` ran AffectDF's Appendix-G check on the corpus every headline number
uses and it **failed**: eight cheap signal statistics separate MUCS bonafide from
XTTS spoof at 1.39% EER, and normalising levels only moves that to 5.17%
(``docs/results/lowlevel_cue_check_v1.md``). Part of that is a bundle bug; the rest
is XTTS itself -- a vocoder's output does not have a lecture hall's peak, RMS
spread or zero-crossing rate.

CM02 gives the gate a sharper question. Every RVC job records the exact source
segment it converted, so the pairing here is:

- **bonafide** = the unique MUCS source segments the RVC jobs read (1,345 clips);
- **spoof** = the RVC conversions of *those same segments* (1,500 clips).

Same recordings, same words, same room. The only thing that differs is the
conversion, so if the eight statistics can still tell the classes apart, the tell
is in the conversion and nowhere else. That is the form of the question P-021
needs answered: RVC and XTTS deviate from real speech in opposite pitch directions,
and this asks whether RVC also differs on the cheap features XTTS fails on.

**Fit/score are disjoint by source speaker** on both classes. The identity that
matters for a low-level-cue check is the recording channel, and the channel comes
from the source, not the target voice -- so the 25 source speakers are split 13/12
and no recording appears on both sides. Target voices are deliberately *mixed*
across the split: if each voice model carried its own gain, that would make the
shortcut easier to find, so a PASS here is conservative.

Two conditions, mirroring the two rows of the original check: ``raw`` (clips as
recorded / as generated) and ``normalised`` (both classes at 16 kHz, RMS -23 dBFS --
what ``preprocess.py`` already does to everything else). Manifest construction is
pure and tested; the audio steps import lazily and run wherever the clips are.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.lowlevel_cue import run_check, verdict

__all__ = [
    "CONDITIONS",
    "cut_sources",
    "normalised_copies",
    "paired_manifest",
    "run_gate",
    "split_source_speakers",
    "unique_sources",
]

CONDITIONS = ("raw", "normalised")
TARGET_DBFS = -23.0
TARGET_SR = 16_000
RESULTS_DIR = "experiments/results"


# --------------------------------------------------------------------------- #
# Manifest construction (pure pandas, tested)
# --------------------------------------------------------------------------- #
def unique_sources(jobs: pd.DataFrame) -> pd.DataFrame:
    """One row per source segment the RVC jobs converted -- the bonafide side.

    A source clip may have been converted into several targets; it is one bonafide
    clip regardless, or the bonafide side would be weighted by how often the job
    builder happened to reuse it.
    """
    frame = jobs.copy()
    frame["source_speaker"] = frame["source_speaker"].astype(str)
    return frame.drop_duplicates("source_utt_id").reset_index(drop=True)


def split_source_speakers(speakers) -> tuple[set[str], set[str]]:
    """Deterministic fit/score split of source speakers: alternate over sorted ids.

    Sorted, not shuffled, so two machines produce the same split from the same job
    table (P-016). Alternating rather than halving keeps any ordering in the ids --
    they are MUCS speaker numbers, which are not random -- from landing on one side.
    """
    ordered = sorted({str(s) for s in speakers})
    fit = set(ordered[::2])
    return fit, set(ordered) - fit


def paired_manifest(
    jobs: pd.DataFrame,
    clips_dir: str,
    sources_dir: str,
    fit_speakers: set[str] | None = None,
) -> pd.DataFrame:
    """Bonafide sources + their RVC conversions, labelled and split by source speaker.

    Columns match what ``lowlevel_cue.extract`` reads (``filepath``, ``label``) plus
    ``speaker``, ``tool`` and ``split`` (``fit`` | ``score``). Paths are built from
    basenames, so the clips may live anywhere the caller points at. Rows whose file
    is missing are kept -- the audio step reports them -- because silently dropping
    one class's misses would bias the check.
    """
    sources = unique_sources(jobs)
    fit = (
        fit_speakers
        if fit_speakers is not None
        else split_source_speakers(sources["source_speaker"])[0]
    )
    fit = {str(s) for s in fit}

    rows: list[dict] = []
    for row in sources.to_dict(orient="records"):
        speaker = str(row["source_speaker"])
        rows.append(
            {
                "filepath": str(Path(sources_dir) / f"{row['source_utt_id']}.wav"),
                "label": "bonafide",
                "speaker": speaker,
                "tool": "none",
                "split": "fit" if speaker in fit else "score",
            }
        )
    for row in jobs.to_dict(orient="records"):
        speaker = str(row["source_speaker"])
        name = Path(str(row["output_path"]).replace("\\", "/")).name
        rows.append(
            {
                "filepath": str(Path(clips_dir) / name),
                "label": "spoof",
                "speaker": speaker,
                "tool": str(row.get("tool", "rvc")),
                "split": "fit" if speaker in fit else "score",
            }
        )
    return pd.DataFrame(rows, columns=["filepath", "label", "speaker", "tool", "split"])


# --------------------------------------------------------------------------- #
# Audio steps (lazy imports; run where the clips are)
# --------------------------------------------------------------------------- #
def cut_sources(jobs: pd.DataFrame, sources_dir: str, data_root: str | None = None) -> dict:
    """Cut each unique source segment out of its MUCS recording, once.

    A job row names a span inside a long lecture wav (``source_wav`` +
    ``start_seconds``/``end_seconds``), and ``source_wav`` is whatever path the
    generating machine had; it is re-rooted here through ``paths.resolve`` so the
    same job table cuts the same clips on any machine that has the corpus.
    """
    from src.data.corpora import load_clip
    from src.utils.audio_utils import save_wav
    from src.utils.paths import resolve

    dest_dir = Path(sources_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    cut = reused = failed = 0
    for row in unique_sources(jobs).to_dict(orient="records"):
        dest = dest_dir / f"{row['source_utt_id']}.wav"
        if dest.is_file():
            reused += 1
            continue
        try:
            audio, sr = load_clip(
                {
                    "wav_path": resolve(str(row["source_wav"]), data_root),
                    "start_seconds": row["start_seconds"],
                    "end_seconds": row["end_seconds"],
                }
            )
            save_wav(str(dest), audio, sr)
            cut += 1
        except Exception as exc:  # noqa: BLE001 - one unreadable recording must not end the run
            failed += 1
            print(f"  [skip] {row['source_utt_id']}: {type(exc).__name__}: {exc}")
    return {"cut": cut, "reused": reused, "failed": failed}


def normalised_copies(src_dir: str, dest_dir: str) -> int:
    """16 kHz, RMS -23 dBFS copies of every wav in ``src_dir`` -- the pipeline's own
    canonical form, applied identically to both classes."""
    from src.utils.audio_utils import load_wav, resample, rms_normalize, save_wav

    out = Path(dest_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = 0
    for wav in sorted(Path(src_dir).glob("*.wav")):
        dest = out / wav.name
        if dest.is_file():
            continue
        audio, sr = load_wav(str(wav))
        if sr != TARGET_SR:
            audio = resample(audio, sr, TARGET_SR)
        save_wav(str(dest), rms_normalize(audio, TARGET_DBFS), TARGET_SR)
        written += 1
    return written


def run_gate(
    jobs: pd.DataFrame,
    clips_dir: str,
    work_dir: str,
    data_root: str | None = None,
    conditions: tuple[str, ...] = CONDITIONS,
    out_dir: str = RESULTS_DIR,
) -> dict[str, dict]:
    """Cut, (normalise,) fit on one speaker half, score the other. One JSON per condition."""
    work = Path(work_dir)
    sources_dir = work / "bonafide_source"
    stats = cut_sources(jobs, str(sources_dir), data_root)
    print(
        f"bonafide sources: {stats['cut']} cut, {stats['reused']} reused, {stats['failed']} failed"
    )

    fit, score = split_source_speakers(unique_sources(jobs)["source_speaker"])
    print(f"source speakers: fit {len(fit)} / score {len(score)}")

    results: dict[str, dict] = {}
    for condition in conditions:
        if condition == "normalised":
            bona = work / "normalised" / "bonafide"
            spoof = work / "normalised" / "spoof"
            print(
                f"  normalised bonafide: {normalised_copies(str(sources_dir), str(bona))} written"
            )
            print(f"  normalised spoof:    {normalised_copies(clips_dir, str(spoof))} written")
            manifest = paired_manifest(jobs, str(spoof), str(bona), fit)
        else:
            manifest = paired_manifest(jobs, clips_dir, str(sources_dir), fit)

        present = manifest[manifest["filepath"].map(lambda p: Path(p).is_file())]
        missing = len(manifest) - len(present)
        fit_rows = present[present["split"] == "fit"].reset_index(drop=True)
        score_rows = present[present["split"] == "score"].reset_index(drop=True)
        print(
            f"\n=== {condition}: fit {len(fit_rows)} -> score {len(score_rows)} clips"
            f"{f'  ({missing} listed files missing)' if missing else ''} ==="
        )
        result = run_check(fit_rows, score_rows)
        result.update(
            {
                "condition": condition,
                "design": (
                    "bonafide = unique MUCS source segments of the RVC jobs; spoof = their "
                    "RVC conversions; fit/score disjoint by source speaker"
                ),
                "fit_speakers": sorted(fit),
                "score_speakers": sorted(score),
                "missing_files": int(missing),
                "verdict": verdict(result["eer"]),
            }
        )
        path = Path(out_dir) / f"lowlevel_cue_check_cm02_{condition}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        ranked = sorted(result["coefficients"].items(), key=lambda kv: abs(kv[1]), reverse=True)
        print(f"  EER {result['eer']:.4f}   AUC {result['auc']:.4f}")
        print("  hinges on: " + ", ".join(f"{k} {v:+.3f}" for k, v in ranked[:3]))
        print(f"  {result['verdict']}")
        print(f"  wrote {path}")
        results[condition] = result
    return results


def main() -> None:
    """CLI: ``python -m src.data.rvc_gate --clips <rvc wavs> --data-root $DATA_ROOT``."""
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Shortcut gate on CM02: sources vs conversions")
    parser.add_argument("--jobs", default="data/manifests/rvc_generation_jobs.csv")
    parser.add_argument("--clips", required=True, help="directory holding the 1,500 rvc_*.wav")
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT"), help="has raw/mucs2021")
    parser.add_argument("--work", default=None, help="default: <data-root>/interim/rvc_gate")
    parser.add_argument("--condition", choices=CONDITIONS, action="append", default=None)
    parser.add_argument("--out-dir", default=RESULTS_DIR)
    args = parser.parse_args()

    if not args.data_root:
        raise SystemExit("--data-root (or $DATA_ROOT) is required: it must contain raw/mucs2021")
    work = args.work or str(Path(args.data_root) / "interim" / "rvc_gate")
    jobs = pd.read_csv(args.jobs)
    results = run_gate(
        jobs,
        args.clips,
        work,
        args.data_root,
        conditions=tuple(args.condition) if args.condition else CONDITIONS,
        out_dir=args.out_dir,
    )
    print("\n=== CM02 shortcut gate ===")
    for condition, result in results.items():
        print(
            f"  {condition:11s} EER {result['eer']:.2%}  AUC {result['auc']:.4f}  -> {result['verdict'].split(':')[0]}"
        )


if __name__ == "__main__":
    main()
