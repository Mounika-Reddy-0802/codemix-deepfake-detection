"""Scoring manifests for the two attacks no detector has been measured on yet.

Every detector number so far is against XTTS-v2 -- the tool the adapter trained
on. CM02 (RVC voice conversion) and CM04 (Tortoise, the held-out tool) both exist
and are QA-screened, but neither has been scored. This module builds the two
real-vs-fake manifests that ``src.training.evaluate`` needs to score them.

The one design rule that matters: **the real clips come from the same speakers
as the fakes.** Otherwise a detector can separate the classes by *who* is talking
rather than by whether the voice is cloned, and the EER measures the wrong thing.

- **CM04** fakes are the 15 eval-pool speakers, so the real side is the eval-pool
  bonafide already in ``codemix_eval.csv`` (the same 15 speakers).
- **CM02** fakes are conversions of real train-pool recordings, so the real side
  is **the exact source segments those conversions were made from** -- the paired
  design ``rvc_gate`` uses (P-022). Same words, same room; only the conversion
  differs.

Only QA-passed fakes are used (``ok == True`` in the committed QA CSVs), as for
every other result: a silent or truncated clip is a generator failure, not an
attack. Paths are written ``${DATA_ROOT}``-relative so no machine path enters git.

CM04 is test-only by rule (``tests/test_splits.py``). Its manifest carries
``split == "eval"`` and must never be passed to a training entry point.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.paths import DATA_ROOT_TOKEN, resolve

COLUMNS = ["filepath", "label", "language", "speaker", "source", "tool", "condition", "split"]


class ScoringManifestError(AssertionError):
    """Raised when a scoring manifest would compare the wrong speakers or clips."""


def portable(rel_dir: str, name: str) -> str:
    """``${DATA_ROOT}/<rel_dir>/<name>`` -- resolvable anywhere, machine-free."""
    return f"{DATA_ROOT_TOKEN}/{rel_dir.strip('/')}/{Path(str(name)).name}"


def qa_passed(qa: pd.DataFrame) -> pd.DataFrame:
    """Rows of a ``generation_qa`` CSV that passed the screen."""
    ok = qa["ok"].astype(str).str.lower().isin({"true", "1"})
    return qa[ok].reset_index(drop=True)


def _pool(pools: pd.DataFrame, name: str) -> set[str]:
    return set(pools.loc[pools["pool"] == name, "speaker"].astype(str))


def _rows(paths, speakers, label: str, tool: str, source: str) -> pd.DataFrame:
    frame = pd.DataFrame({"filepath": list(paths), "speaker": [str(s) for s in speakers]})
    frame["label"] = label
    frame["language"] = "hi-en"
    frame["source"] = source
    frame["tool"] = tool
    frame["condition"] = "clean"
    frame["split"] = "eval"
    return frame[COLUMNS]


def cm04_manifest(
    eval_manifest: pd.DataFrame,
    qa: pd.DataFrame,
    pools: pd.DataFrame,
    bonafide_dir: str = "lora_bundle/clips",
    spoof_dir: str = "cm04",
) -> pd.DataFrame:
    """Eval-pool bonafide vs QA-passed Tortoise clips of the same speakers."""
    eval_pool = _pool(pools, "eval")
    bona = eval_manifest[eval_manifest["label"].str.lower() == "bonafide"]
    bona = bona[bona["speaker"].astype(str).isin(eval_pool)]
    spoof = qa_passed(qa)

    spoof_speakers = set(spoof["speaker"].astype(str))
    if not spoof_speakers <= eval_pool:
        raise ScoringManifestError(
            f"CM04 clips outside the eval pool: {sorted(spoof_speakers - eval_pool)}"
        )
    missing = spoof_speakers - set(bona["speaker"].astype(str))
    if missing:
        raise ScoringManifestError(f"no real clips for CM04 speaker(s) {sorted(missing)}")

    bona_rows = _rows(
        (portable(bonafide_dir, p) for p in bona["filepath"]),
        bona["speaker"],
        "bonafide",
        "none",
        "mucs2021",
    )
    spoof_rows = _rows(
        (portable(spoof_dir, c) for c in spoof["clip"]),
        spoof["speaker"],
        "spoof",
        "tortoise",
        "cm04",
    )
    return pd.concat([bona_rows, spoof_rows], ignore_index=True)


def cm02_manifest(
    jobs: pd.DataFrame,
    qa: pd.DataFrame,
    pools: pd.DataFrame,
    bonafide_dir: str = "interim/rvc_gate/bonafide_source",
    spoof_dir: str = "rvc_cm02/rvc_converted_wavs/rvc_outputs",
) -> pd.DataFrame:
    """The exact source segments vs their QA-passed RVC conversions (paired)."""
    train_pool = _pool(pools, "train")
    sources = jobs.drop_duplicates("source_utt_id")
    spoof = qa_passed(qa)

    used = set(sources["source_speaker"].astype(str)) | set(spoof["speaker"].astype(str))
    if not used <= train_pool:
        raise ScoringManifestError(
            f"CM02 speakers outside the train pool: {sorted(used - train_pool)}"
        )

    bona_rows = _rows(
        (portable(bonafide_dir, f"{u}.wav") for u in sources["source_utt_id"]),
        sources["source_speaker"],
        "bonafide",
        "none",
        "mucs2021",
    )
    spoof_rows = _rows(
        (portable(spoof_dir, c) for c in spoof["clip"]), spoof["speaker"], "spoof", "rvc", "cm02"
    )
    return pd.concat([bona_rows, spoof_rows], ignore_index=True)


def gate_split(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit/score halves for the shortcut gate, disjoint by speaker.

    The low-level-cue gate fits on one half and scores the other; if a speaker
    sat on both sides the regression could key on that speaker's recording
    channel. Alternating over sorted ids (not shuffling) makes the split identical
    on every machine (P-016), matching ``rvc_gate.split_source_speakers``.
    """
    speakers = sorted(set(manifest["speaker"].astype(str)))
    fit = set(speakers[0::2])
    mask = manifest["speaker"].astype(str).isin(fit)
    return manifest[mask].reset_index(drop=True), manifest[~mask].reset_index(drop=True)


def missing_files(manifest: pd.DataFrame, data_root: str) -> list[str]:
    """Manifest rows whose audio does not exist under ``data_root``."""
    return [p for p in manifest["filepath"] if not Path(resolve(p, data_root)).is_file()]


def summary(manifest: pd.DataFrame) -> dict:
    """Counts for the log and the results doc."""
    return {
        "rows": int(len(manifest)),
        "bonafide": int((manifest["label"] == "bonafide").sum()),
        "spoof": int((manifest["label"] == "spoof").sum()),
        "speakers": int(manifest["speaker"].nunique()),
        "tools": sorted(manifest["tool"].unique()),
    }


def main() -> None:
    """CLI: ``python -m src.data.scoring_manifests --attack cm04 --data-root C:/dfdata``."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Build a real-vs-fake scoring manifest")
    parser.add_argument("--attack", required=True, choices=("cm02", "cm04"))
    parser.add_argument("--data-root", required=True, help="root the ${DATA_ROOT} paths resolve to")
    parser.add_argument("--pools", default="data/manifests/speaker_pools.csv")
    parser.add_argument("--out", default=None, help="default: data/manifests/score_<attack>.csv")
    parser.add_argument("--spoof-dir", default=None, help="fakes dir, relative to the data root")
    parser.add_argument(
        "--gate-split",
        action="store_true",
        help="also write speaker-disjoint _fit/_score halves for the shortcut gate",
    )
    args = parser.parse_args()

    pools = pd.read_csv(args.pools)
    if args.attack == "cm04":
        manifest = cm04_manifest(
            pd.read_csv("data/manifests/codemix_eval.csv"),
            pd.read_csv("docs/qa/heldout_generation_qa.csv"),
            pools,
            **({"spoof_dir": args.spoof_dir} if args.spoof_dir else {}),
        )
    else:
        manifest = cm02_manifest(
            pd.read_csv("data/manifests/rvc_generation_jobs.csv"),
            pd.read_csv("docs/qa/rvc_generation_qa.csv"),
            pools,
            **({"spoof_dir": args.spoof_dir} if args.spoof_dir else {}),
        )

    absent = missing_files(manifest, args.data_root)
    print(json.dumps(summary(manifest), indent=2))
    if absent:
        print(f"{len(absent)} file(s) missing under {args.data_root}, e.g. {absent[0]}")
        raise SystemExit(1)
    out = args.out or f"data/manifests/score_{args.attack}.csv"
    manifest.to_csv(out, index=False)
    print(f"all {len(manifest)} files present -> wrote {out}")
    if args.gate_split:
        fit, score = gate_split(manifest)
        stem = out[: -len(".csv")]
        fit.to_csv(f"{stem}_fit.csv", index=False)
        score.to_csv(f"{stem}_score.csv", index=False)
        print(
            f"gate split: fit {len(fit)} ({fit.speaker.nunique()} spk) / "
            f"score {len(score)} ({score.speaker.nunique()} spk)"
        )


if __name__ == "__main__":
    main()
