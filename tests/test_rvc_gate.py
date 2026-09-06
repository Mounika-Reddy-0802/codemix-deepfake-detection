"""Tests for the CM02 shortcut-gate pairing.

The gate itself (``lowlevel_cue``) is already tested. What is new here is the
*pairing* -- bonafide source segments against their own RVC conversions -- and the
three properties that make that pairing a fair test rather than a rigged one:

- one bonafide row per **unique** source clip, however many targets reused it;
- fit and score **disjoint by source speaker** on both classes, and deterministic;
- every RVC conversion present on the spoof side, none silently dropped.

Pure pandas and tmp_path; no audio, no regression fit.
"""

import pandas as pd

from src.data import rvc_gate


def _jobs() -> pd.DataFrame:
    rows = []
    job = 0
    # 4 source speakers, 3 clips each; every clip converted into 2 targets -> 24 jobs
    for s in range(4):
        for c in range(3):
            for t in ("tA", "tB"):
                job += 1
                rows.append(
                    {
                        "job_id": job,
                        "target_speaker": t,
                        "source_speaker": f"s{s}",
                        "source_utt_id": f"s{s}_{c:03d}",
                        "source_wav": f"C:\\dfdata\\raw\\mucs2021\\train\\s{s}.wav",
                        "start_seconds": c * 5.0,
                        "end_seconds": c * 5.0 + 4.0,
                        "duration_seconds": 4.0,
                        "tool": "rvc",
                        "output_path": f"/kaggle/working/rvc_outputs/rvc_{t}_{job:05d}.wav",
                    }
                )
    return pd.DataFrame(rows)


def test_one_bonafide_row_per_unique_source_clip() -> None:
    sources = rvc_gate.unique_sources(_jobs())
    assert len(sources) == 12  # 4 speakers x 3 clips, not 24
    assert sources["source_utt_id"].is_unique


def test_split_is_disjoint_deterministic_and_balanced() -> None:
    fit, score = rvc_gate.split_source_speakers(["s3", "s1", "s0", "s2"])
    assert fit == {"s0", "s2"} and score == {"s1", "s3"}
    assert fit.isdisjoint(score)
    assert rvc_gate.split_source_speakers(["s0", "s1", "s2", "s3"]) == (fit, score)


def test_manifest_has_every_source_and_every_conversion() -> None:
    m = rvc_gate.paired_manifest(_jobs(), "/clips", "/sources")
    assert (m.label == "bonafide").sum() == 12
    assert (m.label == "spoof").sum() == 24
    assert set(m.columns) == {"filepath", "label", "speaker", "tool", "split"}


def test_no_source_speaker_sits_on_both_sides_of_the_split() -> None:
    m = rvc_gate.paired_manifest(_jobs(), "/clips", "/sources")
    for label in ("bonafide", "spoof"):
        side = m[m.label == label]
        fit_spk = set(side[side.split == "fit"].speaker)
        score_spk = set(side[side.split == "score"].speaker)
        assert fit_spk and score_spk
        assert fit_spk.isdisjoint(score_spk), label


def test_both_classes_use_the_same_speaker_split() -> None:
    """A conversion is scored with the recording it came from, never against it."""
    m = rvc_gate.paired_manifest(_jobs(), "/clips", "/sources")
    by_speaker = m.groupby("speaker")["split"].nunique()
    assert (by_speaker == 1).all()


def test_paths_are_built_from_basenames_under_the_given_dirs() -> None:
    m = rvc_gate.paired_manifest(_jobs(), "/clips", "/sources")
    spoof = m[m.label == "spoof"].filepath.iloc[0].replace("\\", "/")
    bona = m[m.label == "bonafide"].filepath.iloc[0].replace("\\", "/")
    assert spoof.startswith("/clips/rvc_") and spoof.endswith(".wav")
    assert bona == "/sources/s0_000.wav"


def test_an_explicit_fit_set_is_honoured() -> None:
    m = rvc_gate.paired_manifest(_jobs(), "/clips", "/sources", fit_speakers={"s0"})
    assert set(m[m.split == "fit"].speaker) == {"s0"}
    assert set(m[m.split == "score"].speaker) == {"s1", "s2", "s3"}


def test_missing_files_are_kept_in_the_manifest_not_dropped() -> None:
    """Dropping one class's misses silently would bias the check; the runner reports them."""
    m = rvc_gate.paired_manifest(_jobs(), "/nowhere", "/nowhere")
    assert len(m) == 36
