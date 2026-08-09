"""CRITICAL-PATH tests (Week 2/4, owner M): anti-leakage split validation.

Speaker leakage between train and eval is the single bug that could invalidate the
whole paper. These tests exercise ``src/data/build_manifests.py`` on synthetic
manifests (no real audio needed), so they run in CI and gate every training launch.

Checklist (tech-stack Section 4): 1) speaker-disjoint train/eval incl. clones,
2) no Tortoise in training, 3) no eval-only corpora in training, 4) no HiACC child
audio anywhere. Check 5 (transcript near-duplicate) needs transcript metadata and
lands in Week 4.
"""

import pandas as pd
import pytest

from src.data import build_manifests as bm


def _row(fp, label, lang, spk, source, tool, split, condition="clean"):
    return {
        "filepath": fp,
        "label": label,
        "language": lang,
        "speaker": spk,
        "source": source,
        "tool": tool,
        "condition": condition,
        "split": split,
    }


def _clean_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row("a.wav", "bonafide", "en", "spk1", "asvspoof2019_la", "none", "train"),
            _row("b.wav", "spoof", "en", "spk1", "asvspoof2019_la", "none", "train"),
            _row("c.wav", "bonafide", "hi-en", "spk2", "mucs2021", "none", "dev"),
            _row("d.wav", "spoof", "hi-en", "spk2", "xtts_v2", "xtts_v2", "dev"),
            _row("e.wav", "bonafide", "hi-en", "spk3", "hiacc", "none", "eval"),
            _row("f.wav", "spoof", "hi-en", "spk3", "tortoise", "tortoise", "eval"),
        ]
    )


def test_clean_manifest_passes_all_checks() -> None:
    bm.run_all_checks(_clean_manifest())  # should not raise


def test_speaker_overlap_is_detected() -> None:
    df = _clean_manifest()
    # Leak spk1 (a training speaker) into eval via a cloned voice.
    df = pd.concat(
        [df, pd.DataFrame([_row("g.wav", "spoof", "en", "spk1", "xtts_v2", "xtts_v2", "eval")])],
        ignore_index=True,
    )
    assert bm.speaker_overlap(df) == {"spk1"}
    with pytest.raises(bm.LeakageError):
        bm.check_speaker_disjoint(df)


def test_tortoise_in_training_is_rejected() -> None:
    df = _clean_manifest()
    df.loc[df["filepath"] == "f.wav", "split"] = "train"  # move Tortoise into train
    with pytest.raises(bm.LeakageError):
        bm.check_no_heldout_tool_in_training(df)


def test_eval_only_source_in_training_is_rejected() -> None:
    df = _clean_manifest()
    df = pd.concat(
        [df, pd.DataFrame([_row("h.wav", "spoof", "hi", "spk9", "indicsynth", "vits", "train")])],
        ignore_index=True,
    )
    with pytest.raises(bm.LeakageError):
        bm.check_no_eval_only_sources_in_training(df)


def test_child_audio_anywhere_is_rejected() -> None:
    df = _clean_manifest()
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [_row("k.wav", "bonafide", "hi-en", "child_07", "hiacc_child", "none", "eval")]
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(bm.LeakageError):
        bm.check_no_child_audio(df)


def test_assign_split_by_speaker_is_disjoint() -> None:
    rows = []
    for s in range(20):
        for i in range(3):
            rows.append(_row(f"s{s}_{i}.wav", "bonafide", "en", f"spk{s}", "mucs2021", "none", "?"))
    df = bm.assign_split_by_speaker(pd.DataFrame(rows), seed=7)
    # No speaker may appear in more than one split.
    per_speaker_splits = df.groupby("speaker")["split"].nunique()
    assert (per_speaker_splits == 1).all()
    assert bm.speaker_overlap(df) == set()


def test_carve_pools_are_disjoint_and_cover_all() -> None:
    speakers = [f"spk{i}" for i in range(50)]
    evalp, adapt = bm.carve_pools(speakers, adaptation_frac=0.3, seed=1)
    assert set(evalp) & set(adapt) == set()
    assert set(evalp) | set(adapt) == set(speakers)
    assert abs(len(adapt) - 15) <= 1


def test_build_manifests_module_is_importable() -> None:
    import importlib

    mod = importlib.import_module("src.data.build_manifests")
    assert mod.__doc__ is not None
