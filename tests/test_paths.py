"""Portable-path tests: one manifest, two machines.

The concrete case these protect is the real one — ``clip_index.csv`` was written
on the dev laptop full of ``C:/dfdata/raw/...`` paths, and the same rows have to
load in a Colab session where the data sits at ``/content/data``. Resolution is
structural (no filesystem probing), so every case here is exact.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.utils import paths

WINDOWS_CLIP = "C:/dfdata/raw/mucs2021/train/audio/rec_001.wav"
COLAB_ROOT = "/content/data"


@pytest.fixture(autouse=True)
def _no_env_override(monkeypatch):
    """Keep a developer's own $DATA_ROOT out of the assertions."""
    monkeypatch.delenv(paths.DATA_ROOT_ENV, raising=False)


def test_default_root_is_the_repo_data_dir() -> None:
    assert paths.data_root().as_posix() == paths.DEFAULT_DATA_ROOT


def test_env_var_sets_the_root(monkeypatch) -> None:
    monkeypatch.setenv(paths.DATA_ROOT_ENV, COLAB_ROOT)
    assert paths.data_root().as_posix() == COLAB_ROOT


def test_argument_beats_the_env_var(monkeypatch) -> None:
    monkeypatch.setenv(paths.DATA_ROOT_ENV, COLAB_ROOT)
    assert paths.data_root("C:/dfdata").as_posix() == "C:/dfdata"


# --------------------------------------------------------------------------- #
# The headline case: a laptop-written manifest read on Colab
# --------------------------------------------------------------------------- #
def test_foreign_absolute_path_is_rebased_on_the_local_root() -> None:
    resolved = paths.resolve(WINDOWS_CLIP, COLAB_ROOT).replace("\\", "/")
    assert resolved == "/content/data/raw/mucs2021/train/audio/rec_001.wav"


def test_backslash_paths_resolve_too() -> None:
    windows_style = r"C:\dfdata\raw\hiacc\Corpus\adult\audio\AD09001.wav"
    resolved = paths.resolve(windows_style, COLAB_ROOT).replace("\\", "/")
    assert resolved == "/content/data/raw/hiacc/Corpus/adult/audio/AD09001.wav"


def test_path_already_under_the_local_root_is_untouched() -> None:
    local = "/content/data/raw/mucs2021/train/x.wav"
    assert paths.resolve(local, COLAB_ROOT).replace("\\", "/") == local


def test_token_paths_substitute_the_local_root() -> None:
    stored = f"{paths.DATA_ROOT_TOKEN}/generated/xtts_v2/clip.wav"
    assert paths.resolve(stored, COLAB_ROOT).replace("\\", "/") == (
        "/content/data/generated/xtts_v2/clip.wav"
    )
    assert paths.resolve(stored, "C:/dfdata").replace("\\", "/") == (
        "C:/dfdata/generated/xtts_v2/clip.wav"
    )


def test_repo_relative_paths_are_left_alone() -> None:
    assert paths.resolve("data/manifests/speaker_pools.csv", COLAB_ROOT).replace("\\", "/") == (
        "data/manifests/speaker_pools.csv"
    )


def test_absolute_path_with_no_anchor_is_returned_unchanged() -> None:
    # Nothing to rebase on: better to hand the caller back the original name so the
    # resulting "file not found" names the path they actually wrote.
    stray = "/somewhere/else/clip.wav"
    assert paths.resolve(stray, COLAB_ROOT).replace("\\", "/") == stray


# --------------------------------------------------------------------------- #
# Writing portable manifests
# --------------------------------------------------------------------------- #
def test_portable_strips_the_local_root() -> None:
    assert paths.portable(WINDOWS_CLIP, "C:/dfdata") == (
        f"{paths.DATA_ROOT_TOKEN}/raw/mucs2021/train/audio/rec_001.wav"
    )


def test_portable_falls_back_to_the_layout_anchor() -> None:
    # Root does not match, but "raw" still marks where the portable part starts.
    assert paths.portable(WINDOWS_CLIP, "/some/other/root") == (
        f"{paths.DATA_ROOT_TOKEN}/raw/mucs2021/train/audio/rec_001.wav"
    )


def test_portable_and_resolve_round_trip() -> None:
    stored = paths.portable(WINDOWS_CLIP, "C:/dfdata")
    assert paths.resolve(stored, COLAB_ROOT).replace("\\", "/") == (
        "/content/data/raw/mucs2021/train/audio/rec_001.wav"
    )


def test_non_data_path_is_left_portable_as_is() -> None:
    assert paths.portable("configs/train_baseline.yaml", "C:/dfdata") == (
        "configs/train_baseline.yaml"
    )


# --------------------------------------------------------------------------- #
# Frame helper
# --------------------------------------------------------------------------- #
def test_resolve_column_rewrites_without_mutating_the_input() -> None:
    frame = pd.DataFrame({"filepath": [WINDOWS_CLIP], "label": ["bonafide"]})
    out = paths.resolve_column(frame, "filepath", COLAB_ROOT)
    assert out.loc[0, "filepath"].replace("\\", "/").startswith(COLAB_ROOT)
    assert frame.loc[0, "filepath"] == WINDOWS_CLIP  # original untouched
    assert list(out.columns) == ["filepath", "label"]


def test_resolve_column_is_a_no_op_when_the_column_is_absent() -> None:
    frame = pd.DataFrame({"speaker": ["spk1"]})
    assert paths.resolve_column(frame, "filepath", COLAB_ROOT) is frame
