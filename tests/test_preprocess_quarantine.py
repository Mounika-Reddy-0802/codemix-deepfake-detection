"""Quarantine tests for the preprocessing walk (Week 3, W3-T2, owner L).

The HiACC child folder is moved to ``_EXCLUDED_children/`` at download time, but
``preprocess_dir`` used to walk the corpus root with ``rglob`` and would have
pulled those files straight back in. These tests pin the exclusion so the hole
cannot reopen.

No audio is read: the walk is tested on empty files, so this runs in CI.
"""

from pathlib import Path

from src.data import preprocess as pp


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _hiacc_tree(root: Path) -> None:
    """A HiACC-shaped tree: adult speakers plus a quarantined child folder."""
    _touch(root / "adult" / "spk001" / "utt_0001.wav")
    _touch(root / "adult" / "spk002" / "utt_0002.flac")
    _touch(root / "_EXCLUDED_children" / "child_07" / "utt_9001.wav")
    _touch(root / "_EXCLUDED_children" / "README.md")


def test_child_audio_is_not_walked(tmp_path: Path) -> None:
    _hiacc_tree(tmp_path)
    found = pp.audio_files(str(tmp_path))
    assert len(found) == 2
    assert not any("_EXCLUDED_children" in str(p) for p in found)


def test_adult_audio_is_still_walked(tmp_path: Path) -> None:
    _hiacc_tree(tmp_path)
    names = {Path(p).name for p in pp.audio_files(str(tmp_path))}
    assert names == {"utt_0001.wav", "utt_0002.flac"}


def test_is_excluded_matches_at_any_depth() -> None:
    assert pp.is_excluded("data/raw/hiacc/_EXCLUDED_children/child_07/a.wav")
    assert pp.is_excluded("_EXCLUDED_children/a.wav")
    assert not pp.is_excluded("data/raw/hiacc/adult/spk001/a.wav")


def test_exclusion_is_exact_not_substring() -> None:
    # A legitimate folder whose name merely contains the word must still be read.
    assert not pp.is_excluded("data/raw/hiacc/adult_EXCLUDED_children_notes/a.wav")


def test_nested_quarantine_is_excluded(tmp_path: Path) -> None:
    _touch(tmp_path / "corpus" / "_EXCLUDED_children" / "deep" / "nest" / "a.wav")
    _touch(tmp_path / "corpus" / "ok.wav")
    found = pp.audio_files(str(tmp_path))
    assert [Path(p).name for p in found] == ["ok.wav"]


def test_preprocess_dir_reports_zero_for_a_quarantine_only_tree(tmp_path: Path) -> None:
    _touch(tmp_path / "_EXCLUDED_children" / "child_01" / "a.wav")
    assert pp.preprocess_dir(str(tmp_path), str(tmp_path / "out")) == 0


def test_excluded_names_are_configurable(tmp_path: Path) -> None:
    _touch(tmp_path / "private" / "a.wav")
    _touch(tmp_path / "ok.wav")
    found = pp.audio_files(str(tmp_path), excluded=("private",))
    assert [Path(p).name for p in found] == ["ok.wav"]
