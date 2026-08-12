"""Tests for the HiACC quarantine audit (Week 3, W3-T2, owner L).

The audit exists because the download script's ``child|kid|minor`` folder regex is
a guess about HiACC's layout. The cases below pin the two outcomes that matter:
child-looking material still living in the corpus, and the quieter failure where
the regex matches nothing because the split is by manifest column.

Built on synthetic trees, so no corpus download is needed to run these.
"""

from pathlib import Path

from src.data import quarantine as q


def _touch(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _clean_tree(root: Path) -> None:
    """Adults live; children already quarantined."""
    _touch(root / "adult" / "spk001" / "utt_0001.wav")
    _touch(root / "adult" / "spk002" / "utt_0002.wav")
    _touch(root / q.QUARANTINE_DIR / "child_07" / "utt_9001.wav")


# --------------------------------------------------------------------------- #
# Pattern matching
# --------------------------------------------------------------------------- #
def test_child_words_are_matched() -> None:
    # Underscore-separated names are the common corpus convention, and the reason
    # a plain \b anchor does not work here.
    for name in ("child_07", "children", "kid_3", "minor_speakers", "juvenile", "spk_child"):
        assert q._matches_child(name) is not None, name


def test_obvious_false_positives_are_not_matched() -> None:
    for name in ("kidney_study", "adult", "spk001", "childish", "minority_report"):
        assert q._matches_child(name) is None, name


# --------------------------------------------------------------------------- #
# Tree scanning
# --------------------------------------------------------------------------- #
def test_missing_root_is_reported_not_crashed(tmp_path: Path) -> None:
    result = q.audit(str(tmp_path / "nope"))
    assert result.exists is False
    assert result.total_audio == 0


def test_quarantined_child_folder_is_counted(tmp_path: Path) -> None:
    _clean_tree(tmp_path)
    result = q.audit(str(tmp_path))
    assert result.quarantine_exists is True
    assert result.total_audio == 3
    assert result.quarantined_audio == 1


def test_clean_tree_has_no_danger_findings(tmp_path: Path) -> None:
    _clean_tree(tmp_path)
    assert q.audit(str(tmp_path)).unquarantined_findings == []


def test_child_folder_outside_quarantine_is_flagged(tmp_path: Path) -> None:
    _clean_tree(tmp_path)
    _touch(tmp_path / "adult" / "child_12" / "utt_5.wav")
    danger = q.audit(str(tmp_path)).unquarantined_findings
    assert danger
    assert any("child_12" in f.path for f in danger)


def test_child_named_audio_file_is_flagged(tmp_path: Path) -> None:
    _clean_tree(tmp_path)
    _touch(tmp_path / "adult" / "spk001" / "child_utt.wav")
    danger = q.audit(str(tmp_path)).unquarantined_findings
    assert any(f.kind == "audio" for f in danger)


# --------------------------------------------------------------------------- #
# The quiet failure: split by manifest, not by folder
# --------------------------------------------------------------------------- #
def test_age_column_in_a_manifest_is_detected(tmp_path: Path) -> None:
    _touch(tmp_path / "adult" / "spk001" / "utt_0001.wav")
    _touch(
        tmp_path / "metadata.csv",
        "speaker,age_group,file\nspk001,adult,utt_0001.wav\nspk002,child,utt_0002.wav\n",
    )
    result = q.audit(str(tmp_path))
    assert "metadata.csv" in result.manifest_columns
    assert "age_group" in result.manifest_columns["metadata.csv"]


def test_nothing_quarantined_plus_age_column_demands_manual_split(tmp_path: Path) -> None:
    # The exact case the plan warns about: regex found no folders, so the
    # quarantine looks clean while the real split sits in a column.
    _touch(tmp_path / "audio" / "utt_0001.wav")
    _touch(tmp_path / "metadata.csv", "speaker,age_group\nspk001,adult\nspk002,child\n")
    result = q.audit(str(tmp_path))
    assert result.quarantined_audio == 0
    assert result.needs_manual_split is True


def test_manual_split_not_flagged_when_children_are_quarantined(tmp_path: Path) -> None:
    _clean_tree(tmp_path)
    _touch(tmp_path / "metadata.csv", "speaker,age_group\nspk001,adult\n")
    assert q.audit(str(tmp_path)).needs_manual_split is False


def test_child_values_in_a_manifest_are_reported(tmp_path: Path) -> None:
    _touch(tmp_path / "audio" / "a.wav")
    _touch(tmp_path / "meta.csv", "speaker,category\ns1,adult\ns2,child\n")
    findings = q.audit(str(tmp_path)).findings
    assert any(f.kind == "manifest" and "category" in f.detail for f in findings)


def test_unreadable_manifest_does_not_crash_the_audit(tmp_path: Path) -> None:
    _touch(tmp_path / "audio" / "a.wav")
    (tmp_path / "broken.csv").write_bytes(b"\x00\x01\x02 not,a;csv\x00")
    assert q.audit(str(tmp_path)).total_audio == 1


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def test_report_never_claims_a_pass(tmp_path: Path) -> None:
    _clean_tree(tmp_path)
    report = q.render_report(q.audit(str(tmp_path)))
    assert "This report is not a pass" in report
    assert "Checked by:" in report  # a human signs it


def test_report_lists_items_outside_quarantine(tmp_path: Path) -> None:
    _clean_tree(tmp_path)
    _touch(tmp_path / "adult" / "child_12" / "utt_5.wav")
    report = q.render_report(q.audit(str(tmp_path)))
    assert "OUTSIDE quarantine" in report
    assert "child_12" in report


def test_report_tells_you_to_download_when_root_is_missing(tmp_path: Path) -> None:
    report = q.render_report(q.audit(str(tmp_path / "absent")))
    assert "not on disk yet" in report
