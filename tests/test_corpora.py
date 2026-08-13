"""Tests for corpus-aware clip indexing (Week 3, W3-T2/W3-T4, owner L).

Speaker identity drives every disjointness guarantee in the project, and neither
downloaded corpus stores it where a path-guess would find it. These tests pin the
two real layouts:

- **MUCS** is Kaldi-style: flat recordings plus ``segments``/``utt2spk``/``wav.scp``.
  Guessing from the path yields the speaker ``"train"`` for all 521 files.
- **HiACC** encodes the speaker as a filename prefix (``AD09001.wav`` -> ``AD09``)
  with metadata in ``speaker_info.csv``.

Small synthetic corpora, no audio decoded.
"""

from pathlib import Path

import pytest

from src.data import corpora as co


# --------------------------------------------------------------------------- #
# MUCS fixtures
# --------------------------------------------------------------------------- #
def _mucs(root: Path, split: str = "train") -> Path:
    base = root / split
    transcripts = base / "transcripts"
    transcripts.mkdir(parents=True)
    (base / "rec_A.wav").write_bytes(b"")
    (base / "rec_B.wav").write_bytes(b"")

    (transcripts / "segments").write_text(
        "100051_recA_0000 recA 0.0 9.0\n"
        "100051_recA_0001 recA 9.0 12.5\n"
        "100099_recB_0000 recB 0.0 4.0\n",
        encoding="utf-8",
    )
    (transcripts / "utt2spk").write_text(
        "100051_recA_0000 100051\n100051_recA_0001 100051\n100099_recB_0000 100099\n",
        encoding="utf-8",
    )
    (transcripts / "wav.scp").write_text("recA rec_A.wav\nrecB rec_B.wav\n", encoding="utf-8")
    (transcripts / "text").write_text(
        "100051_recA_0000 mujhe kal bank jaana hai\n"
        "100051_recA_0001 aaj meeting hai office mein\n"
        "100099_recB_0000 please transfer the amount\n",
        encoding="utf-8",
    )
    return root


def test_mucs_uses_utt2spk_not_the_path(tmp_path: Path) -> None:
    # The bug this exists to prevent: every MUCS file lives in `train/`, so a
    # parent-directory guess makes all 520 speakers into one speaker called
    # "train".
    frame = co.index_mucs(str(_mucs(tmp_path)))
    assert sorted(frame["speaker"].unique()) == ["100051", "100099"]
    assert "train" not in set(frame["speaker"])


def test_mucs_row_is_an_utterance_not_a_file(tmp_path: Path) -> None:
    frame = co.index_mucs(str(_mucs(tmp_path)))
    assert len(frame) == 3  # 3 utterances across 2 recordings
    assert frame["wav_path"].nunique() == 2


def test_mucs_carries_the_time_span(tmp_path: Path) -> None:
    frame = co.index_mucs(str(_mucs(tmp_path))).set_index("utt_id")
    row = frame.loc["100051_recA_0001"]
    assert row["start_seconds"] == 9.0
    assert row["end_seconds"] == 12.5
    assert row["duration_seconds"] == pytest.approx(3.5)


def test_mucs_keeps_transcripts_with_spaces(tmp_path: Path) -> None:
    frame = co.index_mucs(str(_mucs(tmp_path))).set_index("utt_id")
    assert frame.loc["100051_recA_0000", "transcript"] == "mujhe kal bank jaana hai"


def test_mucs_drops_utterances_it_cannot_attribute(tmp_path: Path) -> None:
    root = _mucs(tmp_path)
    segments = root / "train" / "transcripts" / "segments"
    segments.write_text(segments.read_text(encoding="utf-8") + "orphan_recZ_0 recZ 0.0 2.0\n")
    frame = co.index_mucs(str(root))
    assert "orphan_recZ_0" not in set(frame["utt_id"])


def test_mucs_without_kaldi_tables_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "train" / "transcripts").mkdir(parents=True)
    with pytest.raises(co.CorpusError):
        co.index_mucs(str(tmp_path))


def test_kaldi_table_splits_only_the_key(tmp_path: Path) -> None:
    path = tmp_path / "text"
    path.write_text("utt1 several words follow here\n", encoding="utf-8")
    assert co.read_kaldi_table(str(path))["utt1"] == "several words follow here"


def test_kaldi_table_skips_blank_and_keyless_lines(tmp_path: Path) -> None:
    path = tmp_path / "t"
    path.write_text("\na b\nlonelykey\n\n", encoding="utf-8")
    assert co.read_kaldi_table(str(path)) == {"a": "b"}


# --------------------------------------------------------------------------- #
# HiACC fixtures
# --------------------------------------------------------------------------- #
def _hiacc(root: Path, include_undocumented: bool = False) -> Path:
    base = root / "Corpus" / "adult"
    audio = base / "audio" / "train_split"
    audio.mkdir(parents=True)
    (base / "metadata").mkdir(parents=True)

    for name in ("AD09001.wav", "AD09002.wav", "AD13001.wav"):
        (audio / name).write_bytes(b"")
    if include_undocumented:
        (audio / "AD63012.wav").write_bytes(b"")

    (base / "metadata" / "speaker_info.csv").write_text(
        "PID,Gender,Age,L1\nAD09,F,36,Hindi\nAD13,M,35,Hindi\nAD65,F,29,Hindi\n",
        encoding="utf-8",
    )
    (base / "metadata" / "sentence_stats.csv").write_text(
        "audio,sentence,CMI,duration_sec,code_switch_count\n"
        "AD09001.wav,So the question is what is your favourite festival,0.0,7.9,0\n"
        "AD09002.wav,मेरा favourite festival Diwali hai,19.05,10.36,4\n"
        "AD13001.wav,kal office mein meeting thi,25.0,5.2,2\n",
        encoding="utf-8",
    )
    # The quarantined children folder must never be indexed.
    (root / "_EXCLUDED_children" / "children").mkdir(parents=True)
    (root / "_EXCLUDED_children" / "children" / "CH01001.wav").write_bytes(b"")
    return root


def test_hiacc_speaker_comes_from_the_filename_prefix(tmp_path: Path) -> None:
    frame = co.index_hiacc(str(_hiacc(tmp_path)))
    assert sorted(frame["speaker"].unique()) == ["AD09", "AD13"]
    assert "train_split" not in set(frame["speaker"])


def test_hiacc_longest_pid_wins() -> None:
    # A short PID must not shadow a longer one that shares its prefix.
    assert co.hiacc_speaker("AD130001.wav", {"AD13", "AD1"}) == "AD13"


def test_hiacc_unknown_prefix_returns_none() -> None:
    assert co.hiacc_speaker("ZZ99001.wav", {"AD09"}) is None


def test_hiacc_joins_transcript_and_cmi(tmp_path: Path) -> None:
    frame = co.index_hiacc(str(_hiacc(tmp_path))).set_index("utt_id")
    assert frame.loc["AD09002", "cmi"] == pytest.approx(19.05)
    assert frame.loc["AD09002", "code_switch_count"] == 4
    assert "Diwali" in frame.loc["AD09002", "transcript"]


def test_hiacc_undocumented_speaker_is_kept_but_flagged(tmp_path: Path) -> None:
    # AD63 exists on the real corpus with 119 clips and no speaker_info row.
    # Dropping it silently would shrink the corpus without anyone noticing.
    frame = co.index_hiacc(str(_hiacc(tmp_path, include_undocumented=True)))
    assert "AD63" in set(frame["speaker"])
    assert not frame.loc[frame["speaker"] == "AD63", "has_metadata"].any()
    assert frame.loc[frame["speaker"] == "AD09", "has_metadata"].all()


def test_hiacc_never_indexes_quarantined_children(tmp_path: Path) -> None:
    frame = co.index_hiacc(str(_hiacc(tmp_path)))
    assert not any("children" in str(p).lower() for p in frame["wav_path"])
    assert not any(str(s).startswith("CH") for s in frame["speaker"])


def test_hiacc_refuses_the_children_category(tmp_path: Path) -> None:
    with pytest.raises(co.CorpusError, match="only 'adult'"):
        co.index_hiacc(str(_hiacc(tmp_path)), category="children")


def test_hiacc_missing_audio_dir_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(co.CorpusError):
        co.index_hiacc(str(tmp_path))


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def test_summary_reports_speakers_and_hours(tmp_path: Path) -> None:
    summary = co.index_summary(co.index_mucs(str(_mucs(tmp_path))))
    assert summary["clips"] == 3
    assert summary["speakers"] == 2
    assert summary["with_transcript"] == 3


def test_summary_lists_undocumented_speakers(tmp_path: Path) -> None:
    frame = co.index_hiacc(str(_hiacc(tmp_path, include_undocumented=True)))
    assert co.index_summary(frame)["undocumented_speakers"] == ["AD63"]


def test_summary_of_an_empty_index_does_not_crash() -> None:
    import pandas as pd

    assert co.index_summary(pd.DataFrame(columns=co.INDEX_COLUMNS))["clips"] == 0
