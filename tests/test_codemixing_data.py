"""Tests for word-level language tags, the Code-Mixing Index and the control set."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data import codemixing as cm

HI = "मुझे"  # a Devanagari word
HI2 = "कल"


def test_words_are_tagged_by_script_and_numbers_dropped():
    assert cm.token_languages(f"{HI} bank {HI2} 42 ,") == ["hi", "en", "hi"]


def test_cmi_is_zero_for_one_language_and_fifty_for_an_even_mix():
    assert cm.cmi(f"{HI} {HI2} {HI}") == 0.0
    assert cm.cmi("the bank is open") == 0.0
    assert cm.cmi(f"{HI} bank") == pytest.approx(50.0)
    assert cm.cmi("") == 0.0


def test_english_share_and_switches():
    assert cm.english_share(f"{HI} bank account") == pytest.approx(2 / 3)
    assert cm.switch_count(f"{HI} bank {HI2} loan") == 3
    assert cm.switch_count("only english words") == 0


def _index(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "utt_id",
            "speaker",
            "source",
            "wav_path",
            "start_seconds",
            "end_seconds",
            "duration_seconds",
            "transcript",
        ],
    )


def test_control_set_holds_speaker_fixed_across_conditions():
    rows = []
    for i in range(2):
        rows.append([f"s1_en{i}", "s1", "mucs2021", "a.wav", 0, 3, 3.0, "this is all english"])
    for i in range(5):
        rows.append([f"s1_hi{i}", "s1", "mucs2021", "a.wav", 0, 3, 3.0, f"{HI} {HI2} {HI} {HI2}"])
        rows.append([f"s1_mx{i}", "s1", "mucs2021", "a.wav", 0, 3, 3.0, f"{HI} bank {HI2} loan"])
    rows.append(["s2_hi0", "s2", "mucs2021", "b.wav", 0, 3, 3.0, f"{HI} {HI2} {HI}"])  # no English
    rows.append(["s1_long", "s1", "mucs2021", "a.wav", 0, 30, 30.0, "too long english clip"])
    manifest = cm.control_manifest(_index(rows), control_speakers={"s1", "s2"})
    counts = manifest["language_condition"].value_counts().to_dict()
    assert counts == {"all English": 2, "all Hindi": 2, "mixed": 2}
    assert set(manifest["speaker"]) == {"s1"}  # s2 has no all-English clip
    assert set(manifest["label"]) == {"bonafide"}
    assert "s1_long" not in set(manifest["utt_id"])


def test_control_set_is_deterministic():
    rows = [["e0", "s1", "mucs2021", "a.wav", 0, 3, 3.0, "plain english words here"]]
    rows += [[f"h{i}", "s1", "mucs2021", "a.wav", 0, 3, 3.0, f"{HI} {HI2} {HI}"] for i in range(9)]
    a = cm.control_manifest(_index(rows), {"s1"}, seed=7)
    b = cm.control_manifest(_index(rows), {"s1"}, seed=7)
    assert a.equals(b)


def test_control_rows_carry_their_span():
    rows = [["e0", "s1", "mucs2021", "a.wav", 12.5, 15.0, 2.5, "plain english words here"]]
    rows += [["h0", "s1", "mucs2021", "a.wav", 40.0, 43.0, 3.0, f"{HI} {HI2} {HI}"]]
    manifest = cm.control_manifest(_index(rows), {"s1"})
    assert manifest.set_index("utt_id").loc["e0", ["start_seconds", "end_seconds"]].tolist() == [
        12.5,
        15.0,
    ]
