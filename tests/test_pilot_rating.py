"""Tests for the XTTS pilot A/B rating sheet (W3-T5).

The sheet decides whether the Week-4 run synthesises from romanised or Devanagari
transcripts, so the two things that make its answer trustworthy are pinned here:

- the pair really is **matched** (same speaker, same sentence, same tag), or a
  rating difference means nothing;
- the rater cannot tell **which script** they are hearing, or the decision
  measures expectation rather than audio.
"""

import pandas as pd
import pytest

from src.data import pilot_rating as pr

DEVA = "मुझे कल बैंक जाना है पैसे निकालने के लिए"
ROMAN = "mujhe kal baink jaanaa hai paise nikaalne ke lie"


def _roman_jobs(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "job_id": range(1, n + 1),
            "speaker": [f"spk{i}" for i in range(n)],
            "language": ["hi"] * n,
            "transcript": [f"{ROMAN} {i}" for i in range(n)],
            "transcript_source": [f"{DEVA} {i}" for i in range(n)],
            "output_path": [f"outputs/roman_{i:02d}.wav" for i in range(n)],
        }
    )


def _deva_jobs(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "job_id": range(1, n + 1),
            "speaker": [f"spk{i}" for i in range(n)],
            "language": ["hi"] * n,
            "transcript": [f"{DEVA} {i}" for i in range(n)],
            "output_path": [f"outputs/deva_{i:02d}.wav" for i in range(n)],
        }
    )


def _sheet(**kwargs) -> pd.DataFrame:
    return pr.build_ab_sheet(_roman_jobs(), _deva_jobs(), "roman/out", "deva/out", **kwargs)


# --------------------------------------------------------------------------- #
# The pairing must be matched, or the comparison is void
# --------------------------------------------------------------------------- #
def test_mismatched_sentences_are_rejected() -> None:
    deva = _deva_jobs()
    deva.loc[2, "transcript"] = "कुछ और ही वाक्य"
    with pytest.raises(ValueError, match="source sentence"):
        pr.build_ab_sheet(_roman_jobs(), deva, "roman/out", "deva/out")


def test_different_job_counts_are_rejected() -> None:
    with pytest.raises(ValueError, match="different job counts"):
        pr.build_ab_sheet(_roman_jobs(4), _deva_jobs(3), "roman/out", "deva/out")


def test_each_pair_holds_speaker_and_language_constant() -> None:
    sheet = _sheet(config=pr.RatingConfig(blind=False))
    for _, pair in sheet.groupby("pair_id"):
        assert pair["speaker"].nunique() == 1
        assert pair["language"].nunique() == 1


def test_every_clip_appears_once_per_rater() -> None:
    sheet = _sheet()
    counts = sheet.groupby(["blind_id", "rater"]).size()
    assert set(counts) == {1}
    assert sheet["blind_id"].nunique() == 8  # 4 pairs x 2 scripts


# --------------------------------------------------------------------------- #
# Blinding
# --------------------------------------------------------------------------- #
def test_the_sheet_is_blind_by_default() -> None:
    # If the rater can see which clip is Devanagari, the sheet measures their
    # expectation rather than the audio.
    assert "script" not in _sheet().columns


def test_no_column_of_a_blind_sheet_identifies_the_script() -> None:
    # clip_id encodes the script in its a/b suffix and pair_id groups the halves;
    # either one hands the answer to the rater as plainly as a label would.
    assert not set(pr.BLIND_COLUMNS) & set(_sheet().columns)


def test_the_blind_id_is_assigned_after_shuffling() -> None:
    # Numbered before the shuffle, blind_id would run in clip_id order and every
    # odd id would be romanised -- the blinding would be decorative.
    unblinded = _sheet(config=pr.RatingConfig(blind=False))
    ordered = unblinded.drop_duplicates("blind_id").sort_values("blind_id")
    scripts = list(ordered["script"])
    assert scripts != ["roman", "devanagari"] * (len(scripts) // 2)
    assert scripts != ["roman"] * 4 + ["devanagari"] * 4


def test_the_script_can_be_shown_on_request() -> None:
    sheet = _sheet(config=pr.RatingConfig(blind=False))
    assert set(sheet["script"]) == {"roman", "devanagari"}


def test_the_answer_key_covers_every_clip() -> None:
    unblinded = _sheet(config=pr.RatingConfig(blind=False))
    key = pr.answer_key(unblinded)
    assert set(key["blind_id"]) == set(_sheet()["blind_id"])
    assert list(key["script"]).count("roman") == 4
    assert list(key["script"]).count("devanagari") == 4


def test_the_key_cannot_be_built_from_a_blind_sheet() -> None:
    with pytest.raises(ValueError, match="unblinded"):
        pr.answer_key(_sheet())


def test_blind_and_unblinded_sheets_agree_on_the_shuffle() -> None:
    # The key is built from the unblinded pass, so the two must line up or every
    # score is attributed to the wrong script.
    blind = _sheet().drop_duplicates("blind_id").set_index("blind_id")
    full = _sheet(config=pr.RatingConfig(blind=False)).drop_duplicates("blind_id")
    full = full.set_index("blind_id")
    assert (blind["audio_path"] == full["audio_path"]).all()


# --------------------------------------------------------------------------- #
# The readable transcript -- the reason P-014 exists
# --------------------------------------------------------------------------- #
def test_both_halves_of_a_pair_show_the_romanised_text() -> None:
    from src.data.transliteration import has_devanagari

    sheet = _sheet()
    assert not any(has_devanagari(t) for t in sheet["read_along"])
    unblinded = _sheet(config=pr.RatingConfig(blind=False))
    for _, pair in unblinded.groupby("pair_id"):
        assert pair["read_along"].nunique() == 1


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def test_blank_rows_are_reported_not_guessed() -> None:
    summary = pr.summarise_ab(_sheet())
    assert summary["rated"] == 0
    assert summary["unrated"] == summary["rows"]
    assert summary["complete"] is False


def test_scores_are_summarised_per_script() -> None:
    sheet = _sheet(config=pr.RatingConfig(blind=False))
    sheet["sounds_human_1_5"] = [4 if s == "roman" else 2 for s in sheet["script"]]
    sheet["said_the_right_words_1_5"] = 3
    sheet["code_switch_natural_1_5"] = 3
    summary = pr.summarise_ab(sheet)
    assert summary["complete"] is True
    assert summary["roman"]["sounds_human_1_5"] == 4.0
    assert summary["devanagari"]["sounds_human_1_5"] == 2.0


def test_a_blind_sheet_is_scored_by_joining_the_key() -> None:
    sheet = _sheet()
    key = pr.answer_key(_sheet(config=pr.RatingConfig(blind=False)))
    sheet["sounds_human_1_5"] = 5
    sheet["said_the_right_words_1_5"] = 5
    sheet["code_switch_natural_1_5"] = 5
    summary = pr.summarise_ab(sheet, key)
    assert summary["roman"]["sounds_human_1_5"] == 5.0
    assert summary["devanagari"]["sounds_human_1_5"] == 5.0
