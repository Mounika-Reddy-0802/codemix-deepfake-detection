"""Tests for the channel-sim listening test builder (Week 3, W3-T2, owner L).

The sheet is the artefact that proves a human actually listened, so the pieces
worth pinning are: the sample is spread across corpora, every rater gets every
clip, and a half-filled sheet reports itself as incomplete rather than averaging
whatever happens to be there.

Pure pandas, no audio rendering, so this runs in CI.
"""

import pandas as pd

from src.data import listening_test as lt


def _clips(n_mucs: int = 30, n_hiacc: int = 10) -> pd.DataFrame:
    rows = [
        {"filepath": f"mucs/spk{i}/clip{i}.wav", "speaker": f"spk{i}", "source": "mucs2021"}
        for i in range(n_mucs)
    ]
    rows += [
        {"filepath": f"hiacc/adult{i}/clip{i}.wav", "speaker": f"adult{i}", "source": "hiacc"}
        for i in range(n_hiacc)
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def test_sample_size_matches_the_config() -> None:
    assert len(lt.sample_clips(_clips())) == lt.DEFAULT_N_CLIPS


def test_sample_covers_every_source() -> None:
    # 20 MUCS clips and no HiACC would leave the noisier corpus unjudged.
    sampled = lt.sample_clips(_clips())
    assert set(sampled["source"]) == {"mucs2021", "hiacc"}


def test_sampling_is_deterministic() -> None:
    first = lt.sample_clips(_clips())["filepath"].tolist()
    second = lt.sample_clips(_clips())["filepath"].tolist()
    assert first == second


def test_small_corpus_is_not_padded() -> None:
    sampled = lt.sample_clips(_clips(n_mucs=3, n_hiacc=2))
    assert len(sampled) == 5


def test_empty_index_gives_an_empty_sample() -> None:
    assert lt.sample_clips(pd.DataFrame(columns=["filepath", "source"])).empty


def test_index_without_a_source_column_still_samples() -> None:
    clips = pd.DataFrame({"filepath": [f"a{i}.wav" for i in range(25)]})
    assert len(lt.sample_clips(clips)) == lt.DEFAULT_N_CLIPS


# --------------------------------------------------------------------------- #
# Filenames
# --------------------------------------------------------------------------- #
def test_pair_filenames_are_distinct_and_numbered() -> None:
    clean = lt.clean_filename("data/x/utt_0001.wav", 3)
    channel = lt.channel_filename("data/x/utt_0001.wav", 3)
    assert clean == "pair03_utt_0001.clean.wav"
    assert channel == "pair03_utt_0001.channel.wav"
    assert clean != channel


def test_windows_paths_are_handled() -> None:
    assert lt.clean_filename(r"data\x\utt_0001.wav", 1) == "pair01_utt_0001.clean.wav"


# --------------------------------------------------------------------------- #
# Rating sheet
# --------------------------------------------------------------------------- #
def test_every_rater_gets_every_clip() -> None:
    sampled = lt.sample_clips(_clips())
    sheet = lt.build_rating_sheet(sampled, "out")
    assert len(sheet) == len(sampled) * 3
    assert set(sheet["rater"]) == {"L", "M", "SK"}


def test_sheet_has_the_expected_columns() -> None:
    sheet = lt.build_rating_sheet(lt.sample_clips(_clips()), "out")
    assert list(sheet.columns) == lt.RATING_COLUMNS


def test_score_columns_start_blank() -> None:
    sheet = lt.build_rating_sheet(lt.sample_clips(_clips()), "out")
    assert (sheet["sounds_like_telephony_1_5"] == "").all()


# --------------------------------------------------------------------------- #
# Summarising a filled-in sheet
# --------------------------------------------------------------------------- #
def test_blank_sheet_is_reported_as_incomplete() -> None:
    sheet = lt.build_rating_sheet(lt.sample_clips(_clips()), "out")
    summary = lt.summarise_ratings(sheet)
    assert summary["rated"] == 0
    assert summary["complete"] is False
    assert summary["mean_telephony"] is None


def test_partially_filled_sheet_counts_the_gaps() -> None:
    sheet = lt.build_rating_sheet(lt.sample_clips(_clips(n_mucs=2, n_hiacc=2)), "out")
    sheet.loc[0, "sounds_like_telephony_1_5"] = 4
    summary = lt.summarise_ratings(sheet)
    assert summary["rated"] == 1
    assert summary["unrated"] == len(sheet) - 1
    assert summary["complete"] is False


def test_fully_filled_sheet_averages_the_scores() -> None:
    sheet = lt.build_rating_sheet(lt.sample_clips(_clips(n_mucs=1, n_hiacc=1)), "out")
    sheet["sounds_like_telephony_1_5"] = [5, 3, 4, 4, 4, 4]
    sheet["intelligible_1_5"] = 4
    summary = lt.summarise_ratings(sheet)
    assert summary["complete"] is True
    assert summary["mean_telephony"] == 4.0
    assert summary["mean_intelligible"] == 4.0
