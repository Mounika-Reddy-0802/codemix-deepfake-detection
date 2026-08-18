"""Tests for the XTTS-v2 pilot job assembly (Week 3, W3-T5, owner SK).

Two things must hold or the pilot is worthless:

- **the speaker-pool firewall** -- cloning a voice from the adaptation or eval
  pool would put a fake of an evaluation speaker into the training data and
  silently void every generalisation number in the paper;
- **the experimental design** -- the pilot compares scripts and language tags, so
  the reference voice has to be constant across cells. If cell A used different
  speakers from cell B, a difference between them would be uninterpretable.

Pure pandas, no audio, no model.
"""

import pandas as pd
import pytest

from src.data import pilot_jobs as pj

DEVA = "मुझे कल बैंक जाना है पैसे निकालने के लिए और फिर ऑफिस भी जाना है"
LATIN = "mujhe kal bank jaana hai paise nikaalne ke liye aur phir office bhi jaana hai"
MIXED = "मुझे कल bank जाना है paise निकालने के लिए aur phir office भी जाना है"


def _pools() -> pd.DataFrame:
    rows = [{"speaker": f"trn{i}", "pool": "train", "source": "mucs2021"} for i in range(8)]
    rows += [{"speaker": f"adp{i}", "pool": "adaptation", "source": "mucs2021"} for i in range(3)]
    rows += [{"speaker": f"evl{i}", "pool": "eval", "source": "mucs2021"} for i in range(4)]
    return pd.DataFrame(rows)


def _index() -> pd.DataFrame:
    rows = []
    for pool_prefix, count in (("trn", 8), ("adp", 3), ("evl", 4)):
        for i in range(count):
            for j, text in enumerate((DEVA, LATIN, MIXED)):
                rows.append(
                    {
                        "utt_id": f"{pool_prefix}{i}_{j}",
                        "speaker": f"{pool_prefix}{i}",
                        "source": "mucs2021",
                        "wav_path": f"/data/{pool_prefix}{i}.wav",
                        "start_seconds": 0.0,
                        "end_seconds": 12.0,
                        "duration_seconds": 12.0,
                        "transcript": text,
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Script classification
# --------------------------------------------------------------------------- #
def test_devanagari_fraction_extremes() -> None:
    assert pj.devanagari_fraction(DEVA) == pytest.approx(1.0)
    assert pj.devanagari_fraction(LATIN) == pytest.approx(0.0)


def test_mixed_script_lands_between() -> None:
    assert 0.2 < pj.devanagari_fraction(MIXED) < 0.8


def test_script_classes() -> None:
    assert pj.script_class(DEVA) == "mostly_devanagari"
    assert pj.script_class(LATIN) == "mostly_latin"
    assert pj.script_class(MIXED) == "mixed"


def test_punctuation_and_digits_do_not_skew_the_fraction() -> None:
    assert pj.script_class("12345 !!! ,,, " + LATIN) == "mostly_latin"


def test_empty_transcript_is_latin_by_default() -> None:
    assert pj.devanagari_fraction("") == 0.0


# --------------------------------------------------------------------------- #
# The pool firewall -- the one that protects the paper
# --------------------------------------------------------------------------- #
def test_only_train_pool_speakers_are_cloned() -> None:
    jobs = pj.build_pilot_jobs(_index(), _pools())
    assert set(jobs["pool"]) == {"train"}
    assert all(str(s).startswith("trn") for s in jobs["speaker"])


def test_no_eval_or_adaptation_speaker_appears() -> None:
    jobs = pj.build_pilot_jobs(_index(), _pools())
    used = set(jobs["speaker"].astype(str))
    assert not any(s.startswith(("evl", "adp")) for s in used)


def test_firewall_rejects_a_smuggled_eval_speaker() -> None:
    pools = _pools()
    jobs = pj.build_pilot_jobs(_index(), pools)
    jobs.loc[0, "speaker"] = "evl0"
    with pytest.raises(pj.PilotError, match="outside the train pool"):
        pj.assert_train_pool_only(jobs, pools)


def test_firewall_names_the_offending_pool() -> None:
    pools = _pools()
    jobs = pd.DataFrame([{"speaker": "adp1"}])
    with pytest.raises(pj.PilotError, match="adaptation"):
        pj.assert_train_pool_only(jobs, pools)


def test_no_train_speakers_is_an_error() -> None:
    pools = _pools()
    pools = pools[pools["pool"] != "train"]
    with pytest.raises(pj.PilotError):
        pj.build_pilot_jobs(_index(), pools)


# --------------------------------------------------------------------------- #
# Experimental design
# --------------------------------------------------------------------------- #
def test_every_speaker_appears_once_in_every_cell() -> None:
    # Otherwise a cell-to-cell difference is a speaker difference.
    jobs = pj.build_pilot_jobs(_index(), _pools())
    table = pd.crosstab(jobs["speaker"], jobs["cell"])
    assert (table == 1).all().all()


def test_all_four_cells_are_populated() -> None:
    jobs = pj.build_pilot_jobs(_index(), _pools())
    assert set(jobs["cell"]) == {"deva_hi", "latn_hi", "latn_en", "mixed_hi"}
    assert len(jobs) == 20


def test_cells_carry_the_intended_language_tag() -> None:
    jobs = pj.build_pilot_jobs(_index(), _pools()).set_index("cell")
    assert set(jobs.loc["latn_en", "language"]) == {"en"}
    assert set(jobs.loc["deva_hi", "language"]) == {"hi"}


def test_cells_carry_the_intended_script() -> None:
    jobs = pj.build_pilot_jobs(_index(), _pools())
    for cell, expected in (
        ("deva_hi", "mostly_devanagari"),
        ("latn_hi", "mostly_latin"),
        ("mixed_hi", "mixed"),
    ):
        got = jobs.loc[jobs["cell"] == cell, "transcript"].map(pj.script_class)
        assert set(got) == {expected}, cell


def test_output_paths_are_unique() -> None:
    jobs = pj.build_pilot_jobs(_index(), _pools())
    assert jobs["output_path"].nunique() == len(jobs)


def test_seeds_are_unique_per_job() -> None:
    jobs = pj.build_pilot_jobs(_index(), _pools())
    assert jobs["seed"].nunique() == len(jobs)


def test_short_transcripts_are_excluded() -> None:
    # A three-word utterance cannot exercise a code-switch boundary.
    index = _index()
    index["transcript"] = "hi bye"
    picked = pj.choose_transcripts(index, "mostly_latin", 5)
    assert picked.empty


def test_short_references_are_excluded() -> None:
    index = _index()
    index["duration_seconds"] = 2.0
    assert pj.choose_references(index, _pools()).empty


# --------------------------------------------------------------------------- #
# Reproducibility across machines
# --------------------------------------------------------------------------- #
# The CPU laptop and the GPU laptop built the same pilot with speakers 4 and 5
# swapped in every cell: MUCS durations are whole seconds, 177570 and 850754 were
# both exactly 20.0 s, and the tie fell through to clip_index.csv row order, which
# follows the filesystem walk. `deva_hi_04` was therefore a different person on
# each machine and the rating sheet would have mis-attributed every score.
def _shuffled(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """The same rows a different filesystem walk would have produced."""
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _tied_index() -> pd.DataFrame:
    """Train-pool speakers whose references are all exactly the same length."""
    index = _index()
    index["duration_seconds"] = 12.0  # every candidate ties
    index["end_seconds"] = 12.0
    return index


def test_reference_choice_survives_a_reordered_index() -> None:
    pools = _pools()
    index = _tied_index()
    baseline = pj.choose_references(index, pools)["speaker"].tolist()
    for seed in (0, 1, 2, 17, 99):
        assert pj.choose_references(_shuffled(index, seed), pools)["speaker"].tolist() == baseline


def test_transcript_choice_survives_a_reordered_index() -> None:
    index = _tied_index()
    baseline = pj.choose_transcripts(index, "mostly_latin", 5)["utt_id"].tolist()
    for seed in (0, 1, 2, 17, 99):
        got = pj.choose_transcripts(_shuffled(index, seed), "mostly_latin", 5)["utt_id"].tolist()
        assert got == baseline


def test_the_whole_job_table_is_reproducible_across_index_orderings() -> None:
    # The end-to-end guarantee: two machines that index the same corpus into the
    # same rows in a different order must generate a byte-identical pilot.
    pools = _pools()
    index = _tied_index()
    baseline = pj.build_pilot_jobs(index, pools)
    for seed in (0, 1, 2, 17, 99):
        got = pj.build_pilot_jobs(_shuffled(index, seed), pools)
        pd.testing.assert_frame_equal(got, baseline)


# --------------------------------------------------------------------------- #
# Romanised packs (team decision, 17 Aug 2026: one script, and it is Latin)
# --------------------------------------------------------------------------- #
ROMANISE = pj.PilotConfig(romanise=True)


def test_romanised_jobs_carry_no_devanagari() -> None:
    from src.data.transliteration import has_devanagari

    jobs = pj.build_pilot_jobs(_index(), _pools(), config=ROMANISE)
    assert not any(has_devanagari(t) for t in jobs["transcript"])


def test_romanised_jobs_keep_the_source_text_for_provenance() -> None:
    # The datasheet has to be able to show what the corpus actually said.
    jobs = pj.build_pilot_jobs(_index(), _pools(), config=ROMANISE)
    assert (jobs["transcript_source"] != jobs["transcript"]).any()
    assert set(jobs["transcript_source"]) <= set(_index()["transcript"])


def test_the_default_pack_is_untouched_devanagari() -> None:
    # Romanisation is opt-in; the default path must not change silently.
    jobs = pj.build_pilot_jobs(_index(), _pools())
    assert (jobs["transcript_source"] == jobs["transcript"]).all()


def test_cells_are_still_chosen_by_the_SOURCE_script() -> None:
    # After transliteration every transcript is Latin. Classifying the *output*
    # would put all 20 jobs in one cell and the pilot would compare nothing.
    jobs = pj.build_pilot_jobs(_index(), _pools(), config=ROMANISE)
    assert set(jobs["cell"]) == {"deva_hi", "latn_hi", "latn_en", "mixed_hi"}
    for cell, expected in (
        ("deva_hi", "mostly_devanagari"),
        ("latn_hi", "mostly_latin"),
        ("mixed_hi", "mixed"),
    ):
        source = jobs.loc[jobs["cell"] == cell, "transcript_source"].map(pj.script_class)
        assert set(source) == {expected}, cell


def test_the_character_limit_is_applied_to_the_romanised_text() -> None:
    # Romanisation inflates length ~13% (up to 58%): on the real corpus 14 of the
    # 20 pilot transcripts cross the 150-char Hindi limit once transliterated.
    # Measuring the Devanagari would hand XTTS text it silently truncates -- the
    # P-010 failure, re-introduced through the back door.
    jobs = pj.build_pilot_jobs(_index(), _pools(), config=ROMANISE)
    over = [
        (row.language, len(row.transcript))
        for row in jobs.itertuples(index=False)
        if len(row.transcript) > pj.char_limit(row.language)
    ]
    assert over == []


def test_a_transcript_that_only_fits_before_romanising_is_rejected() -> None:
    # 60 Devanagari chars -> ~100 romanised. Under a 70-char cap it passes on the
    # source and fails on what XTTS receives; only the latter is correct.
    index = _index_of_lengths([60])
    assert len(pj.choose_transcripts(index, "mostly_devanagari", 1, language="hi")) == 1
    romanised = pj.choose_transcripts(
        index, "mostly_devanagari", 1, language="hi", max_chars=70, romanise=True
    )
    assert len(romanised) == 0


def test_spoken_text_matches_what_the_job_table_stores() -> None:
    # One helper decides what the model is handed, so filters and assembly cannot
    # disagree about it.
    jobs = pj.build_pilot_jobs(_index(), _pools(), config=ROMANISE)
    for row in jobs.itertuples(index=False):
        assert row.transcript == pj.spoken_text(row.transcript_source, romanise=True)


def test_romanised_pack_is_reproducible_across_index_orderings() -> None:
    pools = _pools()
    index = _tied_index()
    baseline = pj.build_pilot_jobs(index, pools, config=ROMANISE)
    for seed in (0, 3, 42):
        pd.testing.assert_frame_equal(
            pj.build_pilot_jobs(_shuffled(index, seed), pools, config=ROMANISE), baseline
        )


def test_summary_reports_the_design() -> None:
    summary = pj.pilot_summary(pj.build_pilot_jobs(_index(), _pools()))
    assert summary["jobs"] == 20
    assert summary["speakers"] == 5
    assert summary["pools_used"] == ["train"]
    assert sorted(summary["languages"]) == ["en", "hi"]


# --------------------------------------------------------------------------- #
# Loading the pack for the generator
# --------------------------------------------------------------------------- #
def test_load_pilot_jobs_resolves_pack_relative_references(tmp_path) -> None:
    # References are stored as `refs/<speaker>.wav` so the pack can move between
    # machines (local -> Drive -> Colab) without anyone editing the CSV.
    jobs_table = pj.build_pilot_jobs(_index(), _pools())
    csv = tmp_path / "generation_jobs.csv"
    jobs_table.to_csv(csv, index=False)

    loaded = pj.load_pilot_jobs(str(csv), str(tmp_path), out_dir="outputs")
    assert len(loaded) == 20
    assert all(str(tmp_path) in j.reference_wav for j in loaded)
    assert all(j.reference_wav.endswith(".wav") for j in loaded)


def test_loaded_jobs_keep_the_train_pool_tag(tmp_path) -> None:
    jobs_table = pj.build_pilot_jobs(_index(), _pools())
    csv = tmp_path / "generation_jobs.csv"
    jobs_table.to_csv(csv, index=False)
    loaded = pj.load_pilot_jobs(str(csv), str(tmp_path))
    assert {j.pool for j in loaded} == {"train"}
    assert {j.tool for j in loaded} == {"xtts_v2"}


def test_loaded_jobs_have_distinct_outputs_and_seeds(tmp_path) -> None:
    jobs_table = pj.build_pilot_jobs(_index(), _pools())
    csv = tmp_path / "generation_jobs.csv"
    jobs_table.to_csv(csv, index=False)
    loaded = pj.load_pilot_jobs(str(csv), str(tmp_path))
    assert len({j.output_path for j in loaded}) == 20
    assert len({j.seed for j in loaded}) == 20


# --------------------------------------------------------------------------- #
# XTTS character limit
# --------------------------------------------------------------------------- #
# The first real pilot run put 19 of 20 jobs over XTTS-v2's per-language limit
# (median 278 chars against a 150-char Hindi cap). XTTS truncates silently and
# synthesises only the opening fragment, so raters would have been scoring
# truncation rather than code-switch quality -- the one thing the pilot measures.
def _index_of_lengths(lengths: list[int], script: str = "deva") -> pd.DataFrame:
    unit = "क" if script == "deva" else "a"
    return pd.DataFrame(
        {
            "utt_id": [f"u{i}" for i in range(len(lengths))],
            "speaker": ["trn0"] * len(lengths),
            "source": "mucs2021",
            "wav_path": "/data/trn0.wav",
            "start_seconds": 0.0,
            "end_seconds": 12.0,
            "duration_seconds": 12.0,
            "transcript": [unit * n for n in lengths],
        }
    )


def test_hindi_limit_is_tighter_than_english() -> None:
    assert pj.char_limit("hi") < pj.char_limit("en")


def test_unknown_language_falls_back_to_the_default_limit() -> None:
    assert pj.char_limit("ta") == pj.DEFAULT_CHAR_LIMIT


def test_transcripts_over_the_limit_are_excluded() -> None:
    chosen = pj.choose_transcripts(
        _index_of_lengths([100, 140, 200, 400]), "mostly_devanagari", 5, language="hi"
    )
    assert len(chosen) == 2
    assert chosen["transcript"].str.len().max() <= pj.char_limit("hi")


def test_longest_sayable_transcript_still_wins() -> None:
    # The bound must not invert the design: within the limit, longest-first holds.
    chosen = pj.choose_transcripts(
        _index_of_lengths([50, 140, 300]), "mostly_devanagari", 1, language="hi"
    )
    assert len(chosen.iloc[0]["transcript"]) == 140


def test_the_english_cell_may_use_longer_text_than_the_hindi_cells() -> None:
    index = _index_of_lengths([200], script="latn")
    assert len(pj.choose_transcripts(index, "mostly_latin", 1, language="hi")) == 0
    assert len(pj.choose_transcripts(index, "mostly_latin", 1, language="en")) == 1


def test_every_built_job_is_within_its_own_language_limit() -> None:
    jobs = pj.build_pilot_jobs(_index(), _pools())
    over = [
        (row.language, len(row.transcript))
        for row in jobs.itertuples(index=False)
        if len(row.transcript) > pj.char_limit(row.language)
    ]
    assert over == []


def test_hindi_transcripts_with_digits_are_excluded() -> None:
    # XTTS calls num2words(n, lang="hi"), which raises NotImplementedError. This
    # crashed the second pilot run at clip 7 on a real MUCS lecture transcript.
    assert pj.is_synthesisable("क" * 100, "hi") is True
    assert pj.is_synthesisable("क" * 50 + " 42 " + "क" * 40, "hi") is False


def test_english_transcripts_with_digits_are_allowed() -> None:
    # num2words does support English, so digits are only a problem under "hi".
    assert pj.is_synthesisable("a" * 50 + " 42 " + "a" * 40, "en") is True


def test_over_length_transcripts_are_not_synthesisable() -> None:
    assert pj.is_synthesisable("क" * (pj.char_limit("hi") + 1), "hi") is False


def test_choose_transcripts_skips_numeric_hindi() -> None:
    index = _index_of_lengths([100, 100])
    index.loc[0, "transcript"] = "क" * 50 + " 2024 " + "क" * 40
    chosen = pj.choose_transcripts(index, "mostly_devanagari", 5, language="hi")
    assert len(chosen) == 1
    assert not any(c.isdigit() for c in chosen.iloc[0]["transcript"])


def test_the_pilot_uses_one_length_cap_across_all_cells() -> None:
    # The tightest of the tags in use, so cells stay comparable.
    assert pj.pilot_char_limit() == pj.char_limit("hi")


def test_every_cell_gets_comparable_transcript_lengths() -> None:
    # Without a shared cap the en cell would draw 250-char text while the hi cells
    # drew 150, and a latn_en vs latn_hi rating difference could be length rather
    # than the language tag -- the one comparison the cell exists to make.
    jobs = pj.build_pilot_jobs(_index(), _pools())
    longest = jobs.groupby("cell")["transcript"].apply(lambda s: s.str.len().max())
    assert longest.max() <= pj.pilot_char_limit()
