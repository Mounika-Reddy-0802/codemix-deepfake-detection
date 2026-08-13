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
