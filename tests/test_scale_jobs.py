"""Tests for the W4-T1 generation job table (~4,000 clips).

This table decides what the training corpus contains, so the invariants below are
the ones that make it usable at all:

- **the pool firewall** -- a clone of an evaluation voice inside training data
  voids every generalisation number in the paper;
- **no repeated (speaker, transcript) pair** -- a duplicate clip teaches the
  detector the sentence rather than the artefact;
- **reproducibility** -- two machines must build the same 4,000 jobs, for the
  same reason the pilot must (P-016).

Pure pandas, no audio, no model.
"""

import pandas as pd
import pytest

from src.data import scale_jobs as sj
from src.data.pilot_jobs import PilotError

DEVA = "मुझे कल बैंक जाना है पैसे निकालने के लिए और फिर ऑफिस भी जाना है"


def _pools(n_train: int = 5) -> pd.DataFrame:
    rows = [{"speaker": f"trn{i}", "pool": "train", "source": "mucs2021"} for i in range(n_train)]
    rows += [{"speaker": f"evl{i}", "pool": "eval", "source": "mucs2021"} for i in range(3)]
    rows += [{"speaker": f"adp{i}", "pool": "adaptation", "source": "mucs2021"} for i in range(2)]
    return pd.DataFrame(rows)


def _index(n_train: int = 5, texts_per_speaker: int = 8) -> pd.DataFrame:
    rows = []
    for prefix, count in (("trn", n_train), ("evl", 3), ("adp", 2)):
        for i in range(count):
            for j in range(texts_per_speaker):
                rows.append(
                    {
                        "utt_id": f"{prefix}{i}_{j:03d}",
                        "speaker": f"{prefix}{i}",
                        "source": "mucs2021",
                        "wav_path": f"/data/{prefix}{i}.wav",
                        "start_seconds": 0.0,
                        "end_seconds": 12.0,
                        "duration_seconds": 12.0,
                        # Letters, not digits: a digit makes a transcript
                        # unsynthesisable under the "hi" tag (P-011).
                        "transcript": f"{DEVA} {prefix}{chr(97 + i)}{chr(97 + j)}",
                    }
                )
    return pd.DataFrame(rows)


def _cfg(**kw):
    return sj.ScaleConfig(**{"n_target": 40, **kw})


# --------------------------------------------------------------------------- #
# The firewall
# --------------------------------------------------------------------------- #
def test_only_train_pool_speakers_are_cloned():
    jobs = sj.build_scale_jobs(_index(), _pools(), config=_cfg())
    assert set(jobs["pool"]) == {"train"}
    assert all(str(s).startswith("trn") for s in jobs["speaker"])


def test_the_firewall_rejects_a_smuggled_speaker():
    pools = _pools()
    jobs = sj.build_scale_jobs(_index(), pools, config=_cfg())
    jobs.loc[0, "speaker"] = "evl0"
    with pytest.raises(PilotError, match="outside the train pool"):
        sj.assert_scale_invariants(jobs, pools)


# --------------------------------------------------------------------------- #
# No duplicate clips
# --------------------------------------------------------------------------- #
def test_no_speaker_transcript_pair_is_generated_twice():
    jobs = sj.build_scale_jobs(_index(), _pools(), config=_cfg())
    assert not jobs[["speaker", "transcript"]].duplicated().any()


def test_a_duplicated_pair_is_rejected():
    pools = _pools()
    jobs = sj.build_scale_jobs(_index(), pools, config=_cfg())
    jobs.loc[1, ["speaker", "transcript"]] = jobs.loc[0, ["speaker", "transcript"]].values
    with pytest.raises(PilotError, match="duplicate"):
        sj.assert_scale_invariants(jobs, pools)


def test_asking_for_more_than_capacity_is_an_error():
    # 5 speakers and 40 pooled texts. gcd(5, 40) = 5, so the stride would wrap
    # after 40 jobs; the builder trims to 39 texts to make them coprime, giving a
    # real capacity of 5 x 39 = 195 reachable pairs.
    with pytest.raises(PilotError, match="only 195 unique"):
        sj.build_scale_jobs(_index(), _pools(), config=_cfg(n_target=196))


def test_the_stride_never_wraps_within_capacity():
    # The bug this guards: with gcd(S, T) > 1 the stride repeats a pair long
    # before S x T, so a "capacity" of S x T would be a lie.
    jobs = sj.build_scale_jobs(_index(), _pools(), config=_cfg(n_target=195))
    assert not jobs[["speaker", "transcript"]].duplicated().any()
    assert len(jobs) == 195


# --------------------------------------------------------------------------- #
# Balance -- no speaker may dominate the corpus
# --------------------------------------------------------------------------- #
def test_every_speaker_gets_the_same_number_of_clips():
    jobs = sj.build_scale_jobs(_index(), _pools(), config=_cfg(n_target=40))
    counts = jobs["speaker"].value_counts()
    assert counts.min() == counts.max() == 8


def test_a_speakers_texts_are_spread_not_clustered():
    # The stride must not hand speaker 0 only the first N transcripts, or the
    # corpus would repeat a narrow slice of the material.
    jobs = sj.build_scale_jobs(_index(), _pools(), config=_cfg(n_target=40))
    first = jobs[jobs["speaker"] == "trn0"]
    assert first["transcript"].nunique() == len(first)


# --------------------------------------------------------------------------- #
# Reproducibility across machines (P-016)
# --------------------------------------------------------------------------- #
def test_the_table_is_identical_under_a_reordered_index():
    pools, index = _pools(), _index()
    baseline = sj.build_scale_jobs(index, pools, config=_cfg())
    for seed in (0, 1, 7, 42):
        shuffled = index.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        pd.testing.assert_frame_equal(sj.build_scale_jobs(shuffled, pools, config=_cfg()), baseline)


# --------------------------------------------------------------------------- #
# What reaches XTTS
# --------------------------------------------------------------------------- #
def test_transcripts_are_romanised_by_default():
    from src.data.transliteration import has_devanagari

    jobs = sj.build_scale_jobs(_index(), _pools(), config=_cfg())
    assert not any(has_devanagari(t) for t in jobs["transcript"])
    assert all(has_devanagari(t) for t in jobs["transcript_source"])


def test_romanisation_can_be_turned_off():
    jobs = sj.build_scale_jobs(_index(), _pools(), config=_cfg(romanise=False))
    assert (jobs["transcript"] == jobs["transcript_source"]).all()


def test_nothing_xtts_would_silently_truncate_survives():
    jobs = sj.build_scale_jobs(_index(), _pools(), config=_cfg())
    from src.data.pilot_jobs import char_limit

    assert all(len(r.transcript) <= char_limit(r.language) for r in jobs.itertuples(index=False))


def test_transcripts_are_deduplicated_on_the_spoken_text():
    # The same sentence under two utterance ids is one transcript; generating
    # both would be a duplicate wearing a different id.
    index = _index()
    index.loc[1, "transcript"] = index.loc[0, "transcript"]
    texts = sj.usable_transcripts(index, _pools())
    assert texts["spoken_text"].is_unique


def test_summary_reports_the_shape_of_the_run():
    jobs = sj.build_scale_jobs(_index(), _pools(), config=_cfg())
    s = sj.scale_summary(jobs)
    assert s["jobs"] == 40
    assert s["speakers"] == 5
    assert s["per_speaker_min"] == s["per_speaker_max"] == 8
    assert s["pools_used"] == ["train"]


def test_no_train_pool_speaker_is_an_error():
    pools = _pools()
    pools = pools[pools["pool"] != "train"]
    with pytest.raises(PilotError):
        sj.build_scale_jobs(_index(), pools, config=_cfg())
