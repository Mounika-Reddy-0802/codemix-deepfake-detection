"""The voice-check adoption rule, on hand-made measures (no score file is read)."""

from __future__ import annotations

from src.reporting import voice_check_adoption as va

REF = {
    "hiacc": 99.3,
    "read_new_speakers": 99.9,
    "read_control": 99.9,
    "xtts": 99.4,
    "tortoise_new": 81.1,
    "tortoise_cm04": 76.9,
    "indicf5": 12.5,
    "chatterbox": 4.4,
    "rvc": 45.4,
    "genuine_live": 12,
    "genuine_replay": 6,
}


def _better(**changes) -> dict:
    m = dict(REF, indicf5=60.0, chatterbox=70.0, rvc=80.0)  # unseen mean 73.6
    m.update(changes)
    return m


def test_the_reference_values_give_the_thresholds_written_in_the_rule():
    names = [n for n, _, _ in va.UNSEEN]
    assert round(sum(REF[n] for n in names) / len(names) + 10, 1) == va.UNSEEN_MEAN_MIN
    assert round(REF["hiacc"] - 2, 1) == va.SPONTANEOUS_MIN


def test_adopted_only_when_both_seeds_pass_every_condition():
    both = va.decide({va.REFERENCE: REF, va.CANDIDATES[0]: _better(), va.CANDIDATES[1]: _better()})
    assert both["decision"] == f"adopt {va.ADOPTED}"
    one = va.decide(
        {va.REFERENCE: REF, va.CANDIDATES[0]: _better(), va.CANDIDATES[1]: _better(hiacc=97.0)}
    )
    assert one["decision"] == "the current detector stays"
    assert one["conditions"][va.CANDIDATES[1]]["1_spontaneous"] is False


def test_each_condition_can_fail_on_its_own():
    assert va.conditions(_better(genuine_live=10), REF)["2_read"] is False
    assert va.conditions(_better(read_control=98.9), REF)["2_read"] is False
    assert va.conditions(_better(xtts=97.9), REF)["4_xtts"] is False
    # a large gain elsewhere does not excuse a row more than 10 points below the reference
    assert va.conditions(_better(tortoise_cm04=66.8), REF)["3_unseen"] is False
    # nor does keeping every row: the mean must rise 10 points
    assert va.conditions(dict(REF), REF)["3_unseen"] is False


def test_a_missing_measure_keeps_the_current_detector():
    out = va.decide(
        {
            va.REFERENCE: REF,
            va.CANDIDATES[0]: _better(),
            va.CANDIDATES[1]: _better(genuine_live=None),
        }
    )
    assert out["conditions"][va.CANDIDATES[1]]["2_read"] is None
    assert out["decision"].startswith("incomplete")
