"""Voice check v2: its training manifest and its adoption rule (hand-made measures)."""

from __future__ import annotations

import pandas as pd

from src.data import generator_study as gs
from src.data import spont_recipe
from src.data import voice_check_v2 as v2
from src.reporting import voice_check_v2 as rule

REF = {
    "hiacc": 99.3,
    "read_new_speakers": 99.9,
    "read_control": 99.9,
    "xtts": 99.4,
    "indicf5": 12.5,
    "chatterbox": 4.4,
    "tortoise_new": 81.1,
    "tortoise_cm04": 76.9,
    "rvc": 45.4,
    "genuine_live": 12,
    "genuine_replay": 6,
}


def _good(**changes) -> dict:
    m = dict(REF, indicf5=70.0, chatterbox=80.0, rvc=60.0)
    m.update(changes)
    return m


def _decide(a: dict, b: dict) -> dict:
    return rule.decide({rule.REFERENCE: REF, rule.CANDIDATES[0]: a, rule.CANDIDATES[1]: b})


def test_adopted_only_when_both_seeds_pass_every_condition():
    assert _decide(_good(), _good())["decision"] == f"adopt {rule.ADOPTED}"
    one = _decide(_good(), _good(read_control=98.7))
    assert one["decision"] == "the current detector stays"
    assert one["conditions"][rule.CANDIDATES[1]]["2_read"] is False


def test_each_condition_can_fail_on_its_own():
    assert rule.conditions(_good(hiacc=97.0), REF)["1_spontaneous"] is False
    assert rule.conditions(_good(genuine_replay=4), REF)["2_read"] is False
    assert rule.conditions(_good(chatterbox=49.9), REF)["3_new_tools"] is False
    assert rule.conditions(_good(tortoise_cm04=66.8), REF)["4_unseen"] is False
    assert rule.conditions(_good(xtts=97.9), REF)["5_xtts"] is False


def test_a_missing_quantity_keeps_the_current_detector():
    out = _decide(_good(chatterbox=None), _good())
    assert out["decision"] == "incomplete: the current detector stays"


def test_the_manifest_is_the_spontaneous_recipe_plus_fold_c_rows():
    out = pd.read_csv(f"data/manifests/{v2.TRAIN_OUT}.csv", dtype={"speaker": str})
    base = pd.read_csv(f"data/manifests/{spont_recipe.TRAIN_OUT}.csv", dtype={"speaker": str})
    assert out.equals(v2.build())
    extra = out[~out["filepath"].isin(base["filepath"])]
    assert set(extra["speaker"]) <= gs.train_speakers()
    assert set(extra.loc[extra["label"] == "spoof", "tool"]) == {"indicf5", "chatterbox"}
    assert not out["tool"].str.lower().eq("tortoise").any()
    assert base["filepath"].isin(out["filepath"]).all()


def test_speakers_are_excluded_only_from_the_sets_that_hold_them(monkeypatch):
    from src.reporting import any_audio_accuracy as acc

    seen = {}

    def fake_accuracy(model, key, rows, expected, exclude=frozenset()):
        seen[(key, rows)] = exclude
        return (100.0, 10)

    monkeypatch.setattr(acc, "accuracy", fake_accuracy)
    monkeypatch.setattr(acc, "call_counts", lambda model: {})
    rule.measures(rule.REFERENCE, frozenset({"s1"}))
    for (key, _), exclude in seen.items():
        assert exclude == (frozenset({"s1"}) if key in rule.POOL_SETS else frozenset()), key
    assert {"cm07", "cm09", "new_speakers", "cm04"} <= {k for k, _ in seen}
