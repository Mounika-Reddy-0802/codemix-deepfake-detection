"""v2 averaged with the spontaneous recipe (P-046): the averaged files and the rule."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from src.reporting import any_audio_accuracy as acc
from src.reporting import rvc_average as avg

V2 = {
    "hiacc": 99.8,
    "read_new_speakers": 100.0,
    "read_control": 100.0,
    "genuine_live": 12,
    "genuine_replay": 6,
    "xtts": 100.0,
    "rvc": 73.5,
    "indicf5": 89.3,
    "chatterbox": 93.8,
    "tortoise_new": 94.1,
    "tortoise_cm04": 91.9,
}


def _measures(**changes) -> dict:
    out = {}
    for seed in avg.SEEDS:
        out[f"ssl_vcv2_s{seed}"] = dict(V2)
        out[avg.model(seed)] = {**V2, "rvc": 80.0, **changes}
    return out


def test_all_five_met_in_both_pairs_is_recommended():
    assert avg.decide(_measures())["answer"] == "recommended"


@pytest.mark.parametrize(
    ("change", "failed"),
    [
        ({"hiacc": 97.0}, "1_genuine"),
        ({"genuine_live": 10}, "1_genuine"),
        ({"xtts": 97.9}, "2_xtts"),
        ({"rvc": 78.4}, "3_rvc_gain"),
        ({"chatterbox": 88.7}, "4_new_tools_kept"),
        ({"tortoise_cm04": 86.8}, "5_tortoise_kept"),
    ],
)
def test_each_condition_can_fail_on_its_own(change, failed):
    out = avg.decide(_measures(**change))
    assert out["answer"] == "not recommended"
    assert [k for k, v in out["conditions"]["1234"].items() if not v] == [failed]


def test_a_missing_quantity_is_incomplete():
    m = _measures()
    m[avg.model(2025)]["rvc"] = None
    assert avg.decide(m)["answer"] == "incomplete"


def test_clips_are_averaged_as_probabilities_and_stored_as_log_odds(tmp_path):
    a = tmp_path / avg.V2_CLIPS.format(seed=7)
    b = tmp_path / avg.SPONT_CLIPS.format(seed=7)
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    clips = pd.DataFrame({"filepath": ["x.wav", "y.wav"], "label": ["bonafide", "spoof"]})
    clips.assign(score=[0.0, 2.0]).to_csv(a / "vcv2_s7__rvc_test_scores.csv", index=False)
    clips.assign(score=[2.0, -2.0]).to_csv(b / "spont_s7__rvc_test_scores.csv", index=False)
    clips.assign(score=[1.0, 1.0]).to_csv(a / "vcv2_s7__cm09_scores.csv", index=False)  # no pair
    assert avg.build_clips(7, tmp_path) == ["rvc_test"]
    out = pd.read_csv(tmp_path / avg.OUT_CLIPS.format(seed=7) / "avg_s7__rvc_test_scores.csv")
    p = 1 / (1 + np.exp(-out["score"].to_numpy()))
    sig = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731
    assert p == pytest.approx([(sig(0.0) + sig(2.0)) / 2, (sig(2.0) + sig(-2.0)) / 2])


def test_threshold_files_are_named_after_the_model():
    path = acc.REPO / "configs/threshold_ssl_aasist_vcv2_s1234.yaml"
    expected = yaml.safe_load(path.read_text(encoding="utf-8"))["threshold"]
    assert acc.threshold("ssl_vcv2_s1234") == expected
    assert avg.THRESHOLD.format(seed=1234).endswith("threshold_ssl_aasist_vcv2avg_s1234.yaml")
