"""Tests for src/inference/routing.py: the routing rule and its bookkeeping (no model)."""

import numpy as np
import pandas as pd
import pytest

from src.inference import routing


def test_language_code_strips_the_name() -> None:
    assert routing.language_code("en: English") == "en"
    assert routing.language_code("hi: Hindi") == "hi"


def test_only_english_goes_to_s1() -> None:
    assert routing.is_english("en")
    assert not routing.is_english("hi")
    assert not routing.is_english("ur")


def test_route_takes_s1_scores_where_the_gate_chose_s1() -> None:
    goes = pd.Series([True, False, False, True])
    s1 = pd.Series([0.1, 0.2, 0.3, 0.4])
    other = pd.Series([0.9, 0.8, 0.7, 0.6])
    assert routing.route(goes, s1, other).tolist() == [0.1, 0.8, 0.7, 0.4]


def _lid(languages, p_en, p_hi) -> pd.DataFrame:
    return pd.DataFrame({"language": languages, "p_en": p_en, "p_hi": p_hi})


def test_the_three_rules() -> None:
    lid = _lid(["en", "cy", "hi", "te"], [0.9, 0.3, 0.1, 0.2], [0.0, 0.1, 0.6, 0.3])
    assert routing.to_s1(lid, "top_english").tolist() == [True, False, False, False]
    assert routing.to_s1(lid, "english_over_hindi").tolist() == [True, True, False, False]
    assert routing.to_s1(lid, "not_indian").tolist() == [True, True, False, False]


def test_selection_maximises_the_worst_dev_set_and_breaks_ties_simply() -> None:
    english = _lid(["en", "cy", "en", "la"], [0.9, 0.4, 0.8, 0.3], [0.0, 0.1, 0.1, 0.2])
    hinglish = _lid(["hi", "te", "bn", "hi"], [0.0, 0.1, 0.0, 0.2], [0.7, 0.3, 0.2, 0.6])
    chosen, table = routing.select_rule({"en": (english, True), "hi": (hinglish, False)})
    assert table.loc["top_english", "worst"] == 50.0
    # english_over_hindi and not_indian are both perfect here; the earlier one wins
    assert chosen == "english_over_hindi"


def test_sample_is_fixed_and_balanced() -> None:
    manifest = pd.DataFrame(
        {"filepath": [f"c{i}.wav" for i in range(100)], "label": ["bonafide"] * 20 + ["spoof"] * 80}
    )
    a = routing.sample_manifest(manifest, 20)
    b = routing.sample_manifest(manifest, 20)
    assert a["filepath"].tolist() == b["filepath"].tolist()
    assert a["label"].value_counts().to_dict() == {"bonafide": 10, "spoof": 10}


class _FakeGate:
    def __init__(self) -> None:
        self.calls = 0

    def classify(self, waveform: np.ndarray) -> dict:
        self.calls += 1
        return {"language": "en", "p_en": 0.9, "p_hi": 0.05}


def test_identify_resumes_from_its_output(tmp_path, monkeypatch) -> None:
    sf = pytest.importorskip("soundfile")  # audio wheel, not installed in CI

    for i in range(3):
        sf.write(tmp_path / f"c{i}.wav", np.zeros(16_000, dtype=np.float32), 16_000)
    manifest = pd.DataFrame(
        {"filepath": [f"${{DATA_ROOT}}/c{i}.wav" for i in range(3)], "label": ["bonafide"] * 3}
    )
    out = tmp_path / "lid.csv"
    first = _FakeGate()
    routing.identify(manifest.iloc[:2], out, data_root=str(tmp_path), gate=first)
    second = _FakeGate()
    result = routing.identify(manifest, out, data_root=str(tmp_path), gate=second)
    assert (first.calls, second.calls) == (2, 1)
    assert len(result) == 3
    assert routing.confusion(result)["to_s1_percent"] == 100.0


def test_compare_scores_three_systems_on_the_same_clips() -> None:
    sample = pd.DataFrame(
        {
            "label": ["bonafide", "bonafide", "spoof", "spoof"],
            "language": ["en", "hi", "en", "hi"],
            "p_en": [0.9, 0.0, 0.8, 0.1],
            "p_hi": [0.0, 0.9, 0.1, 0.8],
            "s1": [0.9, 0.1, 0.2, 0.8],
            "other": [0.3, 0.9, 0.6, 0.1],
        }
    )
    result = routing.compare(sample, "top_english")
    # routed takes S1 on the English rows and the other model elsewhere: perfect here
    assert result["routed"]["eer"] == 0.0
    assert result["s1"]["eer"] > 0.0 and result["other"]["eer"] > 0.0
    assert result["gate"]["to_s1_percent"] == 50.0


def test_assemble_joins_on_file_name_and_refuses_gaps() -> None:
    lid = pd.DataFrame(
        {
            "filepath": ["${DATA_ROOT}/clips/a.wav", "${DATA_ROOT}/clips/b.wav"],
            "language": ["en", "hi"],
            "p_en": [0.9, 0.1],
            "p_hi": [0.0, 0.8],
        }
    )
    s1 = pd.DataFrame({"filepath": ["D:/x/clips/a.wav", "D:/x/clips/b.wav"], "score": [0.7, 0.2]})
    other = pd.DataFrame(
        {
            "filepath": ["/k/clips/b.wav", "/k/clips/a.wav"],
            "label": ["spoof", "bonafide"],
            "score": [0.1, 0.6],
        }
    )
    sample = routing.assemble(lid, s1, other).set_index("clip")
    assert sample.loc["a.wav", ["s1", "other", "label"]].tolist() == [0.7, 0.6, "bonafide"]
    with pytest.raises(ValueError):
        routing.assemble(lid, s1.iloc[:1], other)
