"""Tests for the CM02/CM04 scoring manifests.

Pure pandas, no audio. The properties that make an EER on these attacks mean
something: real and fake sides share speakers, only QA-passed fakes count, the
held-out attack stays in the eval pool, and no machine path is written.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data import scoring_manifests as sm


def _pools() -> pd.DataFrame:
    rows = [{"speaker": f"t{i}", "pool": "train"} for i in range(3)]
    rows += [{"speaker": f"e{i}", "pool": "eval"} for i in range(2)]
    return pd.DataFrame(rows)


def _qa(clips_speakers, fail_first=False) -> pd.DataFrame:
    rows = [
        {"clip": c, "speaker": s, "ok": not (fail_first and i == 0)}
        for i, (c, s) in enumerate(clips_speakers)
    ]
    return pd.DataFrame(rows)


def _eval_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "filepath": ["${DATA_ROOT}/clips/e0_a.wav", "${DATA_ROOT}/clips/e1_a.wav", "x.wav"],
            "label": ["bonafide", "bonafide", "spoof"],
            "speaker": ["e0", "e1", "e0"],
        }
    )


def _jobs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_utt_id": ["t0_u1", "t0_u1", "t1_u2"],
            "source_speaker": ["t0", "t0", "t1"],
        }
    )


def test_cm04_pairs_eval_bonafide_with_qa_passed_tortoise():
    qa = _qa([("tortoise_e0_1.wav", "e0"), ("tortoise_e1_1.wav", "e1")], fail_first=True)
    m = sm.cm04_manifest(_eval_manifest(), qa, _pools())
    assert (m["label"] == "bonafide").sum() == 2
    assert (m["label"] == "spoof").sum() == 1  # the failed clip is excluded
    assert set(m.loc[m.label == "spoof", "tool"]) == {"tortoise"}
    assert set(m["split"]) == {"eval"}


def test_cm04_refuses_a_clip_outside_the_eval_pool():
    qa = _qa([("tortoise_t0_1.wav", "t0")])
    with pytest.raises(sm.ScoringManifestError, match="outside the eval pool"):
        sm.cm04_manifest(_eval_manifest(), qa, _pools())


def test_cm04_refuses_a_speaker_with_no_real_clips():
    em = _eval_manifest()
    em = em[em.speaker != "e1"]
    qa = _qa([("tortoise_e1_1.wav", "e1")])
    with pytest.raises(sm.ScoringManifestError, match="no real clips"):
        sm.cm04_manifest(em, qa, _pools())


def test_cm02_uses_each_source_segment_once():
    qa = _qa([("rvc_t2_1.wav", "t2"), ("rvc_t2_2.wav", "t2")])
    m = sm.cm02_manifest(_jobs(), qa, _pools())
    bona = m[m.label == "bonafide"]
    assert len(bona) == 2  # t0_u1 converted twice is still one real clip
    assert bona["filepath"].str.endswith(("t0_u1.wav", "t1_u2.wav")).all()


def test_cm02_refuses_speakers_outside_the_train_pool():
    qa = _qa([("rvc_e0_1.wav", "e0")])
    with pytest.raises(sm.ScoringManifestError, match="outside the train pool"):
        sm.cm02_manifest(_jobs(), qa, _pools())


def test_paths_are_portable():
    qa = _qa([("tortoise_e0_1.wav", "e0")])
    m = sm.cm04_manifest(_eval_manifest(), qa, _pools())
    assert m["filepath"].str.startswith("${DATA_ROOT}/").all()
    assert not m["filepath"].str.contains(r"C:|\\|/kaggle/", regex=True).any()


def test_missing_files_reports_absent_audio(tmp_path):
    (tmp_path / "cm04").mkdir()
    (tmp_path / "cm04" / "here.wav").write_bytes(b"x")
    m = pd.DataFrame({"filepath": ["${DATA_ROOT}/cm04/here.wav", "${DATA_ROOT}/cm04/gone.wav"]})
    assert sm.missing_files(m, str(tmp_path)) == ["${DATA_ROOT}/cm04/gone.wav"]


def test_qa_passed_accepts_string_booleans():
    qa = pd.DataFrame({"clip": ["a", "b", "c"], "ok": ["True", "False", "true"]})
    assert list(sm.qa_passed(qa)["clip"]) == ["a", "c"]


def test_gate_split_is_speaker_disjoint_and_complete():
    m = pd.DataFrame({"speaker": ["a", "b", "c", "a", "d"], "label": ["bonafide"] * 5})
    fit, score = sm.gate_split(m)
    assert not set(fit.speaker) & set(score.speaker)
    assert len(fit) + len(score) == len(m)


def test_gate_split_is_order_independent():
    m = pd.DataFrame({"speaker": ["d", "a", "c", "b"], "label": ["spoof"] * 4})
    f1, _ = sm.gate_split(m)
    f2, _ = sm.gate_split(m.iloc[::-1].reset_index(drop=True))
    assert set(f1.speaker) == set(f2.speaker)
