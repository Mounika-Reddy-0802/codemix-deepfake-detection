"""The generator study's floors: the right sets, and no training-half speaker in them."""

from __future__ import annotations

import pandas as pd

from src.training import floor_final as fl
from src.training import floor_genstudy as fg


def test_the_study_sets_are_floor_final_sets_unchanged():
    assert set(fg.TEST_SETS) <= set(fl.TEST_SETS)
    assert {"cm09", "cm07", "new_tortoise"} <= set(fg.TEST_SETS)
    assert all(fg.TEST_SETS[k] == fl.TEST_SETS[k] for k in fg.TEST_SETS)


def test_held_out_drops_the_training_half_whatever_the_speaker_dtype():
    frame = pd.DataFrame({"speaker": [101, 102, 103], "filepath": ["a", "b", "c"]})
    out = fg.held_out(frame, {"101", "103"})
    assert out["filepath"].tolist() == ["b"]
