"""Smoke tests for src.training.floor_new_sets."""

import pandas as pd

from src.training import floor_new_sets as fns


def test_select_keeps_genuine_and_the_named_tool():
    frame = pd.DataFrame(
        {
            "label": ["bonafide", "spoof", "spoof"],
            "tool": ["none", "xtts_v2", "tortoise"],
            "filepath": ["a", "b", "c"],
        }
    )
    assert fns.select(frame, ("tortoise",))["filepath"].tolist() == ["a", "c"]


def test_every_system_names_a_training_manifest_that_exists():
    for system in fns.SYSTEMS:
        assert (fns.Path("data/manifests") / system.train_manifest).is_file(), system.name


def test_markdown_row():
    row = {
        "system": "S",
        "test_set": "cm07",
        "eer_a": 0.1,
        "eer_b": 0.3,
        "diff": -0.2,
        "diff_ci": [-0.25, -0.15],
        "verdict": "beats floor",
    }
    assert "| S | cm07 | 10.00 | 30.00 | -20.00 [-25.00, -15.00] | beats floor |" in fns.markdown(
        [row]
    )
