"""Tests for src.inference.any_audio_eval on made-up scored rows (no model, no audio)."""

import pandas as pd

from src.inference import any_audio_eval as ev


def _rows(split, name, front, genuine_ok, fake_ok, n=10):
    rows = []
    for i in range(n):
        rows.append(
            {
                "filepath": f"{name}/g{i}.wav",
                "label": "bonafide",
                "tool": "none",
                "set": name,
                "split": split,
                "front_end": front,
                "verdict": "real" if i < genuine_ok else "cloned",
                "p_genuine": 0.9 if i < genuine_ok else 0.1,
            }
        )
        rows.append(
            {
                "filepath": f"{name}/f{i}.wav",
                "label": "spoof",
                "tool": "xtts_v2",
                "set": name,
                "split": split,
                "front_end": front,
                "verdict": "cloned" if i < fake_ok else "real",
                "p_genuine": 0.2 if i < fake_ok else 0.95,
            }
        )
    return rows


def test_front_end_is_chosen_on_dev_rows_only():
    rows = (
        _rows("dev", "dev", "g711", 10, 9)
        + _rows("dev", "dev", "none", 10, 6)
        # the test rows favour "none"; they must not influence the choice
        + _rows("test", "cm09", "g711", 5, 5)
        + _rows("test", "cm09", "none", 10, 10)
    )
    best, by = ev.choose_front_end(pd.DataFrame(rows))
    assert best == "g711"
    assert by["g711"] == 0.95 and by["none"] == 0.8


def test_ties_go_to_g711():
    rows = _rows("dev", "dev", "g711", 9, 9) + _rows("dev", "dev", "none", 9, 9)
    assert ev.choose_front_end(pd.DataFrame(rows))[0] == "g711"


def test_report_lists_every_set_kind_and_front_end():
    rows = _rows("dev", "dev", "g711", 10, 9) + _rows("test", "cm09", "g711", 8, 7)
    text = ev.report(pd.DataFrame(rows))
    assert "| dev (dev) | genuine | 100.0 (10) |" in text
    assert "| cm09 | xtts_v2 | 70.0 (10) |" in text
    assert "-> **g711**" in text


def test_errors_are_left_out_of_the_report():
    rows = _rows("test", "cm09", "g711", 10, 10)
    rows.append({**rows[0], "filepath": "bad.wav", "verdict": "error: unreadable"})
    assert "| cm09 | genuine | 100.0 (10) |" in ev.report(pd.DataFrame(rows))
