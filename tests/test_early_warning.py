"""The start-of-call gate: set from dev levels, applied only early, verdicts replayed."""

from __future__ import annotations

import pandas as pd
import pytest

from live_call.verdict_engine import EngineConfig
from src.inference import early_warning as ew
from src.inference import real_calls as rc


def _call(call_id: str, scores: list[float], levels: list[float]) -> pd.DataFrame:
    rows, ema = [], None
    for w, (s, level) in enumerate(zip(scores, levels, strict=True)):
        ema = s if ema is None else 0.5 * s + 0.5 * ema
        rows.append(
            {
                "filepath": f"${{DATA_ROOT}}/g711/{call_id}.wav",
                "window": w,
                "end_seconds": 4.0 + 2 * w,
                "level_dbfs": level,
                "score": s,
                "smoothed": ema,
            }
        )
    return pd.DataFrame(rows)


def test_gate_level_is_the_dev_quantile() -> None:
    dev = pd.DataFrame({"level_dbfs": [-40.0] + [-23.0] * 99})
    assert ew.gate_level(dev, share=0.01) == pytest.approx(-23.17, abs=0.01)
    assert ew.gate_level(dev, share=0.5) == pytest.approx(-23.0)


def test_quiet_start_gates_only_early_quiet_windows() -> None:
    # quiet start (-35), then speech (-23), then a quiet window late in the call (-35)
    levels = [-35.0, -23.0, -23.0, -23.0, -35.0, -23.0]
    windows = _call("C05", [0.0, 0.9, 0.9, 0.9, 0.9, 0.9], levels)
    gated = ew.quiet_start(windows, level=-30.0, first_seconds=10.0).set_index("window")
    assert pd.isna(gated.loc[0, "score"])  # quiet and early: silence
    assert gated.loc[4, "score"] == pytest.approx(0.9)  # quiet but late: still scored
    assert gated.loc[1, "smoothed"] == pytest.approx(0.9)  # the average restarts after the gap


def test_gate_can_clear_a_warning_that_started_in_the_quiet() -> None:
    config = EngineConfig(threshold=0.687, suspicious_after=2, fake_after=4)
    windows = _call("C05", [0.0, 0.5, 0.9, 0.9, 0.9, 0.9], [-35.0] + [-23.0] * 5)
    before = rc.call_verdicts(windows, config).iloc[0]
    after = rc.call_verdicts(ew.quiet_start(windows, -30.0), config).iloc[0]
    assert before["caution"] and not after["caution"]


def test_outcome_counts_and_timing() -> None:
    calls = pd.DataFrame(
        {
            "call_id": ["C19", "C20", "C25"],
            "kind": ["clone_replay"] * 3,
            "tool": ["xtts_v2", "xtts_v2", "chatterbox"],
            "caution": [True, True, False],
            "likely_cloned": [True, False, False],
            "first_caution_s": [8.0, 12.0, None],
            "first_likely_cloned_s": [12.0, None, None],
        }
    )
    rows = {(r["kind"], r["tool"]): r for r in ew.outcome(calls)}
    xtts = rows[("clone_replay", "xtts_v2")]
    assert (xtts["calls"], xtts["warned"], xtts["likely_cloned"]) == (2, 2, 1)
    assert xtts["median_first_caution_s"] == pytest.approx(10.0)
    assert xtts["warned_calls"] == ["C19", "C20"]
    assert rows[("clone_replay", "chatterbox")]["median_first_caution_s"] is None
