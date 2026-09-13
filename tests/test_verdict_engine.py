"""Verdict state machine and escalation ladder (owner SK)."""

from __future__ import annotations

import pytest

from live_call.verdict_engine import (
    EngineConfig,
    EventKind,
    State,
    VerdictEngine,
    load_config,
)

CFG = EngineConfig(threshold=0.6, margin=0.05, min_windows=2, suspicious_after=2, fake_after=4)


def _feed(engine: VerdictEngine, scores):
    events = []
    for i, s in enumerate(scores, 1):
        events += engine.update(2.0 * i + 2.0, s)
    return events


def test_listening_until_enough_speech() -> None:
    engine = VerdictEngine(CFG)
    engine.update(4.0, 0.9)
    assert engine.state == State.LISTENING
    engine.update(6.0, 0.9)
    assert engine.state == State.GENUINE


def test_ladder_beep_then_sms() -> None:
    engine = VerdictEngine(CFG)
    events = _feed(engine, [0.9, 0.9, 0.2, 0.2, 0.2, 0.2])
    assert [e.kind for e in events] == [EventKind.BEEP, EventKind.SMS]
    assert engine.state == State.LIKELY_FAKE


def test_one_bad_window_does_not_alert() -> None:
    engine = VerdictEngine(CFG)
    assert _feed(engine, [0.9, 0.9, 0.2, 0.9, 0.2, 0.9]) == []
    assert engine.state == State.GENUINE


def test_silence_neither_advances_nor_resets() -> None:
    engine = VerdictEngine(CFG)
    events = _feed(engine, [0.9, 0.9, 0.2, None, None, 0.2])
    assert [e.kind for e in events] == [EventKind.BEEP]
    assert engine.summary.skipped_windows == 2


def test_hysteresis_band_does_not_recover() -> None:
    engine = VerdictEngine(CFG)
    _feed(engine, [0.9, 0.9, 0.2, 0.2])
    assert engine.state == State.SUSPICIOUS
    _feed(engine, [0.62, 0.62, 0.62, 0.62])  # above threshold, inside the margin
    assert engine.state == State.SUSPICIOUS


def test_recovery_needs_a_run_above_the_margin() -> None:
    engine = VerdictEngine(CFG)
    _feed(engine, [0.9, 0.9, 0.2, 0.2, 0.9, 0.9])
    assert engine.state == State.SUSPICIOUS
    engine.update(20.0, 0.9)
    assert engine.state == State.GENUINE


def test_no_second_beep_when_stepping_down_from_fake() -> None:
    engine = VerdictEngine(CFG)
    events = _feed(engine, [0.9, 0.9, 0.2, 0.2, 0.2, 0.2, 0.2])
    assert [e.kind for e in events].count(EventKind.BEEP) == 1


def test_summary_records_the_call() -> None:
    engine = VerdictEngine(CFG)
    _feed(engine, [0.9, 0.9, 0.2, 0.2, 0.2, 0.2, None])
    summary_event = engine.end_call()
    s = engine.summary
    assert summary_event.kind == EventKind.SUMMARY
    assert s.peak_state == State.LIKELY_FAKE
    assert s.scored_windows == 6 and s.fake_leaning_windows == 4
    assert s.worst_score == pytest.approx(0.2)
    assert s.fake_fraction == pytest.approx(4 / 6)
    with pytest.raises(RuntimeError):
        engine.update(30.0, 0.9)


def test_config_rejects_nonsense() -> None:
    with pytest.raises(ValueError):
        EngineConfig(threshold=1.0)
    with pytest.raises(ValueError):
        EngineConfig(threshold=0.5, suspicious_after=5, fake_after=2)


def test_missing_threshold_file_refuses_to_guess(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="results freeze"):
        load_config(str(tmp_path / "threshold.yaml"))


def test_threshold_file_is_read(tmp_path) -> None:
    path = tmp_path / "threshold.yaml"
    path.write_text("threshold: 0.97\nfake_after: 6\nsource: det curve\n")
    cfg = load_config(str(path))
    assert cfg.threshold == 0.97 and cfg.fake_after == 6
