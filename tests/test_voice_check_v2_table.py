"""Voice check v2's full table: training speakers left out exactly where they can occur."""

from __future__ import annotations

from src.reporting import any_audio_accuracy as acc
from src.reporting import voice_check_v2 as rule
from src.reporting import voice_check_v2_table as vt


def _fake(monkeypatch) -> list[tuple[str, str, frozenset]]:
    calls = []

    def accuracy(model, key, rows, expected, exclude=frozenset()):
        calls.append((model, key, exclude))
        return 90.0, 10

    monkeypatch.setattr(acc, "accuracy", accuracy)
    monkeypatch.setattr(
        acc, "call_counts", lambda m: {(k, t): (6, 6) for k, t, _ in acc.CALL_KINDS}
    )
    monkeypatch.setattr(acc, "threshold", lambda m: 0.5)
    return calls


def test_every_model_and_condition_has_a_cell(monkeypatch):
    _fake(monkeypatch)
    text = vt.table(frozenset({"s1", "s2"}))
    assert text.splitlines()[0] == "| Condition | " + " | ".join(vt.MODELS) + " |"
    measured = [c for c in acc.CONDITIONS if c[1] is not None]
    assert text.count("90.0 (n=10)") == len(vt.MODELS) * len(measured)
    assert text.count("6/6") == len(vt.MODELS) * len(acc.CALL_KINDS)
    assert "the 2 `gen_train` speakers are left out" in text


def test_speakers_are_left_out_of_the_pool_sets_only(monkeypatch):
    calls = _fake(monkeypatch)
    out = frozenset({"s1"})
    vt.rows(out)
    for _, key, exclude in calls:
        assert exclude == (out if key in vt.POOL_SETS else frozenset())
    # the reference and both candidates are measured on the same clips
    by_key: dict[str, set] = {}
    for _, key, exclude in calls:
        by_key.setdefault(key, set()).add(exclude)
    assert all(len(v) == 1 for v in by_key.values())


def test_the_rule_columns_come_first_and_last():
    assert vt.MODELS[0] == rule.REFERENCE
    assert vt.MODELS[-2:] == rule.CANDIDATES


def test_the_rule_and_the_table_leave_out_the_same_g711_sets():
    assert rule.POOL_SETS <= vt.POOL_SETS
    assert {s for s in vt.POOL_SETS if not s.endswith("_amr")} == rule.POOL_SETS
