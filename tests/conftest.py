"""Shared pytest fixtures.

The ethics gate (``src/data/ethics_gate.py``) refuses every generation entry point
until a signed mentor note exists, which is the correct production behaviour and
also means a test of anything *underneath* the gate never reaches its subject.
``open_ethics_gate`` opens it for one test so the layer below can be exercised.

Using it is always deliberate: `tests/test_ethics_gate.py` separately asserts that
each entry point really does call the gate, so neutralising it here cannot hide a
missing gate.
"""

import pytest


@pytest.fixture
def open_ethics_gate(monkeypatch):
    """Neutralise the ethics gate for one test.

    The generation modules import ``require_signoff`` inside the function body, so
    patching the attribute on the gate module is what takes effect at call time.
    """
    from src.data import ethics_gate

    monkeypatch.setattr(ethics_gate, "require_signoff", lambda *a, **kw: None)
    return ethics_gate
