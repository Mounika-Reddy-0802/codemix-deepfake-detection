"""Tests for the ethics gate (Week 3, W3-T5, owner SK).

The rule the project actually has to keep: no voice cloning, not even a pilot,
until a mentor has signed the ethics note. These tests pin the three ways that
rule could quietly fail — a README being mistaken for a signature, an empty
placeholder file satisfying the check, and a generation entry point forgetting to
call the gate at all.

Stdlib only; no model is loaded anywhere here.
"""

from pathlib import Path

import pytest

from src.data import ethics_gate as gate


def _signed(tmp_path: Path, name: str = "mentor_signoff_2026-08-14.pdf") -> Path:
    path = tmp_path / name
    path.write_bytes(b"%PDF-1.4 scanned signature")
    return path


# --------------------------------------------------------------------------- #
# Closed by default
# --------------------------------------------------------------------------- #
def test_gate_is_closed_when_the_folder_is_empty(tmp_path: Path) -> None:
    assert gate.is_signed(str(tmp_path)) is False


def test_gate_is_closed_when_the_folder_does_not_exist(tmp_path: Path) -> None:
    assert gate.is_signed(str(tmp_path / "nope")) is False


def test_readme_is_not_a_signature(tmp_path: Path) -> None:
    # docs/ethics/README.md says what the sign-off must cover. It is not one.
    (tmp_path / "README.md").write_text("# what the sign-off must cover", encoding="utf-8")
    assert gate.is_signed(str(tmp_path)) is False


def test_empty_placeholder_is_not_a_signature(tmp_path: Path) -> None:
    (tmp_path / "mentor_signoff.pdf").write_bytes(b"")
    assert gate.is_signed(str(tmp_path)) is False


def test_unrelated_pdf_is_not_a_signature(tmp_path: Path) -> None:
    (tmp_path / "licences_summary.pdf").write_bytes(b"%PDF-1.4")
    assert gate.is_signed(str(tmp_path)) is False


# --------------------------------------------------------------------------- #
# Open when genuinely signed
# --------------------------------------------------------------------------- #
def test_signed_pdf_opens_the_gate(tmp_path: Path) -> None:
    _signed(tmp_path)
    assert gate.is_signed(str(tmp_path)) is True


def test_a_photo_of_the_signature_also_counts(tmp_path: Path) -> None:
    (tmp_path / "mentor_signoff.jpg").write_bytes(b"\xff\xd8\xff scan")
    assert gate.is_signed(str(tmp_path)) is True


def test_status_reports_the_artefact_it_found(tmp_path: Path) -> None:
    _signed(tmp_path, "mentor_signoff_v2.pdf")
    status = gate.signoff_status(str(tmp_path))
    assert status.signed is True
    assert status.artefacts == ("mentor_signoff_v2.pdf",)
    assert "OPEN" in status.describe()


def test_closed_status_names_the_directory(tmp_path: Path) -> None:
    assert str(tmp_path) in gate.signoff_status(str(tmp_path)).describe()


# --------------------------------------------------------------------------- #
# require_signoff
# --------------------------------------------------------------------------- #
def test_require_signoff_raises_when_unsigned(tmp_path: Path) -> None:
    with pytest.raises(gate.EthicsGateError):
        gate.require_signoff(str(tmp_path))


def test_require_signoff_is_silent_when_signed(tmp_path: Path) -> None:
    _signed(tmp_path)
    gate.require_signoff(str(tmp_path))  # must not raise


def test_error_names_the_blocked_action_and_the_fix(tmp_path: Path) -> None:
    with pytest.raises(gate.EthicsGateError) as excinfo:
        gate.require_signoff(str(tmp_path), action="XTTS-v2 pilot")
    message = str(excinfo.value)
    assert "XTTS-v2 pilot" in message
    assert "no override" in message


BYPASS_WORDS = ("force", "override", "skip", "bypass", "disable", "ignore")


def test_no_public_name_looks_like_a_bypass() -> None:
    # A bypass flag would make the gate decorative. There must not be one.
    for name in vars(gate):
        if name.startswith("_"):
            continue
        assert not any(w in name.lower() for w in BYPASS_WORDS), f"gate exposes {name}"


def test_no_function_takes_a_bypass_argument() -> None:
    import inspect

    for name, obj in vars(gate).items():
        if not inspect.isfunction(obj) or obj.__module__ != gate.__name__:
            continue
        for param in inspect.signature(obj).parameters:
            assert not any(w in param.lower() for w in BYPASS_WORDS), f"{name}({param})"


def test_gate_cannot_be_opened_by_an_environment_variable() -> None:
    # The realistic sneaky route: SKIP_ETHICS=1 in a Kaggle secret. The gate reads
    # the filesystem and nothing else, so it must not even reach os.
    import inspect

    source = inspect.getsource(gate)
    assert "os.environ" not in source
    assert "os.getenv" not in source
    assert not hasattr(gate, "os"), "gate imports os; it must not read env vars"


# --------------------------------------------------------------------------- #
# Generation entry points actually call it
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("module", "function"),
    [
        ("src.data.spoof_generation", "load_xtts"),
        ("src.data.spoof_generation", "generate_batch"),
        ("src.data.heldout_tts", "load_tortoise"),
        ("src.data.heldout_tts", "generate_heldout_batch"),
    ],
)
def test_generation_entry_points_are_gated(module: str, function: str) -> None:
    import importlib
    import inspect

    mod = importlib.import_module(module)
    source = inspect.getsource(getattr(mod, function))
    assert "require_signoff" in source, f"{module}.{function} is not gated"


def test_real_repo_gate_state_is_self_consistent() -> None:
    """The gate's answer must match what is actually on disk.

    The signed note carries real handwritten signatures, so it is deliberately
    **never committed** (see .gitignore). That means this is True on a team
    machine that holds the scan and False in CI or on a fresh clone -- both are
    valid states. What must never happen is the three answers disagreeing, which
    would mean the gate could report "open" while nothing had been signed.
    """
    status = gate.signoff_status()
    assert status.signed == bool(gate.find_signoff())
    assert status.signed == gate.is_signed()
    if status.signed:
        assert all(a.startswith("mentor_signoff") for a in status.artefacts)
        gate.require_signoff()  # must not raise when a note is present
    else:
        with pytest.raises(gate.EthicsGateError):
            gate.require_signoff()
