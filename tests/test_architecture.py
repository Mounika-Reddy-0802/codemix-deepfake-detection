"""The architecture figure must describe modules that actually exist."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.reporting import architecture as arch


@pytest.mark.parametrize(
    "label,module",
    [*arch.CARRIERS, *arch.CORE, *arch.DELIVERY],
)
def test_every_box_names_a_module_that_exists(label, module):
    """A diagram that claims a component the repository lacks is a lie in a figure."""
    assert Path(module).is_file(), f"{label!r} points at {module}, which does not exist"


def test_the_figure_is_written_in_both_formats(tmp_path):
    pytest.importorskip("matplotlib")
    written = arch.draw(str(tmp_path))
    assert {Path(p).suffix for p in written} == {".pdf", ".png"}
    for path in written:
        assert Path(path).stat().st_size > 5_000, f"{path} looks empty"
