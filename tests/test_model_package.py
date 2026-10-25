"""The publication package for the detector weights (week 14)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.inference import model_package as mp

CARD = "# Model card\n\n> **Status: draft, not published.**\n> More of the notice.\n\n## Model\n\nText.\n"


def test_model_card_has_metadata_drops_the_draft_notice_and_says_how_to_load() -> None:
    text = mp.model_card(CARD)
    assert text.startswith("---\nlicense: cc-by-nc-sa-4.0\n")
    assert "Status: draft" not in text and "More of the notice" not in text
    assert "## Model" in text and "## Loading this package" in text


def test_checksums_catch_a_changed_or_missing_file(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("one")
    (tmp_path / "b.txt").write_text("two")
    mp.write_sums(tmp_path)
    mp.check_sums(tmp_path)
    (tmp_path / "a.txt").write_text("changed")
    with pytest.raises(ValueError, match="a.txt"):
        mp.check_sums(tmp_path)
    (tmp_path / "a.txt").write_text("one")
    (tmp_path / "b.txt").unlink()
    with pytest.raises(ValueError, match="b.txt"):
        mp.check_sums(tmp_path)


def test_export_and_load_round_trip(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")  # model wheels, not installed in CI
    pytest.importorskip("safetensors")
    import numpy as np

    torch.manual_seed(0)
    model = torch.nn.Linear(3, 2)
    state = {
        "model": model.state_dict(),
        "config": {"encoder": "tiny", "lr": 1e-4, "out_dir": Path("checkpoints/x")},
        "metrics": {"dev_eer": np.float64(0.0123)},
    }
    config = mp.export(state, tmp_path / "pkg", {"threshold": 0.687, "fake_after": 5}, CARD)
    assert config["parameters"] == 8
    assert {p.name for p in (tmp_path / "pkg").iterdir()} == {
        mp.WEIGHTS,
        mp.CONFIG,
        mp.CARD,
        mp.SUMS,
    }
    saved = json.loads((tmp_path / "pkg" / mp.CONFIG).read_text())
    assert saved["training"]["out_dir"] == str(Path("checkpoints/x"))
    assert saved["dev_metrics"]["dev_eer"] == pytest.approx(0.0123)
    assert saved["operating_point"]["threshold"] == 0.687

    loaded, device = mp.load_package(
        tmp_path / "pkg", device="cpu", build=lambda encoder, st: torch.nn.Linear(3, 2)
    )
    assert device == "cpu"
    x = torch.randn(4, 3)
    assert torch.allclose(loaded(x), model(x))


def test_a_tampered_package_is_refused(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    state = {"model": torch.nn.Linear(3, 2).state_dict(), "config": {}, "metrics": {}}
    mp.export(state, tmp_path, {"threshold": 0.5}, CARD)
    (tmp_path / mp.CONFIG).write_text("{}")
    with pytest.raises(ValueError, match=mp.CONFIG):
        mp.load_package(tmp_path, device="cpu", build=lambda e, s: torch.nn.Linear(3, 2))
