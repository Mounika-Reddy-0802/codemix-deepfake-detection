"""Device-selection tests.

These run in CI on a CPU-only runner, so the CUDA branches are exercised by
monkeypatching :func:`src.utils.device.cuda_available` rather than by needing a
GPU. The behaviour that matters is the asymmetry: auto-detection falls back to the
CPU quietly, an *explicit* request for an absent GPU raises — a Colab run that
silently drops to CPU burns a session before anyone notices.
"""

from __future__ import annotations

import pytest

from src.utils import device as dev


@pytest.fixture(autouse=True)
def _no_env_override(monkeypatch):
    """Keep a developer's own $DFD_DEVICE out of the assertions."""
    monkeypatch.delenv(dev.DEVICE_ENV, raising=False)


def test_auto_falls_back_to_cpu_without_accelerators(monkeypatch) -> None:
    monkeypatch.setattr(dev, "cuda_available", lambda: False)
    monkeypatch.setattr(dev, "mps_available", lambda: False)
    assert dev.resolve_device() == dev.CPU


def test_auto_prefers_cuda_when_present(monkeypatch) -> None:
    monkeypatch.setattr(dev, "cuda_available", lambda: True)
    assert dev.resolve_device() == dev.CUDA
    assert dev.resolve_device("auto") == dev.CUDA


def test_env_var_pins_the_device(monkeypatch) -> None:
    monkeypatch.setenv(dev.DEVICE_ENV, "cpu")
    monkeypatch.setattr(dev, "cuda_available", lambda: True)
    assert dev.resolve_device() == dev.CPU


def test_argument_beats_the_env_var(monkeypatch) -> None:
    monkeypatch.setenv(dev.DEVICE_ENV, "cuda")
    monkeypatch.setattr(dev, "cuda_available", lambda: False)
    assert dev.resolve_device("cpu") == dev.CPU


def test_explicit_cuda_without_a_gpu_raises(monkeypatch) -> None:
    monkeypatch.setattr(dev, "cuda_available", lambda: False)
    with pytest.raises(dev.DeviceUnavailableError):
        dev.resolve_device("cuda")


def test_unknown_device_is_rejected() -> None:
    with pytest.raises(ValueError):
        dev.resolve_device("tpu")


def test_device_index_survives_resolution(monkeypatch) -> None:
    monkeypatch.setattr(dev, "cuda_available", lambda: True)
    assert dev.resolve_device("cuda:1") == "cuda:1"
    assert dev.device_family("cuda:1") == dev.CUDA
    assert dev.is_cuda("cuda:1")


def test_amp_is_off_on_cpu_even_when_requested() -> None:
    assert dev.amp_enabled(dev.CPU, want_amp=True) is False
    assert dev.amp_enabled("cuda:0", want_amp=True) is True
    assert dev.amp_enabled("cuda", want_amp=False) is False


def test_dataloader_kwargs_track_the_device() -> None:
    cpu = dev.dataloader_kwargs(dev.CPU, num_workers=0)
    assert cpu == {"num_workers": 0, "pin_memory": False}

    gpu = dev.dataloader_kwargs("cuda", num_workers=4)
    assert gpu["pin_memory"] is True
    assert gpu["num_workers"] == 4
    assert gpu["persistent_workers"] is True


def test_describe_names_the_cpu_case() -> None:
    assert "cpu" in dev.describe(dev.CPU)
