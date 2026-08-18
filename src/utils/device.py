"""Device selection shared by training, generation and inference.

The project is developed on a CPU-only laptop and trained on a Colab/Kaggle GPU
from the *same* checkout, so no entry point may hard-code ``"cuda"``. Everything
asks :func:`resolve_device` instead, and the answer flows into AMP, the gradient
scaler and the DataLoader options through the helpers below.

``DFD_DEVICE`` overrides the automatic choice. Setting it to ``cpu`` on a GPU box
reproduces a laptop run exactly; setting it to ``cuda`` makes a run **fail loudly**
if no GPU is visible, which is what you want on Colab — a silent fall back to CPU
turns a two-hour job into a two-day one and nobody notices until the session dies.

Torch is imported lazily so this module stays importable in CI without the ML
stack (``resolve_device()`` then simply reports ``cpu``).
"""

from __future__ import annotations

import os

#: Environment variable that pins the device for every entry point.
DEVICE_ENV = "DFD_DEVICE"

CPU = "cpu"
CUDA = "cuda"
MPS = "mps"

_KNOWN_FAMILIES = (CPU, CUDA, MPS)


class DeviceUnavailableError(RuntimeError):
    """Raised when a device was asked for explicitly and is not present."""


def cuda_available() -> bool:
    """Whether this process can see a CUDA device (False if torch is absent)."""
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def mps_available() -> bool:
    """Whether Apple Metal is usable (False if torch is absent or built without it)."""
    try:
        import torch
    except ImportError:
        return False
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def device_family(device: str) -> str:
    """Family of a device string: ``"cuda:0"`` -> ``"cuda"``."""
    return str(device).split(":", 1)[0].strip().lower()


def is_cuda(device: str) -> bool:
    """Whether ``device`` names a CUDA device."""
    return device_family(device) == CUDA


def resolve_device(prefer: str | None = None) -> str:
    """Pick the device to run on.

    Precedence: the ``prefer`` argument, then ``$DFD_DEVICE``, then auto-detection
    (CUDA, else MPS, else CPU). An explicit request for an absent accelerator
    raises :class:`DeviceUnavailableError` rather than quietly using the CPU.
    """
    requested = (prefer or os.environ.get(DEVICE_ENV) or "auto").strip().lower()
    if requested in {"", "auto"}:
        if cuda_available():
            return CUDA
        if mps_available():
            return MPS
        return CPU

    family = device_family(requested)
    if family not in _KNOWN_FAMILIES:
        raise ValueError(f"unknown device {requested!r} (use auto | cpu | cuda[:N] | mps)")
    if family == CUDA and not cuda_available():
        raise DeviceUnavailableError(
            f"device {requested!r} was requested but no CUDA device is visible. "
            "On Colab: Runtime -> Change runtime type -> GPU. "
            f"To run on the CPU instead, unset {DEVICE_ENV} or set it to 'cpu'."
        )
    if family == MPS and not mps_available():
        raise DeviceUnavailableError(f"device {requested!r} was requested but MPS is unavailable")
    return requested


def amp_enabled(device: str, want_amp: bool = True) -> bool:
    """Whether mixed precision should actually be used.

    AMP is a CUDA optimisation; on CPU it costs accuracy for no speed, so a config
    asking for ``amp: true`` is honoured only when the run really landed on a GPU.
    """
    return bool(want_amp) and is_cuda(device)


def make_grad_scaler(device: str, enabled: bool):
    """Build a gradient scaler for ``device`` across torch versions.

    ``torch.cuda.amp.GradScaler`` is deprecated from torch 2.4; the replacement
    takes the device type as its first argument. Both spellings are tried so the
    laptop's torch and Colab's torch both work.
    """
    import torch

    family = device_family(device)
    try:
        return torch.amp.GradScaler(family, enabled=enabled)
    except (AttributeError, TypeError):  # torch < 2.4
        return torch.cuda.amp.GradScaler(enabled=enabled)


def dataloader_kwargs(device: str, num_workers: int = 0) -> dict:
    """DataLoader options tuned for the resolved device.

    ``pin_memory`` only helps when pages are copied to a GPU, and worker processes
    are kept alive between epochs only when there are any (spawning them per epoch
    on Windows is pure overhead).
    """
    workers = max(0, int(num_workers))
    kwargs: dict = {"num_workers": workers, "pin_memory": is_cuda(device)}
    if workers > 0:
        kwargs["persistent_workers"] = True
    return kwargs


def describe(device: str | None = None) -> str:
    """One-line, log-friendly description of the device a run will use."""
    resolved = device or resolve_device()
    if not is_cuda(resolved):
        return f"device={resolved} (no GPU; training here is for smoke tests only)"
    try:
        import torch

        index = int(resolved.split(":", 1)[1]) if ":" in resolved else 0
        name = torch.cuda.get_device_name(index)
        total_gb = torch.cuda.get_device_properties(index).total_memory / 1024**3
        return f"device={resolved} ({name}, {total_gb:.1f} GiB)"
    except Exception:  # noqa: BLE001 - a description must never break a run
        return f"device={resolved}"


def main() -> None:
    """CLI: ``python -m src.utils.device`` prints what this machine would use."""
    print(describe())


if __name__ == "__main__":
    main()
