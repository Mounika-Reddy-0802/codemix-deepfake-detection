"""LoRA adaptation for Stage-3: adapt Stage-1 to code-mixed speech cheaply.

Low-rank adapters are attached to the encoder's attention projections. The base
weights stay frozen; only the rank-``r`` factors (plus pooling + head) train, so
Stage-3 touches roughly 1-2% of the parameters instead of all 90.7M.

Hand-rolled rather than pulled from ``peft``, for the same reason the metrics and
the channel simulation are hand-rolled (P-004/P-005/P-006): CI installs only
ruff/pytest/numpy/pandas/pyyaml, and every piece that produces a paper number is
unit-tested here rather than trusted to a dependency.

**``lora_B`` is initialised to zeros.** With ``B = 0`` the adapter contributes
exactly nothing on the first forward pass, so an adapted model starts life
numerically identical to the Stage-1 checkpoint it was loaded from. That is what
makes the before/after comparison in the gap-closure table honest: any change in
EER is the adaptation, not a different starting point. :func:`apply_lora` is
verified against this property in ``tests/test_lora.py``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

#: Attention projections in wav2vec2 / WavLM / XLSR encoder layers. These are the
#: standard LoRA targets: adapting attention moves *what the model attends to*,
#: which is where a language shift shows up, without touching the feed-forward
#: capacity that holds the general speech representation.
DEFAULT_TARGETS: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "out_proj")

#: Submodules trained alongside the adapters. The classifier must move: Stage-1's
#: decision boundary was fitted on English studio audio, and leaving it frozen
#: asks the adapters to do the head's job as well as their own.
DEFAULT_TRAINABLE: tuple[str, ...] = ("pooling", "head")


class LoRALinear(nn.Module):
    """Frozen ``nn.Linear`` plus a trainable rank-``r`` update.

    Computes ``base(x) + (alpha / r) * B @ A @ x``. ``base`` keeps its original
    weights and is frozen in place; only ``lora_A`` and ``lora_B`` require grad.
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive, got {r}")
        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False

        self.r = int(r)
        self.scaling = float(alpha) / float(r)
        self.lora_A = nn.Parameter(torch.empty(self.r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_dropout: nn.Module = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @property
    def in_features(self) -> int:
        """Input width of the wrapped layer."""
        return int(self.base.in_features)

    @property
    def out_features(self) -> int:
        """Output width of the wrapped layer."""
        return int(self.base.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` ``(..., in_features)`` -> ``(..., out_features)``."""
        delta = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return self.base(x) + self.scaling * delta


def apply_lora(
    model: nn.Module,
    targets: tuple[str, ...] | list[str] = DEFAULT_TARGETS,
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> int:
    """Wrap every ``nn.Linear`` named in ``targets`` with :class:`LoRALinear`.

    Returns the number of layers wrapped. **Zero is an error, not a no-op** — it
    means the target names do not match this encoder's module names, and training
    would silently proceed with nothing adapting.
    """
    names = tuple(targets)
    wrapped = 0
    for _, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if child_name in names and isinstance(child, nn.Linear):
                setattr(parent, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                wrapped += 1
    if wrapped == 0:
        raise ValueError(
            f"apply_lora matched no layers for targets {names}. "
            "Check the encoder's module names before spending GPU hours."
        )
    return wrapped


def mark_only_lora_trainable(
    model: nn.Module, also_train: tuple[str, ...] | list[str] = DEFAULT_TRAINABLE
) -> None:
    """Freeze everything except the LoRA factors and ``also_train`` submodules."""
    prefixes = tuple(f"{name}." for name in also_train)
    for name, param in model.named_parameters():
        is_lora = "lora_A" in name or "lora_B" in name
        param.requires_grad = is_lora or name.startswith(prefixes)


def trainable_parameter_summary(model: nn.Module) -> dict[str, float]:
    """Count total vs trainable parameters, for the paper's "~1% trainable" claim."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": float(total),
        "trainable": float(trainable),
        "trainable_pct": 100.0 * trainable / total if total else 0.0,
    }


def lora_state_dict(
    model: nn.Module, also_train: tuple[str, ...] | list[str] = DEFAULT_TRAINABLE
) -> dict[str, torch.Tensor]:
    """Only the adapted tensors — a few MB, against 362 MB for the full model.

    Stage-3 produces one adapter per configuration; storing the frozen 90M-parameter
    encoder again with each of them would be 362 MB of identical bytes per run.
    """
    prefixes = tuple(f"{name}." for name in also_train)
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if "lora_A" in name or "lora_B" in name or name.startswith(prefixes)
    }
