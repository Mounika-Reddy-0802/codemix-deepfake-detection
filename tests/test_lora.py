"""Unit tests for the Stage-3 LoRA adapters.

No SSL encoder is downloaded here: the adapters are plain ``nn.Linear`` wrappers,
so a small stand-in module with the same submodule names exercises every path and
the suite still runs on a laptop with no GPU and no network.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from src.models.lora import (  # noqa: E402
    DEFAULT_TARGETS,
    LoRALinear,
    apply_lora,
    lora_state_dict,
    mark_only_lora_trainable,
    trainable_parameter_summary,
)


class _Attention(nn.Module):
    """Stand-in with the submodule names wav2vec2/WavLM use."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        return self.out_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))


class _Tiny(nn.Module):
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.encoder = _Attention(dim)
        self.pooling = nn.Linear(dim, dim)
        self.head = nn.Linear(dim, 2)

    def forward(self, x):
        return self.head(self.pooling(self.encoder(x)))


def test_lora_starts_as_a_no_op():
    """B is zero-initialised, so an adapted model must reproduce the original.

    This is the property the gap-closure table depends on: if adaptation changed
    the output before a single gradient step, the "before" column would not be
    the Stage-1 model.
    """
    torch.manual_seed(0)
    model = _Tiny()
    x = torch.randn(4, 16)
    before = model(x)

    apply_lora(model, DEFAULT_TARGETS, r=4, alpha=8.0)
    after = model(x)

    assert torch.allclose(before, after, atol=1e-6)


def test_apply_lora_wraps_every_target():
    model = _Tiny()
    wrapped = apply_lora(model, DEFAULT_TARGETS, r=4)
    assert wrapped == 4
    assert isinstance(model.encoder.q_proj, LoRALinear)
    assert isinstance(model.encoder.out_proj, LoRALinear)
    assert not isinstance(model.pooling, LoRALinear)


def test_unmatched_targets_raise_rather_than_silently_adapting_nothing():
    model = _Tiny()
    with pytest.raises(ValueError, match="matched no layers"):
        apply_lora(model, ("query", "value"), r=4)


def test_rank_must_be_positive():
    with pytest.raises(ValueError, match="rank must be positive"):
        LoRALinear(nn.Linear(8, 8), r=0)


def test_base_weights_are_frozen_and_factors_are_not():
    layer = LoRALinear(nn.Linear(8, 8), r=2)
    assert not layer.base.weight.requires_grad
    assert not layer.base.bias.requires_grad
    assert layer.lora_A.requires_grad
    assert layer.lora_B.requires_grad


def test_only_adapters_pooling_and_head_train():
    model = _Tiny()
    apply_lora(model, DEFAULT_TARGETS, r=4)
    mark_only_lora_trainable(model, ("pooling", "head"))

    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert all("lora_" in n or n.startswith(("pooling.", "head.")) for n in trainable)
    # the frozen base of a wrapped layer must not have slipped back in
    assert "encoder.q_proj.base.weight" not in trainable
    assert any("lora_A" in n for n in trainable)
    assert "head.weight" in trainable


def test_trainable_share_is_a_small_fraction():
    model = _Tiny(dim=64)
    full = trainable_parameter_summary(model)["trainable_pct"]
    apply_lora(model, DEFAULT_TARGETS, r=2)
    mark_only_lora_trainable(model, ("pooling", "head"))
    adapted = trainable_parameter_summary(model)["trainable_pct"]

    assert full == pytest.approx(100.0)
    assert adapted < full


def test_adapters_actually_learn():
    """A gradient step must move the output; a frozen-everything bug would not."""
    torch.manual_seed(0)
    model = _Tiny()
    apply_lora(model, DEFAULT_TARGETS, r=4, alpha=8.0)
    mark_only_lora_trainable(model, ("pooling", "head"))

    x = torch.randn(8, 16)
    target = torch.randint(0, 2, (8,))
    before = model(x).detach().clone()

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.1)
    for _ in range(5):
        opt.zero_grad()
        nn.functional.cross_entropy(model(x), target).backward()
        opt.step()

    assert not torch.allclose(before, model(x), atol=1e-5)


def test_base_weights_do_not_move_during_training():
    torch.manual_seed(0)
    model = _Tiny()
    apply_lora(model, DEFAULT_TARGETS, r=4)
    mark_only_lora_trainable(model, ("pooling", "head"))
    original = model.encoder.q_proj.base.weight.detach().clone()

    x = torch.randn(8, 16)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.1)
    for _ in range(3):
        opt.zero_grad()
        nn.functional.cross_entropy(model(x), torch.randint(0, 2, (8,))).backward()
        opt.step()

    assert torch.equal(original, model.encoder.q_proj.base.weight)


def test_adapter_checkpoint_holds_only_adapted_tensors():
    model = _Tiny()
    apply_lora(model, DEFAULT_TARGETS, r=4)
    mark_only_lora_trainable(model, ("pooling", "head"))

    keys = set(lora_state_dict(model, ("pooling", "head")))
    assert any("lora_A" in k for k in keys)
    assert "head.weight" in keys
    assert not any(".base.weight" in k for k in keys)

    full = len(model.state_dict())
    assert len(keys) < full


def test_scaling_follows_alpha_over_r():
    layer = LoRALinear(nn.Linear(8, 8), r=4, alpha=16.0)
    assert layer.scaling == pytest.approx(4.0)


def test_wrapped_layer_reports_the_base_shape():
    layer = LoRALinear(nn.Linear(12, 5), r=2)
    assert layer.in_features == 12
    assert layer.out_features == 5
    assert layer(torch.randn(3, 12)).shape == (3, 5)
