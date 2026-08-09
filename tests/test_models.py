"""Shape tests for pooling + head (Week 3, owner M).

These need torch (not transformers), so they run wherever torch is installed and
skip cleanly in the light CI env. The full Detector (which downloads an SSL model)
is exercised by the smoke-train run, not unit tests.
"""

import pytest

torch = pytest.importorskip("torch")


def test_attentive_pooling_output_shape() -> None:
    from src.models.pooling import AttentiveStatsPooling

    pool = AttentiveStatsPooling(input_dim=32)
    x = torch.randn(4, 50, 32)  # (B, T, H)
    out = pool(x)
    assert out.shape == (4, 64)  # mean+std -> 2H


def test_attentive_pooling_respects_lengths() -> None:
    from src.models.pooling import AttentiveStatsPooling

    pool = AttentiveStatsPooling(input_dim=8)
    x = torch.randn(2, 10, 8)
    lengths = torch.tensor([10, 3])
    out = pool(x, lengths=lengths)
    assert out.shape == (2, 16)
    assert torch.isfinite(out).all()


def test_mlp_head_output_shape() -> None:
    from src.models.classifier import MLPHead

    head = MLPHead(input_dim=64, num_classes=2)
    logits = head(torch.randn(4, 64))
    assert logits.shape == (4, 2)


def test_train_config_loads_from_yaml() -> None:
    from src.training.train import load_config

    cfg = load_config("configs/train_baseline.yaml")
    assert cfg.encoder == "wav2vec2-base"
    assert cfg.seed == 1234
