"""Tests for the learned layer mix, the epoch-aware sampler and checkpoint rebuilding.

Torch-only (no transformers download), so they skip cleanly in the light CI env.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.models.encoder import LayerWeightedSum  # noqa: E402
from src.training.sampling import EpochSampler  # noqa: E402


def test_layer_mix_starts_as_the_plain_average():
    layers = [torch.full((2, 5, 3), float(i)) for i in range(4)]
    out = LayerWeightedSum(4)(layers)
    assert out.shape == (2, 5, 3)
    assert torch.allclose(out, torch.full((2, 5, 3), 1.5))


def test_layer_mix_weights_sum_to_one_and_learn():
    mix = LayerWeightedSum(3)
    layers = [torch.randn(2, 4, 6, requires_grad=False) for _ in range(3)]
    loss = mix(layers).pow(2).sum()
    loss.backward()
    assert mix.logits.grad is not None
    assert torch.isclose(mix.weights().sum(), torch.tensor(1.0))


def test_layer_mix_can_select_one_layer():
    mix = LayerWeightedSum(3)
    with torch.no_grad():
        mix.logits.copy_(torch.tensor([-50.0, 50.0, -50.0]))
    layers = [torch.full((1, 2, 2), float(i)) for i in range(3)]
    assert torch.allclose(mix(layers), torch.ones(1, 2, 2), atol=1e-5)


def test_layer_mix_refuses_the_wrong_number_of_layers():
    with pytest.raises(ValueError):
        LayerWeightedSum(3)([torch.zeros(1, 1, 1)] * 2)


def test_epoch_sampler_is_a_permutation_carrying_its_epoch():
    sampler = EpochSampler(10, seed=1)
    sampler.set_epoch(4)
    keys = list(sampler)
    assert sorted(i for i, _ in keys) == list(range(10))
    assert {e for _, e in keys} == {4}
    assert len(sampler) == 10


def test_epoch_sampler_order_is_reproducible_and_changes_with_epoch():
    a, b = EpochSampler(50, seed=1), EpochSampler(50, seed=1)
    a.set_epoch(2)
    b.set_epoch(2)
    assert list(a) == list(b)
    b.set_epoch(3)
    assert [i for i, _ in a] != [i for i, _ in b]


def test_build_detector_reads_the_layer_mix_from_the_checkpoint(monkeypatch):
    from src.models import detector as detector_module
    from src.training import evaluate

    built = []

    class FakeDetector:
        def __init__(self, encoder, layer_weighting=False):
            built.append((encoder, layer_weighting))

    monkeypatch.setattr(detector_module, "Detector", FakeDetector)
    evaluate.build_detector("wav2vec2-base", {"model": {"head.fc.weight": 0}})
    evaluate.build_detector("wav2vec2-base", {"model": {evaluate.LAYER_MIX_KEY: 0}})
    evaluate.build_detector("wav2vec2-base", {evaluate.LAYER_MIX_KEY: 0})  # bare state dict
    assert [lw for _, lw in built] == [False, True, True]


class _TinyEncoder(torch.nn.Module):
    def __init__(self, layer_mix: bool):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2)
        self.layer_mix = LayerWeightedSum(3) if layer_mix else None


class _TinyDetector(torch.nn.Module):
    def __init__(self, layer_mix: bool):
        super().__init__()
        self.encoder = _TinyEncoder(layer_mix)
        self.head = torch.nn.Linear(2, 2)


def test_stage1_checkpoint_loads_into_a_model_with_a_layer_mix():
    from src.training.train import load_initial_weights

    stage1 = _TinyDetector(layer_mix=False)
    model = _TinyDetector(layer_mix=True)
    load_initial_weights(model, {"model": stage1.state_dict()})
    assert torch.equal(model.head.weight, stage1.head.weight)
    assert torch.equal(model.encoder.layer_mix.logits, torch.zeros(3))  # starts equal


def test_any_other_mismatch_still_fails_the_load():
    from src.training.train import load_initial_weights

    stage1 = _TinyDetector(layer_mix=False).state_dict()
    del stage1["head.weight"]
    with pytest.raises(RuntimeError, match="head.weight"):
        load_initial_weights(_TinyDetector(layer_mix=True), stage1)
    extra = {**_TinyDetector(layer_mix=False).state_dict(), "stray.weight": torch.zeros(1)}
    with pytest.raises(RuntimeError, match="stray.weight"):
        load_initial_weights(_TinyDetector(layer_mix=False), extra)


def test_new_training_options_default_to_the_published_recipe():
    from src.training.train import load_config

    cfg = load_config("configs/train_lora_norm_rvc_channel.yaml")
    assert cfg.crop_per_epoch is False
    assert cfg.augment is None
    assert cfg.layer_weighting is False
