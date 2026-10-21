"""Tests for the SSL-AASIST port: weight renaming, RawBoost and the SSL training path."""

import numpy as np
import pandas as pd
import pytest

from src.baselines import aasist

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
sf = pytest.importorskip("soundfile")


def test_fairseq_names_map_onto_the_hugging_face_network():
    from src.baselines.ssl_aasist_model import _rename

    assert _rename("feature_extractor.conv_layers.2.0.weight") == (
        "feature_extractor.conv_layers.2.conv.weight"
    )
    assert _rename("feature_extractor.conv_layers.2.2.1.bias") == (
        "feature_extractor.conv_layers.2.layer_norm.bias"
    )
    assert _rename("layer_norm.weight") == "feature_projection.layer_norm.weight"
    assert _rename("post_extract_proj.bias") == "feature_projection.projection.bias"
    assert _rename("encoder.layers.7.self_attn.q_proj.weight") == (
        "encoder.layers.7.attention.q_proj.weight"
    )
    assert (
        _rename("encoder.layers.7.self_attn_layer_norm.bias") == "encoder.layers.7.layer_norm.bias"
    )
    assert (
        _rename("encoder.layers.7.fc1.weight")
        == "encoder.layers.7.feed_forward.intermediate_dense.weight"
    )
    assert _rename("encoder.layers.7.fc2.bias") == "encoder.layers.7.feed_forward.output_dense.bias"
    assert (
        _rename("encoder.layers.7.final_layer_norm.weight")
        == "encoder.layers.7.final_layer_norm.weight"
    )
    for dropped in (
        "mask_emb",
        "quantizer.vars",
        "project_q.weight",
        "final_proj.bias",
        "encoder.pos_conv.0.weight_g",
        "encoder.pos_conv.0.weight_v",
    ):
        assert _rename(dropped) is None


def test_xlsr_config_matches_the_300m_checkpoint():
    from src.baselines.ssl_aasist_model import xlsr_300m_config

    cfg = xlsr_300m_config()
    assert (cfg.hidden_size, cfg.num_hidden_layers, cfg.intermediate_size) == (1024, 24, 4096)
    assert cfg.do_stable_layer_norm and cfg.feat_extract_norm == "layer"
    assert not cfg.apply_spec_augment and cfg.layerdrop == 0.0


def test_rawboost_keeps_length_and_changes_the_signal():
    wave = (0.1 * np.sin(np.arange(aasist.NB_SAMP) / 20)).astype(np.float32)
    np.random.seed(0)
    boosted = aasist.rawboost(wave)
    assert boosted.shape == wave.shape and boosted.dtype == np.float32
    assert np.isfinite(boosted).all() and not np.allclose(boosted, wave)


class _Tiny(torch.nn.Module):
    """A stand-in for SSL-AASIST: logits only, and no ``Freq_aug`` argument."""

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(aasist.NB_SAMP, 2)

    def forward(self, x):
        return self.layer(x)


def test_ssl_training_path_runs_with_rawboost(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(6):
        label = "bonafide" if i % 2 else "spoof"
        sf.write(
            tmp_path / f"c{i}.wav", (0.1 * rng.standard_normal(20_000)).astype(np.float32), 16_000
        )
        rows.append({"filepath": f"${{DATA_ROOT}}/c{i}.wav", "label": label, "speaker": "s"})
    df = pd.DataFrame(rows)
    monkeypatch.setattr(aasist, "build", lambda weights, device, kind="aasist": _Tiny())
    best = aasist.train(
        df,
        df,
        str(tmp_path),
        str(tmp_path / "ckpt"),
        init=None,
        epochs=1,
        batch_size=2,
        kind="ssl",
        augment=True,
        log=lambda m: None,
    )
    assert 0.0 <= best <= 1.0 and (tmp_path / "ckpt" / "best.pt").is_file()
