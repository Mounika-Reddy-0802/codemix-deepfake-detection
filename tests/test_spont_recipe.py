"""Tests for src/data/spont_recipe.py on small synthetic manifests."""

import pandas as pd
import pytest

from src.data import spont_recipe as sr


def _rows(speakers, label, tool, condition, split, prefix):
    return pd.DataFrame(
        {
            "filepath": [f"${{DATA_ROOT}}/clips/{prefix}{s}.wav" for s in speakers],
            "label": label,
            "language": "hi",
            "speaker": speakers,
            "source": "x",
            "tool": tool,
            "condition": condition,
            "split": split,
            "utt_id": [f"{prefix}{s}" for s in speakers],
            "text": "kept out",
            "gender": "kept out",
        }
    )


def _amr(frame: pd.DataFrame) -> pd.DataFrame:
    """The AMR-NB copy: same clips in their own folder, as src.data.amr_channel writes them."""
    return frame.assign(
        condition="amr_nb_12k2", filepath=frame["filepath"].str.replace("/clips/", "/clips/amr/")
    )


@pytest.fixture
def manifests(tmp_path):
    d = tmp_path
    _rows(["m1", "m2"], "bonafide", "none", "channel_g711_20db", "train", "f").to_csv(
        d / "codemix_adapt_train_final.csv", index=False
    )
    _rows(["m3"], "bonafide", "none", "channel_g711_20db", "dev", "d").to_csv(
        d / "codemix_adapt_dev_channel20.csv", index=False
    )
    genuine = pd.concat(
        [
            _rows(["iv_a"], "bonafide", "none", "channel_g711_20db", "iv_train", "g"),
            _rows(["iv_d"], "bonafide", "none", "channel_g711_20db", "iv_dev", "g"),
            _rows(["iv_t"], "bonafide", "none", "channel_g711_20db", "iv_test", "g"),
        ]
    )
    genuine.to_csv(d / "indicvoices_genuine_channel20.csv", index=False)
    _amr(genuine).to_csv(d / "indicvoices_genuine_amr.csv", index=False)
    for pool, spk in (("train", "iv_a"), ("dev", "iv_d"), ("test", "iv_t")):
        clones = _rows([spk], "spoof", "xtts_v2", "channel_g711_20db", "x", f"c{pool}")
        clones.to_csv(d / f"iv_xtts_{pool}_channel20.csv", index=False)
        _amr(clones).to_csv(d / f"iv_xtts_{pool}_amr.csv", index=False)
    pd.DataFrame(
        {
            "speaker": ["iv_a", "iv_d", "iv_t"],
            "pool": ["iv_train", "iv_dev", "iv_test"],
            "source": "indicvoices",
        }
    ).to_csv(d / "indicvoices_pools.csv", index=False)
    return d


def test_training_adds_train_pool_genuine_and_clones_on_both_codecs(manifests) -> None:
    train, dev = sr.build(str(manifests))
    iv = train[train["speaker"].str.startswith("iv_")]
    assert set(iv["speaker"]) == {"iv_a"}
    assert sorted(zip(iv["label"], iv["condition"], strict=True)) == [
        ("bonafide", "amr_nb_12k2"),
        ("bonafide", "channel_g711_20db"),
        ("spoof", "amr_nb_12k2"),
        ("spoof", "channel_g711_20db"),
    ]
    assert set(iv["split"]) == {"train"}
    assert list(train.columns) == sr.COLUMNS  # no transcripts or demographics
    sr.check(train, dev, pd.read_csv(manifests / "indicvoices_pools.csv"))


def test_dev_adds_the_dev_pool_only(manifests) -> None:
    _, dev = sr.build(str(manifests))
    assert set(dev.loc[dev["speaker"].str.startswith("iv_"), "speaker"]) == {"iv_d"}
    assert set(dev["split"]) == {"dev"}


def test_a_test_speaker_in_training_is_refused(manifests) -> None:
    train, dev = sr.build(str(manifests))
    leaked = pd.concat([train, train.assign(speaker="iv_t").head(1)])
    with pytest.raises(ValueError, match="pool roles"):
        sr.check(leaked, dev, pd.read_csv(manifests / "indicvoices_pools.csv"))


def test_test_manifests_hold_the_test_speakers_genuine_and_clones_per_codec(manifests) -> None:
    tests = sr.build_test(str(manifests))
    assert set(tests) == set(sr.TEST_OUT.values())
    for name, frame in tests.items():
        assert set(frame["speaker"]) == {"iv_t"}
        assert sorted(frame["label"]) == ["bonafide", "spoof"]
        assert set(frame["split"]) == {"eval"}
        assert list(frame.columns) == sr.COLUMNS
        amr = name.endswith("_amr")
        assert frame["filepath"].str.contains("/clips/amr/").all() == amr
    sr.check_test(tests, pd.read_csv(manifests / "indicvoices_pools.csv"))


def test_test_manifests_refuse_other_pools(manifests) -> None:
    tests = sr.build_test(str(manifests))
    pools = pd.read_csv(manifests / "indicvoices_pools.csv")
    bad = {
        "x": pd.concat(
            [tests["score_iv_test_amr"], tests["score_iv_test_amr"].assign(speaker="iv_a")]
        )
    }
    with pytest.raises(ValueError, match="outside iv_test"):
        sr.check_test(bad, pools)
