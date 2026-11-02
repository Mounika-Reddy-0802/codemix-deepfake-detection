"""Every committed training config must obey the dataset rule on its real manifests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.training import config_guard as cg

CONFIGS = sorted(str(p).replace("\\", "/") for p in Path("configs").glob("train_*.yaml"))


def _rows(**overrides) -> pd.DataFrame:
    row = {
        "filepath": "${DATA_ROOT}/clips/a.wav",
        "label": "spoof",
        "language": "hi-en",
        "speaker": "t1",
        "source": "mucs2021",
        "tool": "xtts_v2",
        "condition": "clean",
        "split": "train",
    }
    row.update(overrides)
    return pd.DataFrame([row])


S3 = {"system": "s3_native", "lora": False}


@pytest.mark.parametrize("config", CONFIGS)
def test_committed_config_obeys_the_dataset_rule(config):
    report = cg.check_config(config)
    assert report.ok, report.problems


def test_the_spontaneous_config_trains_on_the_indicvoices_training_pool_only():
    """Decision D1: iv_train speakers in its training manifest, iv_dev in its dev set."""
    from src.data import indicvoices_pools as ivp

    manifest = Path("data/manifests/codemix_adapt_train_spont.csv")
    if not manifest.is_file():
        pytest.skip(
            "the spontaneous-recipe manifests are not built (python -m src.data.spont_recipe)"
        )
    report = cg.check_config("configs/train_lora_spont.yaml")
    assert report.ok, report.problems
    speakers = set(pd.read_csv(manifest, usecols=["speaker"])["speaker"].astype(str))
    indicvoices = {s for s in speakers if s.startswith("iv_")}
    assert indicvoices and indicvoices <= ivp.training_speakers()


def test_the_s3_configs_exist_and_are_checked():
    s3 = [c for c in CONFIGS if "s3_native" in c]
    assert len(s3) == 4
    for config in s3:
        report = cg.check_config(config)
        assert report.codemix_rows > 0 and report.rows == report.codemix_rows


def test_clean_s3_rows_pass():
    assert cg.check_rows(_rows(), S3, eval_speakers={"e1"}, rvc_test_speakers={"r9"}) == []


def test_code_mixed_rows_need_lora_or_s3():
    problems = cg.check_rows(_rows(), {"lora": False}, set(), set())
    assert any("neither LoRA" in p for p in problems)
    assert cg.check_rows(_rows(), {"lora": True}, set(), set()) == []


def test_s3_must_not_start_from_the_english_checkpoint():
    cfg = dict(S3, init_from="checkpoints/baseline/best.pt")
    assert any("pretrained encoder" in p for p in cg.check_rows(_rows(), cfg, set(), set()))


def test_tortoise_is_rejected():
    problems = cg.check_rows(_rows(tool="tortoise"), S3, set(), set())
    assert any("held-out tool" in p for p in problems)


def test_hiacc_is_evaluation_only():
    problems = cg.check_rows(_rows(source="hiacc", tool="none"), S3, set(), set())
    assert any("evaluation-only" in p for p in problems)


def test_field_recordings_are_evaluation_only():
    # the consent form promises the team's field recordings never train a model
    problems = cg.check_rows(_rows(source="field_validation", tool="none"), S3, set(), set())
    assert any("evaluation-only" in p for p in problems)


def test_eval_pool_speaker_is_rejected():
    problems = cg.check_rows(_rows(speaker="e1"), S3, eval_speakers={"e1"}, rvc_test_speakers=set())
    assert any("eval-pool" in p for p in problems)


def test_rvc_from_a_test_half_speaker_is_rejected():
    rows = _rows(tool="rvc", speaker="r9")
    problems = cg.check_rows(rows, S3, eval_speakers=set(), rvc_test_speakers={"r9"})
    assert any("RVC test half" in p for p in problems)


def test_a_config_whose_manifest_is_absent_is_unverifiable_not_passed(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/train_x.yaml").write_text("train_manifest: data/absent.csv\n")
    report = cg.check_config("configs/train_x.yaml", root=str(tmp_path))
    assert report.rows == 0 and report.unverifiable


def test_indicf5_is_rejected_like_tortoise():
    problems = cg.check_rows(_rows(tool="indicf5"), S3, set(), set())
    assert any("held-out tool" in p for p in problems)


def _guard_repo(tmp_path, train: pd.DataFrame, dev: pd.DataFrame) -> str:
    """A minimal repository: frozen pools, an IndicVoices pool file, one config."""
    (tmp_path / "configs").mkdir()
    (tmp_path / "data/manifests").mkdir(parents=True)
    pd.DataFrame({"speaker": ["e1"], "pool": ["eval"], "source": "mucs2021"}).to_csv(
        tmp_path / cg.POOLS, index=False
    )
    pd.DataFrame(
        {
            "speaker": ["iv_t", "iv_d", "iv_a"],
            "pool": ["iv_test", "iv_dev", "iv_train"],
            "source": "indicvoices",
        }
    ).to_csv(tmp_path / cg.POOLS_IV, index=False)
    train.to_csv(tmp_path / "data/train.csv", index=False)
    dev.to_csv(tmp_path / "data/dev.csv", index=False)
    (tmp_path / "configs/train_x.yaml").write_text(
        "lora: true\ntrain_manifest: data/train.csv\ndev_manifest: data/dev.csv\n"
    )
    return "configs/train_x.yaml"


def test_indicvoices_test_pool_is_rejected_anywhere(tmp_path):
    config = _guard_repo(tmp_path, _rows(speaker="t1"), _rows(speaker="iv_t", split="dev"))
    report = cg.check_config(config, root=str(tmp_path))
    assert any("eval-pool speakers" in p for p in report.problems)


def test_the_d1_pools_are_allowed_where_the_decision_puts_them(tmp_path):
    train = pd.concat([_rows(speaker="t1"), _rows(speaker="iv_a", language="hi")])
    config = _guard_repo(tmp_path, train, _rows(speaker="iv_d", split="dev", language="hi"))
    report = cg.check_config(config, root=str(tmp_path))
    assert report.ok, report.problems


def test_indicvoices_training_pool_in_the_dev_set_is_rejected(tmp_path):
    bad = _guard_repo(tmp_path, _rows(speaker="t1"), _rows(speaker="iv_a", split="dev"))
    report = cg.check_config(bad, root=str(tmp_path))
    assert any("training-pool speakers in the dev set" in p for p in report.problems)


def test_indicvoices_dev_pool_in_training_is_rejected(tmp_path):
    bad = _guard_repo(tmp_path, _rows(speaker="iv_d"), _rows(speaker="t1", split="dev"))
    report = cg.check_config(bad, root=str(tmp_path))
    assert any("dev-pool speakers in training" in p for p in report.problems)


def test_indicvoices_speakers_outside_the_pools_are_rejected_whatever_their_source():
    clone = _rows(speaker="iv_x", source="iv_xtts_train", tool="xtts_v2", language="hi")
    problems = cg.check_rows(clone, {"lora": True}, set(), set(), {"iv_a"})
    assert any("outside the iv_train and iv_dev pools" in p for p in problems)
    genuine = _rows(speaker="s9", source="indicvoices", tool="none", label="bonafide")
    problems = cg.check_rows(genuine, {"lora": True}, set(), set(), {"iv_a"})
    assert any("outside the iv_train and iv_dev pools" in p for p in problems)


def test_clones_of_a_training_pool_speaker_pass():
    clone = _rows(speaker="iv_a", source="iv_xtts_train", tool="xtts_v2", language="hi")
    assert cg.check_rows(clone, {"lora": True}, set(), set(), {"iv_a"}) == []


def test_hindi_rows_are_stage3_data_like_code_mixed_rows():
    hindi = _rows(speaker="iv_a", source="indicvoices", tool="none", language="hi")
    problems = cg.check_rows(hindi, {"lora": False}, set(), set(), {"iv_a"})
    assert any("neither LoRA" in p for p in problems)
