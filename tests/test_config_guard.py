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
