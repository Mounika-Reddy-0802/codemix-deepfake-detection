"""Structure smoke tests: the package tree is importable and complete.

These run in CI from Week 1. They import only the top-level packages (whose
``__init__`` files are intentionally empty, so no heavy ML/telephony deps are
pulled in) and check that every expected module file is present on disk. This
keeps CI fast and green while still catching accidental deletions or moves.
"""

import importlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

PACKAGES = [
    "src",
    "src.data",
    "src.models",
    "src.training",
    "src.inference",
    "src.utils",
]

EXPECTED_MODULES = [
    "src/data/preprocess.py",
    "src/data/channel_sim.py",
    "src/data/spoof_generation.py",
    "src/data/rvc_generation.py",
    "src/data/build_manifests.py",
    "src/data/dataset.py",
    "src/models/encoder.py",
    "src/models/pooling.py",
    "src/models/classifier.py",
    "src/models/detector.py",
    "src/models/lora.py",
    "src/training/train.py",
    "src/training/evaluate.py",
    "src/training/metrics.py",
    "src/inference/predict.py",
    "src/inference/streaming.py",
    "src/inference/export_onnx.py",
    "src/utils/audio_utils.py",
    "src/utils/seed.py",
    "src/utils/logging_utils.py",
    "src/utils/device.py",
    "src/utils/paths.py",
]


@pytest.mark.parametrize("pkg", PACKAGES)
def test_package_is_importable(pkg: str) -> None:
    assert importlib.import_module(pkg) is not None


@pytest.mark.parametrize("rel", EXPECTED_MODULES)
def test_expected_module_file_exists(rel: str) -> None:
    assert (REPO_ROOT / rel).is_file(), f"missing module: {rel}"
