"""Smoke tests for src.baselines.aasist (the AASIST outside baseline)."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.baselines import aasist


def test_fixed_length_crops_and_repeats():
    long = np.arange(aasist.NB_SAMP + 500, dtype=np.float32)
    assert np.array_equal(aasist.fixed_length(long), long[: aasist.NB_SAMP])
    cropped = aasist.fixed_length(long, np.random.default_rng(0))
    assert len(cropped) == aasist.NB_SAMP
    short = np.arange(1000, dtype=np.float32)
    repeated = aasist.fixed_length(short)
    assert len(repeated) == aasist.NB_SAMP and repeated[1000] == short[0]


def test_published_weights_are_the_expected_file():
    digest = hashlib.sha256(Path(aasist.WEIGHTS).read_bytes()).hexdigest()
    assert digest == aasist.WEIGHTS_SHA256


def test_model_scores_one_batch():
    torch = pytest.importorskip("torch")
    model = aasist.build(aasist.WEIGHTS).eval()
    with torch.no_grad():
        _, logits = model(torch.zeros(1, aasist.NB_SAMP))
    assert tuple(logits.shape) == (1, 2)


def test_result_switches_to_false_alarms_for_genuine_only():
    genuine = pd.DataFrame(
        {"label": ["bonafide"] * 4, "speaker": ["a", "a", "b", "b"], "score": [1.0, -1.0, 2.0, 3.0]}
    )
    summary = aasist.result(genuine)["summary"]
    assert summary["false_alarm_rate"] == 0.25
    mixed = pd.concat(
        [
            genuine.assign(tool="none"),
            pd.DataFrame(
                {"label": ["spoof"] * 2, "speaker": "c", "score": [-2.0, -3.0], "tool": "x"}
            ),
        ],
        ignore_index=True,
    )
    out = aasist.result(mixed)
    assert out["pooled"]["eer"] <= 0.25 and out["per_attack"][0]["attack"] == "x"
