"""CRITICAL-PATH test (Week 4, owner M): anti-leakage split validation.

Speaker leakage between train and eval is the single bug that could invalidate
the whole paper, so this file enforces the anti-leakage checklist from the
tech-stack doc (Section 4). Stubbed at bootstrap; real assertions land in Week 4
when ``src/data/build_manifests.py`` and the manifests exist.

Anti-leakage checklist enforced here:
  1. No speaker in both a training and any eval manifest (bonafide AND spoof;
     a cloned voice of speaker X counts as speaker X).
  2. No Tortoise-generated file in any training manifest, ever.
  3. No IndicSynth / IndicTTS / IndicVoices audio in any training manifest.
  4. No HiACC child audio anywhere in the pipeline.
  5. No spoof transcript paired with the same speaker's bonafide of the same
     sentence in eval (near-duplicate leakage).
"""

import pytest


@pytest.mark.skip(reason="TODO(week-4, M): implement once manifests exist")
def test_no_speaker_overlap_train_vs_eval() -> None:
    """Checklist 1: train and eval speaker sets are disjoint (incl. clones)."""


@pytest.mark.skip(reason="TODO(week-4, M): implement once manifests exist")
def test_no_tortoise_in_training_manifests() -> None:
    """Checklist 2: the held-out (Tortoise) tool never appears in training."""


@pytest.mark.skip(reason="TODO(week-4, M): implement once manifests exist")
def test_no_eval_only_corpora_in_training() -> None:
    """Checklist 3: IndicSynth / IndicTTS / IndicVoices are eval-only."""


@pytest.mark.skip(reason="TODO(week-4, M): implement once manifests exist")
def test_no_hiacc_child_audio_anywhere() -> None:
    """Checklist 4: no HiACC child utterance appears in any manifest."""


@pytest.mark.skip(reason="TODO(week-4, M): implement once manifests exist")
def test_no_transcript_near_duplicate_leakage() -> None:
    """Checklist 5: spoof transcript != same speaker's eval bonafide sentence."""


def test_build_manifests_module_is_importable() -> None:
    """Smoke: the module exists and imports (no heavy deps at import time)."""
    import importlib

    mod = importlib.import_module("src.data.build_manifests")
    assert mod.__doc__ is not None
