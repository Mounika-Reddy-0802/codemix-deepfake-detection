"""Smoke tests for src/data/download.py (Week 1, owner L).

These do no network I/O: they check the module imports cheaply and that the
eval-only streaming sources are wired to the right repo ids and licences. Actual
streaming (with a HF token) is exercised manually via ``python -m src.data.download
--smoke <source> <subset>``.
"""

from src.data import download


def test_module_imports_without_datasets() -> None:
    """Importing the module must not require the heavy `datasets` dependency."""
    assert download.EVAL_ONLY == "eval-only"


def test_all_sources_are_eval_only() -> None:
    for name, src in download.ALL_SOURCES.items():
        assert src.role == "eval-only", f"{name} must be eval-only"


def test_repo_ids_match_dataset_matrix() -> None:
    assert download.INDICSYNTH.repo_id == "vdivyasharma/IndicSynth"
    assert download.INDICTTS_DEEPFAKE.repo_id == "SherryT997/IndicTTS-Deepfake-Challenge-Data"
    assert download.INDICVOICES.repo_id == "ai4bharat/indicvoices"


def test_indicsynth_licence_is_noncommercial() -> None:
    assert download.INDICSYNTH.licence == "CC-BY-NC-4.0"


def test_unknown_subset_raises() -> None:
    import pytest

    with pytest.raises(KeyError):
        download._stream_from(download.INDICSYNTH, "klingon", "train")


def test_hf_token_reads_environment(monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    assert download.hf_token() is None
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    assert download.hf_token() == "hf_dummy"


def test_describe_sources_lists_all_three() -> None:
    text = download.describe_sources()
    assert "IndicSynth" in text
    assert "IndicTTS-Deepfake" in text
    assert "indicvoices" in text
