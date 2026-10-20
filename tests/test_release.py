"""Smoke tests for src.reporting.release (the metadata release)."""

import pytest

from src.reporting import release


def _repo(tmp_path):
    (tmp_path / "data/manifests").mkdir(parents=True)
    (tmp_path / "data/manifests/a.csv").write_text("filepath\n${DATA_ROOT}/clips/x.wav\n")
    (tmp_path / "data/manifests/clip.wav").write_bytes(b"RIFF")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/train.yaml").write_text("lr: 1\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/datasheet.md").write_text("# data\n")
    return tmp_path


def test_collect_takes_only_allowlisted_types(tmp_path):
    paths = [p.as_posix() for p in release.collect(str(_repo(tmp_path)))]
    assert paths == ["configs/train.yaml", "data/manifests/a.csv", "docs/datasheet.md"]


def test_check_refuses_audio_and_secrets():
    with pytest.raises(ValueError):
        release.check([release.Path("results/x.wav")])
    with pytest.raises(ValueError):
        release.check([release.Path("configs/.env")])


def test_build_and_verify(tmp_path):
    repo = _repo(tmp_path)
    out = tmp_path / "release.zip"
    assert release.build(str(repo), str(out))["files"] == 3
    assert release.verify(str(out)) == 3
