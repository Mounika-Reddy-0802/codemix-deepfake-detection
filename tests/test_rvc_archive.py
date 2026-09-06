"""Tests for the CM02 archive check.

The archive holds the run itself -- 1,500 cloned clips and twelve voice models --
and it lives outside git because none of that can be committed. What git holds is
the description: SHA-256 per model, one metadata record per clip. This module is
what reads that description back, so the cases below are the ways an archive can
be wrong while still looking plausible:

- a file that never made it into the zip,
- a download that truncated (right name, wrong size),
- a **re-run's** weights standing in for this run's (right name, right size,
  different hash) -- the one failure only a checksum can catch, and the reason the
  manifest exists at all,
- clips lost between the session and the archive.

Stdlib and tmp_path only; no audio, no torch.
"""

import json

import pytest

from src.data import rvc_archive


def _manifest(tmp_path, weight_bytes=b"weight-payload", index_bytes=b"index-payload"):
    """A one-model manifest whose hashes match the payloads handed back."""
    from src.data.checksums import sha256_file

    staging = tmp_path / "staging"
    staging.mkdir()
    weight = staging / "rvc_trn0.pth"
    index = staging / "added_IVF10_Flat_nprobe_1_rvc_trn0_v2.index"
    weight.write_bytes(weight_bytes)
    index.write_bytes(index_bytes)

    payload = {
        "note": "test manifest",
        "models": [
            {
                "speaker": "trn0",
                "experiment": "rvc_trn0",
                "weight_file": weight.name,
                "bytes": weight.stat().st_size,
                "sha256": sha256_file(str(weight)),
                "index_file": index.name,
                "index_bytes": index.stat().st_size,
                "index_sha256": sha256_file(str(index)),
                "epochs": 100,
            }
        ],
    }
    path = tmp_path / "checksums.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, weight, index


def _metadata(tmp_path, names=("rvc_trn0_00001.wav", "rvc_trn0_00002.wav")):
    path = tmp_path / "metadata.jsonl"
    path.write_text(
        "\n".join(json.dumps({"output_path": f"/kaggle/working/rvc_outputs/{n}"}) for n in names),
        encoding="utf-8",
    )
    return path


def _archive(tmp_path, weight=None, index=None, clips=()):
    """Build an archive directory, deliberately nested to exercise name matching."""
    root = tmp_path / "archive"
    (root / "models").mkdir(parents=True)
    (root / "audio" / "rvc_outputs").mkdir(parents=True)
    if weight is not None:
        (root / "models" / weight.name).write_bytes(weight.read_bytes())
    if index is not None:
        (root / "models" / index.name).write_bytes(index.read_bytes())
    for name in clips:
        (root / "audio" / "rvc_outputs" / name).write_bytes(b"RIFF")
    return root


# --------------------------------------------------------------------------- #
# The happy path
# --------------------------------------------------------------------------- #
def test_a_complete_archive_verifies(tmp_path) -> None:
    manifest, weight, index = _manifest(tmp_path)
    metadata = _metadata(tmp_path)
    root = _archive(tmp_path, weight, index, ("rvc_trn0_00001.wav", "rvc_trn0_00002.wav"))

    report = rvc_archive.verify_archive(str(root), str(manifest), str(metadata))
    assert report.complete
    assert report.clips_found == 2 and not report.clips_missing
    assert [c.status for c in report.models] == ["ok", "ok"]
    assert "COMPLETE" in rvc_archive.describe(report)


def test_files_are_matched_by_name_at_any_depth(tmp_path) -> None:
    """An archive may be a mounted dataset, an extract, or per-target folders."""
    manifest, weight, index = _manifest(tmp_path)
    root = tmp_path / "deep"
    (root / "a" / "b" / "c").mkdir(parents=True)
    (root / "a" / "b" / "c" / weight.name).write_bytes(weight.read_bytes())
    (root / "a" / index.name).write_bytes(index.read_bytes())

    checks = rvc_archive.verify_models(str(root), str(manifest))
    assert all(c.ok for c in checks)


# --------------------------------------------------------------------------- #
# The four ways an archive is wrong while looking plausible
# --------------------------------------------------------------------------- #
def test_a_missing_weight_is_reported(tmp_path) -> None:
    manifest, weight, index = _manifest(tmp_path)
    root = _archive(tmp_path, weight=None, index=index)

    checks = rvc_archive.verify_models(str(root), str(manifest))
    weight_check = next(c for c in checks if c.kind == "weight")
    assert weight_check.status == "missing"
    assert "MISSING" in weight_check.describe()


def test_a_truncated_download_is_caught_on_size(tmp_path) -> None:
    manifest, weight, index = _manifest(tmp_path)
    root = _archive(tmp_path, weight, index)
    (root / "models" / weight.name).write_bytes(b"trunc")  # right name, wrong size

    check = next(
        c for c in rvc_archive.verify_models(str(root), str(manifest)) if c.kind == "weight"
    )
    assert check.status == "size_mismatch"
    assert check.actual_bytes == 5


def test_a_rerun_producing_different_weights_is_caught_on_hash(tmp_path) -> None:
    """Same name, same size, different content -- only the checksum sees this."""
    manifest, weight, index = _manifest(tmp_path)
    root = _archive(tmp_path, weight, index)
    swapped = b"X" * len(weight.read_bytes())
    (root / "models" / weight.name).write_bytes(swapped)

    check = next(
        c for c in rvc_archive.verify_models(str(root), str(manifest)) if c.kind == "weight"
    )
    assert check.status == "hash_mismatch"
    assert check.actual_bytes == check.expected_bytes  # size alone would have passed


def test_missing_clips_are_listed(tmp_path) -> None:
    manifest, weight, index = _manifest(tmp_path)
    metadata = _metadata(tmp_path)
    root = _archive(tmp_path, weight, index, ("rvc_trn0_00001.wav",))

    report = rvc_archive.verify_archive(str(root), str(manifest), str(metadata))
    assert not report.complete
    assert report.clips_found == 1
    assert report.clips_missing == ["rvc_trn0_00002.wav"]
    assert "INCOMPLETE" in rvc_archive.describe(report)


# --------------------------------------------------------------------------- #
# Behaviour around the edges
# --------------------------------------------------------------------------- #
def test_no_hash_mode_accepts_a_right_sized_impostor(tmp_path) -> None:
    """Documents what --no-hash gives up: speed at the cost of the real check."""
    manifest, weight, index = _manifest(tmp_path)
    root = _archive(tmp_path, weight, index)
    (root / "models" / weight.name).write_bytes(b"X" * len(weight.read_bytes()))

    checks = rvc_archive.verify_models(str(root), str(manifest), verify_hash=False)
    assert all(c.ok for c in checks)


def test_a_model_without_an_index_is_not_an_error(tmp_path) -> None:
    """faiss index training is allowed to fail; conversion runs at index_rate 0."""
    manifest, weight, _ = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    for key in ("index_file", "index_bytes", "index_sha256"):
        payload["models"][0].pop(key)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    root = _archive(tmp_path, weight)

    checks = rvc_archive.verify_models(str(root), str(manifest))
    assert [c.kind for c in checks] == ["weight"]
    assert checks[0].ok


def test_metadata_paths_are_reduced_to_basenames(tmp_path) -> None:
    """The log carries Kaggle or ${DATA_ROOT} paths; an archive knows neither."""
    path = tmp_path / "meta.jsonl"
    path.write_text(
        json.dumps({"output_path": "${DATA_ROOT}/generated/rvc/a.wav"})
        + "\n"
        + json.dumps({"output_path": "C:\\dfdata\\generated\\rvc\\b.wav"})
        + "\n",
        encoding="utf-8",
    )
    assert rvc_archive.expected_clips(str(path)) == ["a.wav", "b.wav"]


def test_summary_is_json_serialisable(tmp_path) -> None:
    manifest, weight, index = _manifest(tmp_path)
    metadata = _metadata(tmp_path)
    root = _archive(tmp_path, weight, index, ("rvc_trn0_00001.wav",))

    summary = rvc_archive.verify_archive(str(root), str(manifest), str(metadata)).as_dict()
    json.dumps(summary)  # must not raise
    assert summary["complete"] is False
    assert summary["clips_missing"] == 1
    assert summary["models_ok"] == 2


def test_a_missing_metadata_log_yields_no_expected_clips(tmp_path) -> None:
    assert rvc_archive.expected_clips(str(tmp_path / "absent.jsonl")) == []


def test_cli_exits_nonzero_on_an_incomplete_archive(tmp_path, monkeypatch, capsys) -> None:
    manifest, weight, index = _manifest(tmp_path)
    metadata = _metadata(tmp_path)
    root = _archive(tmp_path, weight, index, ())
    monkeypatch.setattr(
        "sys.argv",
        [
            "rvc_archive",
            "--root",
            str(root),
            "--manifest",
            str(manifest),
            "--metadata",
            str(metadata),
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        rvc_archive.main()
    assert excinfo.value.code == 1
    assert "INCOMPLETE" in capsys.readouterr().out
