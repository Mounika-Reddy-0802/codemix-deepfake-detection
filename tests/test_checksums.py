"""Tests for corpus archive provenance (Week 3, W3-T2, owner L).

The failure this guards against is quiet: a truncated or repacked archive
extracts without complaint, the corpus silently changes size, and every count in
the paper is wrong. So the tests below care most about *detecting a difference*,
and about not crying wolf when a machine simply doesn't hold every corpus.

Small synthetic files, no real archives needed.
"""

import json
from pathlib import Path

import pytest

from src.data import checksums as ck


def _archive(root: Path, rel: str, content: bytes = b"corpus bytes") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


MUCS = "raw/mucs2021/Hindi-English_train.tar.gz"
MUCS_TEST = "raw/mucs2021/Hindi-English_test.tar.gz"


# --------------------------------------------------------------------------- #
# Hashing
# --------------------------------------------------------------------------- #
def test_hash_matches_hashlib(tmp_path: Path) -> None:
    import hashlib

    path = _archive(tmp_path, MUCS, b"abc" * 1000)
    assert ck.sha256_file(str(path)) == hashlib.sha256(b"abc" * 1000).hexdigest()


def test_hash_is_stable_across_chunk_sizes(tmp_path: Path) -> None:
    # A 23 GB archive is read in chunks; chunking must not change the digest.
    path = _archive(tmp_path, MUCS, b"x" * 100_000)
    assert ck.sha256_file(str(path), chunk_bytes=64) == ck.sha256_file(str(path), chunk_bytes=8192)


def test_empty_file_hashes(tmp_path: Path) -> None:
    path = _archive(tmp_path, MUCS, b"")
    assert len(ck.sha256_file(str(path))) == 64


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def test_only_known_archives_are_found(tmp_path: Path) -> None:
    _archive(tmp_path, MUCS)
    _archive(tmp_path, "raw/mucs2021/random_notes.txt")
    assert ck.find_archives(str(tmp_path)) == [MUCS]


def test_missing_archives_are_not_reported(tmp_path: Path) -> None:
    assert ck.find_archives(str(tmp_path)) == []


# --------------------------------------------------------------------------- #
# Manifest round-trip
# --------------------------------------------------------------------------- #
def test_write_then_read_round_trips(tmp_path: Path) -> None:
    _archive(tmp_path, MUCS, b"train data")
    manifest = str(tmp_path / "checksums.json")
    ck.write_manifest(ck.hash_archives(str(tmp_path)), manifest)

    records = ck.read_manifest(manifest)
    assert len(records) == 1
    assert records[0].relpath == MUCS
    assert records[0].bytes == len(b"train data")


def test_missing_manifest_reads_as_empty(tmp_path: Path) -> None:
    assert ck.read_manifest(str(tmp_path / "nope.json")) == []


def test_manifest_merges_instead_of_overwriting(tmp_path: Path) -> None:
    # No single machine holds every corpus, so recording one must not erase
    # another member's entry for a different one.
    manifest = str(tmp_path / "checksums.json")
    _archive(tmp_path, MUCS, b"train")
    ck.write_manifest(ck.hash_archives(str(tmp_path)), manifest)

    (tmp_path / "raw/mucs2021/Hindi-English_train.tar.gz").unlink()
    _archive(tmp_path, MUCS_TEST, b"test")
    ck.write_manifest(ck.hash_archives(str(tmp_path)), manifest)

    assert {r.relpath for r in ck.read_manifest(manifest)} == {MUCS, MUCS_TEST}


def test_hand_written_mirror_url_survives_a_rehash(tmp_path: Path) -> None:
    manifest = str(tmp_path / "checksums.json")
    _archive(tmp_path, MUCS, b"train")
    ck.write_manifest(ck.hash_archives(str(tmp_path), {MUCS: "https://drive/x"}), manifest)
    ck.write_manifest(ck.hash_archives(str(tmp_path)), manifest)  # rehash, no mirror given
    assert ck.read_manifest(manifest)[0].mirror_url == "https://drive/x"


def test_manifest_is_readable_json(tmp_path: Path) -> None:
    manifest = str(tmp_path / "checksums.json")
    _archive(tmp_path, MUCS, b"train")
    ck.write_manifest(ck.hash_archives(str(tmp_path)), manifest)
    payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    assert "archives" in payload and "note" in payload


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def test_unchanged_archive_verifies_clean(tmp_path: Path) -> None:
    manifest = str(tmp_path / "checksums.json")
    _archive(tmp_path, MUCS, b"train data")
    ck.write_manifest(ck.hash_archives(str(tmp_path)), manifest)
    assert ck.verify(str(tmp_path), manifest) == []


def test_truncated_archive_is_caught(tmp_path: Path) -> None:
    # The realistic failure: an interrupted download that still extracts.
    manifest = str(tmp_path / "checksums.json")
    path = _archive(tmp_path, MUCS, b"complete archive bytes")
    ck.write_manifest(ck.hash_archives(str(tmp_path)), manifest)
    path.write_bytes(b"partial")
    problems = ck.verify(str(tmp_path), manifest)
    assert len(problems) == 1
    assert "size" in problems[0]


def test_repacked_archive_of_the_same_size_is_caught(tmp_path: Path) -> None:
    # Same byte count, different content -- only the hash catches this.
    manifest = str(tmp_path / "checksums.json")
    path = _archive(tmp_path, MUCS, b"AAAAAAAAAA")
    ck.write_manifest(ck.hash_archives(str(tmp_path)), manifest)
    path.write_bytes(b"BBBBBBBBBB")
    problems = ck.verify(str(tmp_path), manifest)
    assert len(problems) == 1
    assert "sha256" in problems[0]


def test_absent_archive_is_not_a_problem(tmp_path: Path) -> None:
    # A teammate without ASVspoof must not see a failure for not having it.
    manifest = str(tmp_path / "checksums.json")
    _archive(tmp_path, MUCS, b"train")
    ck.write_manifest(ck.hash_archives(str(tmp_path)), manifest)
    (tmp_path / MUCS).unlink()
    assert ck.verify(str(tmp_path), manifest) == []


def test_verify_without_a_manifest_says_so(tmp_path: Path) -> None:
    problems = ck.verify(str(tmp_path), str(tmp_path / "nope.json"))
    assert problems and "no manifest" in problems[0]


def test_require_verified_raises_on_a_bad_archive(tmp_path: Path) -> None:
    manifest = str(tmp_path / "checksums.json")
    path = _archive(tmp_path, MUCS, b"good bytes")
    ck.write_manifest(ck.hash_archives(str(tmp_path)), manifest)
    path.write_bytes(b"bad bytes!")
    with pytest.raises(ck.ChecksumError):
        ck.require_verified(str(tmp_path), manifest)


def test_require_verified_ignores_a_missing_manifest(tmp_path: Path) -> None:
    ck.require_verified(str(tmp_path), str(tmp_path / "nope.json"))  # must not raise


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def test_markdown_table_flags_a_missing_mirror(tmp_path: Path) -> None:
    _archive(tmp_path, MUCS, b"train")
    table = ck.as_markdown(ck.hash_archives(str(tmp_path)))
    assert "paste Drive link" in table
    assert "openslr.org" in table
