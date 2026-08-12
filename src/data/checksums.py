"""Corpus archive provenance: SHA-256 manifests for data that never enters git.

The corpora are tens of gigabytes and are gitignored, so the repository can only
carry a *description* of them: where each archive came from, how big it is, and
what it hashes to. That description is what makes the project reproducible --
without it, "we used MUCS 2021" is an unverifiable claim, and a teammate who
downloads from a Drive mirror has no way to tell a complete archive from a
truncated one.

Two things this guards against, both of which have bitten real projects:

- a **partial download** that extracts without error but is missing files, so a
  corpus silently shrinks and every count in the paper is wrong;
- a **re-encoded or repacked mirror** (someone unzips, adds a file, rezips) that
  is no longer the corpus the paper cites.

Hashing streams the file in chunks, so a 23 GB archive costs constant memory.
Stdlib only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

#: Where the manifest lives. Tracked in git; the archives themselves are not.
MANIFEST_PATH = "docs/data_checksums.json"

#: Read size for streaming hashes. 8 MB keeps a 23 GB file at a few seconds/GB.
CHUNK_BYTES = 8 * 1024 * 1024

#: Archives the project expects, relative to the data root, with their source.
KNOWN_ARCHIVES = {
    "raw/asvspoof2019_LA/LA.zip": "https://datashare.ed.ac.uk/handle/10283/3336",
    "raw/mucs2021/Hindi-English_train.tar.gz": "https://www.openslr.org/104/",
    "raw/mucs2021/Hindi-English_test.tar.gz": "https://www.openslr.org/104/",
}


class ChecksumError(AssertionError):
    """Raised when an archive does not match its recorded hash."""


@dataclass(frozen=True)
class ArchiveRecord:
    """One archive's identity: what it is, how big, and what it hashes to."""

    relpath: str
    bytes: int
    sha256: str
    source_url: str = ""
    mirror_url: str = ""

    @property
    def gigabytes(self) -> float:
        return round(self.bytes / (1024**3), 2)

    def describe(self) -> str:
        return f"{self.relpath}  {self.gigabytes} GB  {self.sha256[:16]}..."


def sha256_file(path: str, chunk_bytes: int = CHUNK_BYTES) -> str:
    """Streaming SHA-256 of a file. Constant memory regardless of file size."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def find_archives(data_root: str) -> list[str]:
    """Known archives that actually exist under ``data_root``, as relative paths."""
    root = Path(data_root)
    return sorted(rel for rel in KNOWN_ARCHIVES if (root / rel).is_file())


def hash_archives(data_root: str, mirrors: dict[str, str] | None = None) -> list[ArchiveRecord]:
    """Hash every known archive present under ``data_root``."""
    mirrors = mirrors or {}
    root = Path(data_root)
    records: list[ArchiveRecord] = []
    for rel in find_archives(data_root):
        path = root / rel
        print(f"hashing {rel} ({path.stat().st_size / 1024**3:.2f} GB) ...")
        records.append(
            ArchiveRecord(
                relpath=rel,
                bytes=path.stat().st_size,
                sha256=sha256_file(str(path)),
                source_url=KNOWN_ARCHIVES.get(rel, ""),
                mirror_url=mirrors.get(rel, ""),
            )
        )
    return records


# --------------------------------------------------------------------------- #
# Manifest I/O
# --------------------------------------------------------------------------- #
def read_manifest(path: str = MANIFEST_PATH) -> list[ArchiveRecord]:
    """Load recorded archives. A missing manifest is empty, not an error."""
    file = Path(path)
    if not file.is_file():
        return []
    raw = json.loads(file.read_text(encoding="utf-8"))
    return [ArchiveRecord(**entry) for entry in raw.get("archives", [])]


def write_manifest(records: list[ArchiveRecord], path: str = MANIFEST_PATH) -> None:
    """Write the manifest, merging with anything already recorded.

    Merging matters because no single machine holds every corpus: whoever has
    ASVspoof records it, whoever has MUCS records that, and neither erases the
    other's entry.
    """
    merged = {r.relpath: r for r in read_manifest(path)}
    for record in records:
        existing = merged.get(record.relpath)
        # Keep a mirror URL somebody already filled in by hand.
        if existing and existing.mirror_url and not record.mirror_url:
            record = ArchiveRecord(**{**asdict(record), "mirror_url": existing.mirror_url})
        merged[record.relpath] = record

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "note": (
            "SHA-256 of the ORIGINAL corpus archives. Data is gitignored; this file is "
            "how the project records what was actually used. Verify a copy with "
            "`python -m src.data.checksums --verify`."
        ),
        "archives": [asdict(r) for r in sorted(merged.values(), key=lambda r: r.relpath)],
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def verify(data_root: str, path: str = MANIFEST_PATH) -> list[str]:
    """Check archives on disk against the manifest. Returns a list of problems.

    Archives that are recorded but absent are **not** problems -- no one machine
    holds every corpus. A present archive whose hash or size disagrees is.
    """
    recorded = {r.relpath: r for r in read_manifest(path)}
    if not recorded:
        return ["no manifest recorded yet (run --write on a machine that has the archives)"]

    root = Path(data_root)
    problems: list[str] = []
    for rel, record in sorted(recorded.items()):
        file = root / rel
        if not file.is_file():
            continue  # not on this machine; fine
        size = file.stat().st_size
        if size != record.bytes:
            problems.append(
                f"{rel}: size {size} != recorded {record.bytes} "
                f"(incomplete download or a different build of the corpus)"
            )
            continue
        actual = sha256_file(str(file))
        if actual != record.sha256:
            problems.append(
                f"{rel}: sha256 {actual[:16]}... != recorded {record.sha256[:16]}... "
                f"(repacked or corrupted -- do not use this copy)"
            )
    return problems


def require_verified(data_root: str, path: str = MANIFEST_PATH) -> None:
    """Raise :class:`ChecksumError` if any present archive fails verification."""
    problems = verify(data_root, path)
    real = [p for p in problems if not p.startswith("no manifest")]
    if real:
        raise ChecksumError("archive verification failed:\n  " + "\n  ".join(real))


def as_markdown(records: list[ArchiveRecord]) -> str:
    """Render the manifest as a table for docs/data_sources.md."""
    lines = [
        "| Archive | Size | SHA-256 | Official source | Team mirror |",
        "|---|---|---|---|---|",
    ]
    for r in records:
        mirror = r.mirror_url or "_paste Drive link_"
        lines.append(
            f"| `{r.relpath}` | {r.gigabytes} GB | `{r.sha256[:32]}...` | {r.source_url} | {mirror} |"
        )
    return "\n".join(lines)


def main() -> None:
    """CLI: ``python -m src.data.checksums --write|--verify [--data-root DIR]``."""
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(description="Record or verify corpus archive hashes")
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "data"))
    parser.add_argument("--write", action="store_true", help="hash archives and record them")
    parser.add_argument("--verify", action="store_true", help="check archives against the manifest")
    parser.add_argument("--manifest", default=MANIFEST_PATH)
    parser.add_argument(
        "--mirror",
        action="append",
        default=[],
        metavar="RELPATH=URL",
        help="record a Drive/Kaggle mirror for an archive (repeatable)",
    )
    args = parser.parse_args()

    if args.write:
        mirrors = dict(m.split("=", 1) for m in args.mirror if "=" in m)
        records = hash_archives(args.data_root, mirrors)
        if not records:
            print(f"no known archives found under {args.data_root}")
            sys.exit(1)
        write_manifest(records, args.manifest)
        print(f"\nrecorded {len(records)} archive(s) in {args.manifest}:")
        for record in records:
            print("  " + record.describe())
        print("\nMarkdown table for docs/data_sources.md:\n")
        print(as_markdown(read_manifest(args.manifest)))
        return

    if args.verify:
        problems = verify(args.data_root, args.manifest)
        if not problems:
            print(f"all archives present under {args.data_root} match the manifest")
            return
        for problem in problems:
            print(f"  {problem}")
        sys.exit(1)

    parser.error("pass --write or --verify")


if __name__ == "__main__":
    main()
