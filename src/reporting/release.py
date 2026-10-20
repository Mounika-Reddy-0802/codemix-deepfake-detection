"""Build the metadata release: everything needed to audit every number, and no audio.

The generated clones are cloned speech of identifiable people, so the audio is never
published (``docs/datasheet.md``). What can be published, and is enough to check every
table in the paper, is collected here into one archive:

* the code-free metadata: manifests (clip lists with labels, speakers, attacks and
  conditions; paths are relative ``${DATA_ROOT}`` placeholders), generation job tables,
  speaker pools, and training configurations;
* every result: per-clip scores and the JSON summaries the paper's tables are read from;
* the result write-ups, datasheet and licence notes;
* ``CHECKSUMS.sha256`` for every file in the archive, and a short README.

Membership is an allowlist of folders and file types, limited to files git tracks (so
no half-written or local file can slip in), and a final check refuses any
audio, checkpoint, archive or secret that slips into those folders, so a release can
only ever be too small, never too large.
"""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

INCLUDE = {  # folder -> file suffixes taken from it (recursively)
    "data/manifests": {".csv"},
    "configs": {".yaml"},
    "experiments/results": {".json", ".csv", ".md", ".txt"},
    "results": {".json", ".csv"},
    "docs/results": {".md", ".json"},
}
FILES = ("docs/datasheet.md", "docs/licences.md", "README.md")
FORBIDDEN_SUFFIXES = {".wav", ".flac", ".mp3", ".amr", ".pt", ".pth", ".ckpt", ".zip", ".onnx"}
FORBIDDEN_NAMES = (".env", "kaggle.json", ".env.git")
#: Raw sources and anything holding audio paths into the local machine stay out.
EXCLUDE_PARTS = ("data/raw", "_EXCLUDED_children", "checkpoints")

README = """# Metadata release

Clip lists, generation job tables, configurations, per-clip scores and result
summaries behind every table of the paper. No audio and no model weights are
included: the generated clones are cloned speech of identifiable people.

Manifest paths are written as ${DATA_ROOT}/clips/<name>.wav; the genuine audio is
MUCS 2021 (CC BY-SA 4.0) and HiACC (CC BY 4.0), and the bundles are rebuilt from
them with the project's code. CHECKSUMS.sha256 lists every file in this archive.
"""


def collect(repo: str) -> list[Path]:
    """Every file the release takes from ``repo``, as paths relative to it."""
    root = Path(repo)
    picked: set[Path] = set()
    for folder, suffixes in INCLUDE.items():
        base = root / folder
        if base.is_dir():
            for path in base.rglob("*"):
                if path.is_file() and path.suffix.lower() in suffixes:
                    picked.add(path.relative_to(root))
    for name in FILES:
        if (root / name).is_file():
            picked.add(Path(name))
    tracked = _tracked(root)
    if tracked is not None:  # only what the repository records, never local leftovers
        picked = {p for p in picked if p.as_posix() in tracked}
    return sorted(p for p in picked if not any(part in p.as_posix() for part in EXCLUDE_PARTS))


def _tracked(root: Path) -> set[str] | None:
    """Files git tracks under ``root``, or None outside a git checkout."""
    import subprocess

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        if Path(git("rev-parse", "--show-toplevel")).resolve() != root.resolve():
            return None  # inside some other checkout: its file list says nothing about root
        return set(git("ls-files").splitlines())
    except (OSError, subprocess.CalledProcessError):
        return None


def check(paths: list[Path]) -> None:
    """Refuse audio, weights, archives and secrets, whatever folder they sit in."""
    bad = [p for p in paths if p.suffix.lower() in FORBIDDEN_SUFFIXES or p.name in FORBIDDEN_NAMES]
    if bad:
        raise ValueError(f"refusing to release {[p.as_posix() for p in bad[:5]]}")


def build(repo: str, out: str) -> dict:
    """Write the archive; returns its file count and total uncompressed size."""
    paths = collect(repo)
    check(paths)
    lines, total = [], 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in paths:
            data = (Path(repo) / rel).read_bytes()
            total += len(data)
            lines.append(f"{hashlib.sha256(data).hexdigest()}  {rel.as_posix()}")
            zf.writestr(rel.as_posix(), data)
        zf.writestr("RELEASE_README.md", README)
        zf.writestr("CHECKSUMS.sha256", "\n".join(lines) + "\n")
    return {"files": len(paths), "bytes": total}


def verify(archive: str) -> int:
    """Re-check every checksum inside a built archive; returns how many were checked."""
    with zipfile.ZipFile(archive) as zf:
        sums = zf.read("CHECKSUMS.sha256").decode().splitlines()
        for line in sums:
            digest, name = line.split("  ", 1)
            if hashlib.sha256(zf.read(name)).hexdigest() != digest:
                raise ValueError(f"checksum mismatch: {name}")
        return len(sums)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the metadata release archive")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--out", default="codemix_metadata_release.zip")
    args = parser.parse_args()
    stats = build(args.repo, args.out)
    checked = verify(args.out)
    print(
        f"{args.out}: {stats['files']} files, {stats['bytes'] / 1e6:.1f} MB uncompressed, "
        f"{checked} checksums verified"
    )


if __name__ == "__main__":
    main()
