"""Verify a CM02 archive against what the repository says the run produced.

The run itself cannot live in git. The converted clips are cloned voices of
identifiable MUCS speakers in a public repo, and twelve ~55 MB weights belong in
``checkpoints/**`` which is ignored. So the audio and the models live in a private
archive off to one side, and what git holds is a *description* of them:
``outputs/rvc_model_checksums.json`` (SHA-256, size and epochs per model) and
``outputs/rvc_generation_metadata.jsonl`` (one record per converted clip).

That description was written and never read back. A checksum nobody verifies is
decoration -- it cannot tell a complete archive from one that lost a file in a zip,
a browser download that truncated, or a re-run that quietly produced *different*
weights. This module closes that loop, and it is the only way to answer "is
everything there?" about an archive this machine cannot see.

What it can and cannot prove:

- **Models: fully verified.** Name, byte size and SHA-256 are all in the manifest,
  so a weight file either is the one this run produced or it is not.
- **Clips: presence only.** The metadata log carries no per-clip hash -- hashing
  1,500 files at generation time would have cost minutes for a check nothing was
  going to run. So a missing or renamed clip is caught; a corrupted one is not.
  ``src.data.generation_qa`` is the screen that catches bad audio, and its report
  is already in ``docs/qa/``.

Layout-agnostic by design: an archive may be a mounted Kaggle dataset, an
extracted zip, or a nest of per-target folders, so files are matched by **name**
anywhere under the root rather than by an expected path. Stdlib plus the project's
own ``sha256_file``; no pandas, no audio stack.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from src.data.checksums import sha256_file
from src.data.rvc_report import METADATA_PATH, MODEL_MANIFEST_PATH

__all__ = [
    "ArchiveReport",
    "ModelCheck",
    "describe",
    "expected_clips",
    "expected_models",
    "index_files",
    "verify_archive",
    "verify_clips",
    "verify_models",
]


@dataclass(frozen=True)
class ModelCheck:
    """The verdict on one weight or index file."""

    name: str
    speaker: str
    kind: str  # "weight" | "index"
    status: str  # "ok" | "missing" | "size_mismatch" | "hash_mismatch"
    expected_bytes: int = 0
    actual_bytes: int = 0
    expected_sha256: str = ""
    actual_sha256: str = ""

    @property
    def ok(self) -> bool:
        """Whether this file matched the manifest exactly."""
        return self.status == "ok"

    def describe(self) -> str:
        """One line for the report."""
        if self.status == "ok":
            return f"  ok        {self.name}"
        if self.status == "missing":
            return f"  MISSING   {self.name}  (speaker {self.speaker}, {self.kind})"
        if self.status == "size_mismatch":
            return (
                f"  BAD SIZE  {self.name}  expected {self.expected_bytes} B, "
                f"found {self.actual_bytes} B"
            )
        return (
            f"  BAD HASH  {self.name}  expected {self.expected_sha256[:16]}..., "
            f"found {self.actual_sha256[:16]}..."
        )


@dataclass
class ArchiveReport:
    """Everything checked, and whether the archive is complete."""

    root: str
    models: list[ModelCheck] = field(default_factory=list)
    clips_expected: int = 0
    clips_found: int = 0
    clips_missing: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        """True when every model matched and every clip is present."""
        return all(check.ok for check in self.models) and not self.clips_missing

    def as_dict(self) -> dict:
        """JSON-serialisable summary, for pasting into a run log."""
        return {
            "root": self.root,
            "complete": self.complete,
            "models_checked": len(self.models),
            "models_ok": sum(1 for c in self.models if c.ok),
            "model_problems": {
                status: count
                for status, count in Counter(c.status for c in self.models if not c.ok).items()
            },
            "clips_expected": self.clips_expected,
            "clips_found": self.clips_found,
            "clips_missing": len(self.clips_missing),
        }


def expected_models(manifest_path: str = MODEL_MANIFEST_PATH) -> list[dict]:
    """The model records git says this run produced."""
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return list(payload.get("models", []))


def expected_clips(metadata_path: str = METADATA_PATH) -> list[str]:
    """Basenames of every converted clip the metadata log claims exists.

    Basenames, not paths: the log stores ``${DATA_ROOT}``-relative or Kaggle paths
    that say nothing about where an archive put the file.
    """
    path = Path(metadata_path)
    if not path.exists():
        return []
    names: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            output = json.loads(line).get("output_path", "")
            if output:
                names.append(Path(str(output).replace("\\", "/")).name)
    return names


def index_files(root: str) -> dict[str, Path]:
    """Every file under ``root`` by basename, first match wins on a collision.

    One walk, so a 1,500-file archive is scanned once rather than once per lookup.
    """
    found: dict[str, Path] = {}
    for path in Path(root).rglob("*"):
        if path.is_file():
            found.setdefault(path.name, path)
    return found


def _check_one(
    record: dict, kind: str, present: dict[str, Path], verify_hash: bool
) -> ModelCheck | None:
    """Verify one weight or index entry from the manifest."""
    name_key, bytes_key, hash_key = (
        ("weight_file", "bytes", "sha256")
        if kind == "weight"
        else ("index_file", "index_bytes", "index_sha256")
    )
    name = record.get(name_key)
    if not name:
        return None  # a model may legitimately have no faiss index

    speaker = str(record.get("speaker", ""))
    expected_bytes = int(record.get(bytes_key, 0) or 0)
    expected_hash = str(record.get(hash_key, "") or "")
    path = present.get(name)
    if path is None:
        return ModelCheck(name, speaker, kind, "missing", expected_bytes, 0, expected_hash)

    actual_bytes = path.stat().st_size
    if expected_bytes and actual_bytes != expected_bytes:
        # Size first: it is free, and a truncated download fails here without
        # spending a minute hashing a file that is already known to be wrong.
        return ModelCheck(
            name, speaker, kind, "size_mismatch", expected_bytes, actual_bytes, expected_hash
        )
    if not verify_hash or not expected_hash:
        return ModelCheck(name, speaker, kind, "ok", expected_bytes, actual_bytes, expected_hash)

    actual_hash = sha256_file(str(path))
    status = "ok" if actual_hash == expected_hash else "hash_mismatch"
    return ModelCheck(
        name, speaker, kind, status, expected_bytes, actual_bytes, expected_hash, actual_hash
    )


def verify_models(
    root: str,
    manifest_path: str = MODEL_MANIFEST_PATH,
    verify_hash: bool = True,
    present: dict[str, Path] | None = None,
) -> list[ModelCheck]:
    """Check every weight and index in the manifest against the archive.

    ``verify_hash=False`` checks name and size only -- useful for a quick pass over
    a slow mounted volume, but it cannot tell a re-run's different weights from
    this run's, which is the whole point of the manifest.
    """
    found = index_files(root) if present is None else present
    checks: list[ModelCheck] = []
    for record in expected_models(manifest_path):
        for kind in ("weight", "index"):
            check = _check_one(record, kind, found, verify_hash)
            if check is not None:
                checks.append(check)
    return checks


def verify_clips(
    root: str, metadata_path: str = METADATA_PATH, present: dict[str, Path] | None = None
) -> tuple[int, list[str]]:
    """Count converted clips present in the archive. Returns ``(found, missing)``."""
    found = index_files(root) if present is None else present
    expected = expected_clips(metadata_path)
    missing = [name for name in expected if name not in found]
    return len(expected) - len(missing), missing


def verify_archive(
    root: str,
    manifest_path: str = MODEL_MANIFEST_PATH,
    metadata_path: str = METADATA_PATH,
    verify_hash: bool = True,
) -> ArchiveReport:
    """Full check of an archive against the committed description of the run."""
    present = index_files(root)  # one walk, shared by both checks
    models = verify_models(root, manifest_path, verify_hash, present=present)
    found, missing = verify_clips(root, metadata_path, present=present)
    return ArchiveReport(
        root=str(root),
        models=models,
        clips_expected=len(expected_clips(metadata_path)),
        clips_found=found,
        clips_missing=missing,
    )


def describe(report: ArchiveReport, max_listed: int = 10) -> str:
    """A human-readable verdict, ready to paste into a run log or a PR comment."""
    lines = [f"CM02 archive check: {report.root}", ""]

    ok = sum(1 for c in report.models if c.ok)
    lines.append(f"Models and indexes: {ok}/{len(report.models)} verified")
    problems = [c for c in report.models if not c.ok]
    lines.extend(c.describe() for c in (problems or report.models)[:max_listed])
    if len(problems or report.models) > max_listed:
        lines.append(f"  ... and {len(problems or report.models) - max_listed} more")

    lines.append("")
    lines.append(f"Converted clips: {report.clips_found}/{report.clips_expected} present")
    for name in report.clips_missing[:max_listed]:
        lines.append(f"  MISSING   {name}")
    if len(report.clips_missing) > max_listed:
        lines.append(f"  ... and {len(report.clips_missing) - max_listed} more missing")

    lines.append("")
    if report.complete:
        lines.append("COMPLETE -- every model hashes correctly and every clip is present.")
    else:
        lines.append("INCOMPLETE -- see the entries above before relying on this archive.")
    return "\n".join(lines)


def main() -> None:
    """CLI: ``python -m src.data.rvc_archive --root /kaggle/input/<dataset>``."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Verify a CM02 archive against the manifests committed to git"
    )
    parser.add_argument("--root", required=True, help="archive root (mounted dataset or extract)")
    parser.add_argument("--manifest", default=MODEL_MANIFEST_PATH)
    parser.add_argument("--metadata", default=METADATA_PATH)
    parser.add_argument(
        "--no-hash", action="store_true", help="check name and size only (faster, weaker)"
    )
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = parser.parse_args()

    report = verify_archive(args.root, args.manifest, args.metadata, verify_hash=not args.no_hash)
    print(json.dumps(report.as_dict(), indent=2) if args.json else describe(report))
    sys.exit(0 if report.complete else 1)


if __name__ == "__main__":
    main()
