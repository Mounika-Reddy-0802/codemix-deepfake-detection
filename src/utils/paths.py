"""Portable data paths: one manifest that works on the laptop and on Colab.

A manifest row carries the path to its clip. Written literally, that path is
machine-local — ``C:/dfdata/raw/mucs2021/...`` on the dev laptop,
``/content/data/raw/mucs2021/...`` on a Colab runtime — so the same manifest
cannot be used on both machines without a hand edit. Hand edits are exactly how a
frozen split stops being frozen, so the fix lives here instead.

``DATA_ROOT`` is the single environment variable each machine sets. Two mechanisms
build on it, both honoured by :func:`resolve`:

1. **Write portable.** :func:`portable` replaces the data root with the literal
   token ``${DATA_ROOT}``, and :func:`resolve` substitutes the local root back in.
   New manifests should be written this way.
2. **Read foreign.** Manifests already written with another machine's absolute
   paths (``data/manifests/clip_index.csv`` is full of ``C:/dfdata/...``) are
   re-rooted: :func:`resolve` finds the first known layout anchor — ``raw``,
   ``processed``, ``generated``, ``manifests`` — and rebases the remainder under
   the local ``DATA_ROOT``.

Pure stdlib, no I/O: resolution is structural, not a filesystem probe, so it costs
nothing to call once per row of a 50,000-row manifest.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

#: Environment variable naming the local data root.
DATA_ROOT_ENV = "DATA_ROOT"
#: Default when ``$DATA_ROOT`` is unset — the in-repo ``data/`` skeleton.
DEFAULT_DATA_ROOT = "data"
#: Literal token stored in portable manifests in place of the data root.
DATA_ROOT_TOKEN = "${DATA_ROOT}"

#: Directory names that mark the start of the project-relative part of a path.
#: These are the top-level entries of the ``data/`` tree (see the folder-structure
#: doc), so everything after one of them is identical on every machine.
LAYOUT_ANCHORS = ("raw", "processed", "generated", "manifests", "interim")

_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:/")


def _norm(path: str | os.PathLike[str]) -> str:
    """Backslashes to forward slashes, so one code path handles both platforms."""
    return str(path).replace("\\", "/")


def _is_absolute(text: str) -> bool:
    """True for POSIX absolute paths and for Windows drive-letter paths."""
    return text.startswith("/") or bool(_WINDOWS_DRIVE.match(text))


def data_root(root: str | os.PathLike[str] | None = None) -> Path:
    """The local data root: the argument, else ``$DATA_ROOT``, else ``data/``."""
    return Path(_norm(root or os.environ.get(DATA_ROOT_ENV) or DEFAULT_DATA_ROOT))


def _anchor_relative(text: str) -> str | None:
    """The part of ``text`` from its first layout anchor onwards, if any."""
    parts = [p for p in text.split("/") if p]
    for i, part in enumerate(parts):
        if part in LAYOUT_ANCHORS:
            return "/".join(parts[i:])
    return None


def portable(path: str | os.PathLike[str], root: str | os.PathLike[str] | None = None) -> str:
    """Rewrite a local path as a ``${DATA_ROOT}``-relative one.

    Paths that sit outside the data tree are returned unchanged — a manifest may
    legitimately point at a repo file, and mangling it would be worse than leaving
    it machine-specific.
    """
    text = _norm(path)
    root_text = _norm(data_root(root)).rstrip("/")
    if root_text and text.startswith(root_text + "/"):
        return f"{DATA_ROOT_TOKEN}/{text[len(root_text) + 1 :]}"
    relative = _anchor_relative(text)
    return f"{DATA_ROOT_TOKEN}/{relative}" if relative else text


def resolve(path: str | os.PathLike[str], root: str | os.PathLike[str] | None = None) -> str:
    """Turn a manifest path into one valid on *this* machine.

    Handles, in order: the ``${DATA_ROOT}`` token, paths already under the local
    root, repo-relative paths (left alone), and foreign absolute paths (rebased on
    their first layout anchor). A path with no anchor to rebase on is returned
    unchanged, so the caller sees the original name in the resulting error.
    """
    text = _norm(path)
    root_path = data_root(root)

    if DATA_ROOT_TOKEN in text:
        tail = text.split(DATA_ROOT_TOKEN, 1)[1].lstrip("/")
        return str(root_path / tail)

    if not _is_absolute(text):
        return str(Path(text))

    root_text = _norm(root_path).rstrip("/")
    if root_text and (text == root_text or text.startswith(root_text + "/")):
        return str(Path(text))

    relative = _anchor_relative(text)
    return str(root_path / relative) if relative else str(Path(text))


def resolve_column(frame, column: str = "filepath", root: str | os.PathLike[str] | None = None):
    """Return ``frame`` with one path column resolved for this machine.

    The frame is copied, so a manifest loaded once can be resolved for different
    roots without the first resolution leaking into the second.
    """
    if column not in frame.columns:
        return frame
    out = frame.copy()
    out[column] = out[column].astype(str).map(lambda p: resolve(p, root))
    return out


def main() -> None:
    """CLI: ``python -m src.utils.paths [PATH ...]`` shows how paths resolve here."""
    import argparse

    parser = argparse.ArgumentParser(description="Show how manifest paths resolve on this machine")
    parser.add_argument("paths", nargs="*", help="paths to resolve")
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()

    root = data_root(args.data_root)
    print(f"{DATA_ROOT_ENV}={root}  (exists: {root.is_dir()})")
    for raw in args.paths:
        print(f"  {raw}\n    -> {resolve(raw, root)}\n    portable: {portable(raw, root)}")


if __name__ == "__main__":
    main()
