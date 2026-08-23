"""The ethics gate: no voice cloning happens until the mentor note is signed.

Week 3 (W3-T5, owner SK). Voice cloning is the ethically sensitive step of this
project, so the rule is absolute and predates any code: **no spoof generation of
any kind, including pilots, until a signed mentor note exists in
``docs/ethics/``** (project plan W3-T1, the team git rules, section 10).

Until now that rule lived only in documentation, which means it was enforced by
whoever happened to remember it at 2am on a Kaggle session. This module makes it
mechanical: every generation entry point calls :func:`require_signoff` before a
model is loaded, and raises :class:`EthicsGateError` if the artefact is absent.

Deliberate design choices:

- **No override flag.** An ``--i-know-what-im-doing`` switch would make the gate
  decorative. The only way past it is to actually have the signed note.
- **``README.md`` does not count.** ``docs/ethics/README.md`` states what the
  sign-off must cover; it is not the sign-off. Only a scanned artefact matching
  :data:`SIGNOFF_PATTERNS` satisfies the gate.
- **Stdlib only**, so the gate can never fail to import in a stripped-down
  generation environment and get skipped for the wrong reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: Where the signed artefact is scanned to.
SIGNOFF_DIR = "docs/ethics"

#: Accepted filenames for the signed note. A scan or photo is fine; the point is
#: that a human signed something, not that it is a particular file format.
SIGNOFF_PATTERNS = (
    "mentor_signoff*.pdf",
    "mentor_signoff*.png",
    "mentor_signoff*.jpg",
    "mentor_signoff*.jpeg",
)

#: Files in the ethics folder that explicitly do NOT satisfy the gate.
NOT_A_SIGNOFF = ("README.md",)


class EthicsGateError(RuntimeError):
    """Raised when generation is attempted without a signed mentor note."""


@dataclass(frozen=True)
class SignoffStatus:
    """Whether the gate is open, and what evidence was found."""

    signed: bool
    artefacts: tuple[str, ...] = ()
    directory: str = SIGNOFF_DIR

    def describe(self) -> str:
        """One-line summary for logs and CLI output."""
        if self.signed:
            return f"ethics gate OPEN: signed note found ({', '.join(self.artefacts)})"
        return f"ethics gate CLOSED: no signed note in {self.directory}/"


def find_signoff(directory: str = SIGNOFF_DIR) -> list[str]:
    """Every artefact in ``directory`` that counts as a signed mentor note."""
    root = Path(directory)
    if not root.is_dir():
        return []
    found: list[str] = []
    for pattern in SIGNOFF_PATTERNS:
        for path in sorted(root.glob(pattern)):
            if path.name in NOT_A_SIGNOFF or not path.is_file():
                continue
            if path.stat().st_size == 0:  # an empty placeholder is not a signature
                continue
            found.append(path.name)
    return sorted(set(found))


def signoff_status(directory: str = SIGNOFF_DIR) -> SignoffStatus:
    """Check the gate without raising -- for status displays and dry runs."""
    artefacts = find_signoff(directory)
    return SignoffStatus(signed=bool(artefacts), artefacts=tuple(artefacts), directory=directory)


def is_signed(directory: str = SIGNOFF_DIR) -> bool:
    """Whether generation is permitted."""
    return signoff_status(directory).signed


def require_signoff(directory: str = SIGNOFF_DIR, action: str = "spoof generation") -> None:
    """Raise :class:`EthicsGateError` unless the signed note exists.

    Called at the top of every generation entry point, before any model is
    loaded, so a blocked run costs nothing and fails with a readable reason.
    """
    status = signoff_status(directory)
    if status.signed:
        return
    raise EthicsGateError(
        f"{action} is blocked: no signed mentor ethics note in {directory}/.\n"
        f"Expected one of {list(SIGNOFF_PATTERNS)} (docs/ethics/README.md states what "
        f"the sign-off must cover; it is not itself the sign-off).\n"
        f"This gate has no override. Obtain the signature, scan it into {directory}/, "
        f"and re-run."
    )


def main() -> None:
    """CLI: ``python -m src.data.ethics_gate`` -- prints the gate status."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Report the ethics gate status")
    parser.add_argument("--dir", default=SIGNOFF_DIR)
    args = parser.parse_args()

    status = signoff_status(args.dir)
    print(status.describe())
    if not status.signed:
        print("blocked: XTTS-v2 pilot, RVC pilot, and all Week-4 generation")
        sys.exit(1)


if __name__ == "__main__":
    main()
