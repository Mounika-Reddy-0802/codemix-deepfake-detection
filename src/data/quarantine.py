"""Audit the HiACC child-audio quarantine against the corpus as it actually is.

Week 3 (W3-T2, owner L). ``scripts/01_download_data.sh`` auto-moves folders whose
name matches ``child|kid|minor`` into ``data/raw/hiacc/_EXCLUDED_children/``. That
regex is a **guess about the layout**, and the plan is explicit that it must be
checked by hand before it is trusted: if HiACC separates children by a manifest
column or a filename field rather than by folder, the regex quarantines nothing
and the pipeline silently ingests child speech.

This module produces the evidence for that check. It deliberately **never returns
a pass**: :func:`audit` reports what it found, and a human signs the rendered
report in ``docs/qa/child_quarantine_check.md``. A tool cannot certify an ethics
gate it can only pattern-match.

Pure stdlib + pandas, no audio decoding, so it runs anywhere and in CI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

#: Name of the quarantine directory created at download time.
QUARANTINE_DIR = "_EXCLUDED_children"

#: Patterns that suggest a child speaker. Bounded by letter lookarounds rather
#: than ``\b``: corpus names separate tokens with underscores (``child_07``), and
#: ``_`` is a word character, so ``\b`` would never fire there. The lookarounds
#: still keep obvious false positives out (``kidney``, ``childish``).
CHILD_PATTERNS = (
    r"(?<![a-z])child(?:ren)?(?![a-z])",
    r"(?<![a-z])kids?(?![a-z])",
    r"(?<![a-z])minors?(?![a-z])",
    r"(?<![a-z])juveniles?(?![a-z])",
)

#: Manifest columns that would indicate the split is by metadata, not by folder.
AGE_COLUMN_HINTS = ("age", "age_group", "agegroup", "speaker_type", "category", "group")

AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".m4a")
MANIFEST_SUFFIXES = (".csv", ".tsv", ".txt", ".json")


@dataclass
class Finding:
    """One thing the audit noticed, and whether it is already quarantined."""

    path: str
    kind: str  # "dir" | "audio" | "manifest"
    detail: str
    quarantined: bool

    def as_row(self) -> str:
        state = "quarantined" if self.quarantined else "**OUTSIDE QUARANTINE**"
        return f"| `{self.path}` | {self.kind} | {self.detail} | {state} |"


@dataclass
class AuditResult:
    """Everything the audit saw. Never a verdict -- a human signs the verdict."""

    root: str
    exists: bool = False
    quarantine_exists: bool = False
    total_audio: int = 0
    quarantined_audio: int = 0
    findings: list[Finding] = field(default_factory=list)
    manifest_columns: dict[str, list[str]] = field(default_factory=dict)

    @property
    def unquarantined_findings(self) -> list[Finding]:
        """Child-looking things still inside the live corpus -- the danger list."""
        return [f for f in self.findings if not f.quarantined]

    @property
    def needs_manual_split(self) -> bool:
        """True when nothing was quarantined but age-like metadata columns exist.

        This is the failure mode the plan warns about: the folder regex found
        nothing, so the quarantine looks clean while the real child/adult split
        lives in a manifest column that nobody has acted on.
        """
        return self.quarantined_audio == 0 and bool(self.manifest_columns)


def _matches_child(text: str) -> str | None:
    """The child pattern ``text`` matches, if any."""
    lowered = text.lower()
    for pattern in CHILD_PATTERNS:
        if re.search(pattern, lowered):
            return pattern
    return None


def _is_quarantined(path: Path, root: Path) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    return QUARANTINE_DIR in parts


def scan_tree(root: str) -> list[Finding]:
    """Directories and audio files whose names look like child speakers."""
    root_path = Path(root)
    findings: list[Finding] = []
    if not root_path.exists():
        return findings

    for path in sorted(root_path.rglob("*")):
        if path.name == QUARANTINE_DIR:
            continue
        pattern = _matches_child(path.name)
        if pattern is None:
            continue
        if path.is_dir():
            kind = "dir"
        elif path.suffix.lower() in AUDIO_SUFFIXES:
            kind = "audio"
        else:
            continue
        findings.append(
            Finding(
                path=str(path.relative_to(root_path)),
                kind=kind,
                detail=f"name matches /{pattern}/",
                quarantined=_is_quarantined(path, root_path),
            )
        )
    return findings


def scan_manifests(root: str) -> tuple[dict[str, list[str]], list[Finding]]:
    """Age-like columns in corpus manifests, and rows whose values look child-like.

    Catches the case where HiACC splits adults from children by metadata rather
    than by directory -- in which case the folder regex quarantines nothing and
    the split has to be done by hand.
    """
    import pandas as pd

    root_path = Path(root)
    columns: dict[str, list[str]] = {}
    findings: list[Finding] = []
    if not root_path.exists():
        return columns, findings

    for path in sorted(root_path.rglob("*")):
        if path.suffix.lower() not in MANIFEST_SUFFIXES or not path.is_file():
            continue
        if _is_quarantined(path, root_path):
            continue
        try:
            sep = "\t" if path.suffix.lower() == ".tsv" else None
            frame = pd.read_csv(path, sep=sep, engine="python", nrows=2000)
        except Exception:  # noqa: BLE001 - unreadable manifests are reported, not fatal
            continue

        hits = [c for c in frame.columns if str(c).strip().lower() in AGE_COLUMN_HINTS]
        if hits:
            columns[str(path.relative_to(root_path))] = hits
        for column in hits:
            values = frame[column].astype(str).str.lower().unique()
            child_values = sorted({v for v in values if _matches_child(v)})
            if child_values:
                findings.append(
                    Finding(
                        path=str(path.relative_to(root_path)),
                        kind="manifest",
                        detail=f"column `{column}` has values {child_values[:5]}",
                        quarantined=False,
                    )
                )
    return columns, findings


def count_audio(root: str) -> tuple[int, int]:
    """``(total_audio, quarantined_audio)`` under ``root``."""
    root_path = Path(root)
    if not root_path.exists():
        return 0, 0
    total = quarantined = 0
    for path in root_path.rglob("*"):
        if path.suffix.lower() not in AUDIO_SUFFIXES or not path.is_file():
            continue
        total += 1
        if _is_quarantined(path, root_path):
            quarantined += 1
    return total, quarantined


def audit(root: str = "data/raw/hiacc") -> AuditResult:
    """Run the full audit over an extracted HiACC tree."""
    root_path = Path(root)
    result = AuditResult(root=str(root), exists=root_path.exists())
    if not result.exists:
        return result

    result.quarantine_exists = (root_path / QUARANTINE_DIR).is_dir()
    result.total_audio, result.quarantined_audio = count_audio(root)
    tree_findings = scan_tree(root)
    result.manifest_columns, manifest_findings = scan_manifests(root)
    result.findings = tree_findings + manifest_findings
    return result


def render_report(result: AuditResult) -> str:
    """Render the audit as the markdown report a human signs."""
    lines = [
        "# HiACC child-quarantine check (W3-T2)",
        "",
        "Ethics gate evidence. Generated by `python -m src.data.quarantine`.",
        "**This report is not a pass.** It lists what an automated scan could see;",
        "a team member confirms the adult/child split against the HiACC",
        "documentation and signs at the bottom.",
        "",
        f"- Corpus root: `{result.root}`",
        f"- Root exists: **{result.exists}**",
        f"- `{QUARANTINE_DIR}/` exists: **{result.quarantine_exists}**",
        f"- Audio files found: **{result.total_audio}**",
        f"- Of those, quarantined: **{result.quarantined_audio}**",
        "",
    ]

    if not result.exists:
        lines += [
            "> The corpus is not on disk yet. Run the downloads",
            "> (`bash scripts/01_download_data.sh --run --only hiacc`), then re-run",
            "> this audit. Nothing below can be verified until then.",
            "",
        ]

    lines += ["## Findings", ""]
    if result.findings:
        lines += [
            "| Path | Kind | Detail | State |",
            "|---|---|---|---|",
            *[f.as_row() for f in result.findings],
            "",
        ]
    else:
        lines += ["No name matched the child patterns anywhere under the root.", ""]

    danger = result.unquarantined_findings
    if danger:
        lines += [
            f"### {len(danger)} item(s) OUTSIDE quarantine",
            "",
            "These are child-looking and still live in the corpus. Move them into",
            f"`{QUARANTINE_DIR}/` before any preprocessing or generation runs.",
            "",
        ]

    if result.needs_manual_split:
        lines += [
            "### Folder regex quarantined nothing, but age-like columns exist",
            "",
            "This is the documented failure mode: the split is probably by",
            "manifest column, not by directory, so the automatic quarantine is a",
            "no-op. Do the split by hand before trusting anything.",
            "",
        ]

    if result.manifest_columns:
        lines += ["### Age-like manifest columns", ""]
        for path, cols in result.manifest_columns.items():
            lines.append(f"- `{path}`: {', '.join(f'`{c}`' for c in cols)}")
        lines.append("")

    lines += [
        "## Manual verification (required)",
        "",
        "- [ ] Read the HiACC documentation and record how adult and child",
        "      recordings are actually distinguished (folder / manifest / filename).",
        "- [ ] Confirm every child recording is inside" f" `{QUARANTINE_DIR}/`.",
        "- [ ] Confirm the adult count matches the published HiACC adult subset size.",
        "- [ ] Confirm no child speaker id appears in any manifest under" " `data/manifests/`.",
        "- [ ] Re-run `pytest tests/test_preprocess_quarantine.py tests/test_splits.py`.",
        "",
        "Checked by: ______________________  Date: ____________",
        "",
        "> Until this is signed, spoof generation and preprocessing of HiACC stay",
        "> blocked (see the team git rules §10 and `Works updates.md`).",
    ]
    return "\n".join(lines) + "\n"


#: Markers of content a human wrote into the generated report. The audit cannot
#: reproduce any of them: an incident narrative, a ticked verification box, or a
#: signature line.
HANDWRITTEN_MARKERS = ("⚠️ Incident", "- [x]", "Checked by: ")


def would_lose_handwritten(existing: str, report: str) -> list[str]:
    """Markers present in ``existing`` that a fresh ``report`` would destroy.

    The report is generated, but people write *into* it -- the 12 Aug incident
    narrative and the completed manual verification both live there, and neither
    can be regenerated. Re-running the audit on a second machine overwrote the
    file wholesale and silently lost them; the run prints "wrote ..." either way,
    so the damage only shows up in a git diff someone happens to read.
    """
    return [m for m in HANDWRITTEN_MARKERS if m in existing and m not in report]


def main() -> None:
    """CLI: ``python -m src.data.quarantine [--root DIR] [--out FILE]``."""
    import argparse

    parser = argparse.ArgumentParser(description="Audit the HiACC child quarantine")
    parser.add_argument("--root", default="data/raw/hiacc")
    parser.add_argument("--out", default="docs/qa/child_quarantine_check.md")
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite the report even if it carries hand-written sections",
    )
    args = parser.parse_args()

    result = audit(args.root)
    report = render_report(result)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists() and not args.force:
        handwritten = would_lose_handwritten(out.read_text(encoding="utf-8"), report)
        if handwritten:
            print(f"REFUSING to overwrite {out}")
            print(f"  it carries hand-written content the audit cannot regenerate: {handwritten}")
            print("  write the fresh audit elsewhere:")
            print(f"    python -m src.data.quarantine --root {args.root} --out <new-file>.md")
            print("  or pass --force if you have preserved what matters.")
            raise SystemExit(2)

    out.write_text(report, encoding="utf-8")

    print(f"wrote {out}")
    print(f"audio files: {result.total_audio} (quarantined: {result.quarantined_audio})")
    danger = result.unquarantined_findings
    if danger:
        print(f"WARNING: {len(danger)} child-looking item(s) outside quarantine")
        for finding in danger[:10]:
            print(f"  - {finding.path} ({finding.detail})")
    if result.needs_manual_split:
        print("WARNING: nothing quarantined but age-like manifest columns exist")
    print("this audit does not pass the gate; a human signs the report")


if __name__ == "__main__":
    main()
