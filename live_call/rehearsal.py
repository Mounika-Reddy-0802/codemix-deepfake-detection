"""Does every demo call still do what the demo says it does? (the rehearsal check)

Nothing that could flag a person in front of the reviewers is left to chance: every
call the final-review demo plays has its expected verdict written down in
``configs/demo_rehearsal.json``, for the live model (the final recipe) and for each
lane of the three-detector view. Before the demo is shown, every call is replayed
through the exact live path and compared with that file; the demo goes ahead only
when every row passes, twice in a row.

This module holds the comparison, so the pytest check (``tests/test_demo_rehearsal.py``)
and the command-line check against a running server (``scripts/rehearse_demo.py``) can
never disagree about what counts as a pass. Stdlib only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REHEARSAL_FILE = ROOT / "configs" / "demo_rehearsal.json"
#: The team's own voices: never in the demo folders, under any framing.
TEAM_SPEAKERS = {"P1", "P2", "P3"}
STATES = ("listening", "genuine", "suspicious", "likely_fake")
STATE_WORDS = {
    "listening": "listening",
    "genuine": "genuine",
    "suspicious": "caution",
    "likely_fake": "likely cloned",
}


def load(path: str | Path = REHEARSAL_FILE) -> dict:
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    for call in plan["calls"]:
        for expect in [call["primary"], *call.get("lanes", {}).values()]:
            if expect["peak"] not in STATES:
                raise ValueError(f"{call['id']}: unknown peak state {expect['peak']!r}")
    return plan


@dataclass(frozen=True)
class Row:
    """One call in one detector: what the file expects and what the live path did."""

    call: str
    detector: str  # "primary" or a lane role (s1, deployed, final, ssl)
    expected: str
    observed: str | None
    first_caution_s: float | None
    first_likely_s: float | None
    ok: bool
    why: str = ""

    def cells(self) -> list[str]:
        def secs(x: float | None) -> str:
            return "-" if x is None else f"{x:.0f}"

        return [
            self.call,
            self.detector,
            STATE_WORDS.get(self.expected, self.expected),
            STATE_WORDS.get(self.observed or "", self.observed or "missing"),
            secs(self.first_caution_s),
            secs(self.first_likely_s),
            "PASS" if self.ok else "FAIL",
            self.why,
        ]


def check(call: str, detector: str, expect: dict, observed: dict | None) -> Row:
    """Compare one observed verdict (``peak_state``, first caution and likely times)."""
    if observed is None:
        return Row(call, detector, expect["peak"], None, None, None, False, "not scored")
    peak = observed.get("peak_state")
    caution, likely = observed.get("first_caution_s"), observed.get("first_likely_s")
    problems = []
    if peak != expect["peak"]:
        problems.append(f"peak {peak}, expected {expect['peak']}")
    limit = expect.get("max_first_caution_s")
    if limit is not None and (caution is None or caution > limit):
        problems.append(f"first caution {caution} s, expected by {limit} s")
    limit = expect.get("max_first_likely_s")
    if limit is not None and (likely is None or likely > limit):
        problems.append(f"likely cloned at {likely} s, expected by {limit} s")
    return Row(
        call, detector, expect["peak"], peak, caution, likely, not problems, "; ".join(problems)
    )


def check_clips(clips: list[dict], plan: dict) -> list[str]:
    """Problems with the demo folder itself: unknown calls, missing calls, team audio."""
    problems = []
    planned = {c["id"] for c in plan["calls"]}
    present = {c["id"] for c in clips}
    for missing in sorted(planned - present):
        problems.append(f"call {missing!r} is in the rehearsal file but not in the demo folder")
    for extra in sorted(present - planned):
        problems.append(f"call {extra!r} is in the demo folder but has no expected verdict")
    for clip in clips:
        if str(clip.get("speaker", "")).upper() in TEAM_SPEAKERS:
            problems.append(f"call {clip['id']!r} is a team member's voice: remove it")
    return problems


def rows(
    plan: dict, primary: dict[str, dict], lanes: dict[str, list[dict]] | None = None
) -> list[Row]:
    """Every row of the rehearsal: the live model for each call, then each lane.

    ``primary`` maps a call id to its observed verdict; ``lanes`` maps a call id to the
    lane summaries of a lanes replay (each with a ``role``). Lanes whose role the file
    does not list (a fourth lane added on the night) are reported, not judged.
    """
    out = []
    for call in plan["calls"]:
        cid = call["id"]
        out.append(check(cid, "primary", call["primary"], primary.get(cid)))
        if lanes is None:
            continue
        observed = {lane["role"]: lane for lane in lanes.get(cid, [])}
        for role, expect in call.get("lanes", {}).items():
            out.append(check(cid, role, expect, observed.get(role)))
    return out


def table(result: list[Row]) -> str:
    head = ["call", "detector", "expected", "observed", "caution s", "likely s", "result", "why"]
    body = [r.cells() for r in result]
    widths = [max(len(str(x)) for x in col) for col in zip(head, *body, strict=True)]
    line = lambda cells: "  ".join(str(c).ljust(w) for c, w in zip(cells, widths, strict=True))  # noqa: E731
    return "\n".join([line(head), line(["-" * w for w in widths]), *map(line, body)])


__all__ = [
    "REHEARSAL_FILE",
    "Row",
    "TEAM_SPEAKERS",
    "check",
    "check_clips",
    "load",
    "rows",
    "table",
]
