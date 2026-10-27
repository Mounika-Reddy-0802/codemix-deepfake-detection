"""The paper's real-call table must match the scored calls (experiments/results/real_calls_v1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "experiments/results/real_calls_v1/summary.json"
#: Row label in Table tab:realcalls -> (kind, tool) in the report.
ROWS = {
    "Genuine, live": ("genuine_live", "none"),
    "Genuine, replayed": ("genuine_replay", "none"),
    "XTTS-v2 clones, replayed": ("clone_replay", "xtts_v2"),
    "Chatterbox clones, replayed": ("clone_replay", "chatterbox"),
}


def _counts(entry: dict) -> dict[tuple[str, str], tuple[int, int, int]]:
    """(kind, tool) -> (calls, warned, likely cloned), back from the reported shares."""
    out = {}
    for row in entry["summary"]:
        n = row["calls"]
        out[(row["kind"], row["tool"])] = (
            n,
            round(row["caution_pct"] * n / 100),
            round(row["likely_cloned_pct"] * n / 100),
        )
    return out


def _table_rows(tex: str) -> list[list[str]]:
    table = tex.split(r"\label{tab:realcalls}")[1].split(r"\end{tabular}")[0]
    rows = []
    for line in table.splitlines():
        cells = [c.strip() for c in line.strip().rstrip("\\").strip().split("&")]
        if cells[0] in ROWS:
            rows.append(cells)
    return rows


@pytest.mark.skipif(not SUMMARY.is_file(), reason="real-call results not present")
def test_paper_real_call_table_matches_the_scored_calls() -> None:
    result = json.loads(SUMMARY.read_text(encoding="utf-8"))
    deployed = _counts(result["models"]["deployed_g711"])
    final = _counts(result["models"]["final_g711"])
    rows = _table_rows((ROOT / "paper/sections/system.tex").read_text(encoding="utf-8"))
    assert len(rows) == len(ROWS)
    total = 0
    for cells in rows:
        key = ROWS[cells[0]]
        n, deployed_likely, final_warned, final_likely = map(int, cells[1:5])
        assert (n, deployed_likely) == (deployed[key][0], deployed[key][2]), cells[0]
        assert (final_warned, final_likely) == final[key][1:], cells[0]
        total += n
    assert total == result["calls_recorded"]  # every scored call is in the table


def test_table_rows_are_read_from_the_latex() -> None:
    tex = (
        "\\label{tab:realcalls}\n\\begin{tabular}{lrrrr}\n"
        "Genuine, live & 8 & 8 & 3 & 1 \\\\\n"
        "Chatterbox clones, replayed & 4 & 4 & 1 & 0 \\\\\n\\end{tabular}"
    )
    assert _table_rows(tex) == [
        ["Genuine, live", "8", "8", "3", "1"],
        ["Chatterbox clones, replayed", "4", "4", "1", "0"],
    ]
