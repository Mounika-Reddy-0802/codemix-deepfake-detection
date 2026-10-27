"""Every number the demo pages show equals the committed result file it is read from.

The pages hold no numbers: they ask ``/api/headline``, ``/api/real_calls`` and
``/api/limits``, which answer from ``live_call.results_view``. These tests read the
same files independently and check the two agree, then check that no scene page has a
result typed into its HTML.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from live_call import results_view as rv

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "experiments" / "results" / "real_calls_v1"


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1"})


@pytest.mark.parametrize("model", ["deployed", "final"])
def test_the_real_call_table_recounts_from_the_per_call_csv(model):
    calls = pd.read_csv(REAL / f"calls_{model}_g711.csv")
    summary = json.loads((REAL / "summary.json").read_text(encoding="utf-8"))["models"][
        f"{model}_g711"
    ]
    for row in rv.real_call_table():
        mine = calls[(calls["kind"] == row["kind"]) & (calls["tool"] == row["tool"])]
        got = row[model]
        assert got["calls"]["value"] == len(mine)
        assert got["warned"]["value"] == int(_truthy(mine["caution"]).sum())
        assert got["likely"]["value"] == int(_truthy(mine["likely_cloned"]).sum())
        entry = next(
            s for s in summary["summary"] if s["kind"] == row["kind"] and s["tool"] == row["tool"]
        )
        assert got["warned_pct"]["value"] == entry["caution_pct"]
        assert got["warned_pct"]["value"] == pytest.approx(
            100 * got["warned"]["value"] / len(mine), abs=0.06
        )
        assert got["likely_pct"]["value"] == pytest.approx(
            100 * got["likely"]["value"] / len(mine), abs=0.06
        )
        assert entry["calls"] == len(mine)


def test_the_real_call_table_is_the_reviewed_one():
    """The four rows of the final-review plan, section 5, as the committed files give them."""
    rows = {(r["kind"], r["tool"]): r for r in rv.real_call_table()}
    live = rows[("genuine_live", "none")]
    xtts = rows[("clone_replay", "xtts_v2")]
    chatterbox = rows[("clone_replay", "chatterbox")]
    assert (
        live["deployed"]["likely"]["value"],
        live["final"]["warned"]["value"],
        live["final"]["likely"]["value"],
    ) == (8, 3, 1)
    assert (xtts["final"]["warned"]["value"], xtts["final"]["likely"]["value"]) == (5, 5)
    assert chatterbox["final"]["likely"]["value"] == 0
    assert sum(r["deployed"]["calls"]["value"] for r in rows.values()) == 30


def test_the_call_chain_matches_the_figure():
    from src.reporting.real_call_figure import table

    figure = table()
    chain = rv.call_chain()
    for stage in chain["stages"]:
        for model in ("deployed", "final"):
            expected = figure[(figure["model"] == model) & (figure["stage"] == stage["key"])][
                "mean_p_genuine"
            ]
            assert stage[model]["value"] == pytest.approx(float(expected.iloc[0]), abs=5e-5), (
                model,
                stage["key"],
            )
    assert chain["models"]["final"]["threshold"]["value"] == pytest.approx(0.6868554928153753)


def _resolve(source: str):
    """Follow ``file:field[i].sub`` into a committed JSON file."""
    file, _, path = source.partition(":")
    node = json.loads((ROOT / file).read_text(encoding="utf-8"))
    for part in re.findall(r"[^.\[\]]+|\[\d+\]", path):
        node = node[int(part[1:-1])] if part.startswith("[") else node[part]
    return node


def _cited(tree):
    if isinstance(tree, dict):
        if set(tree) >= {"value", "source"}:
            yield tree
        else:
            for v in tree.values():
                yield from _cited(v)
    elif isinstance(tree, list):
        for v in tree:
            yield from _cited(v)


def test_every_headline_number_is_its_files_value():
    checked = 0
    for entry in _cited(rv.headline()):
        if not entry["source"].split(":")[0].endswith(".json"):
            continue
        raw = _resolve(entry["source"])
        value = entry["value"]
        # a rate is shown either as stored or as a percentage, rounded for display
        as_stored = value == pytest.approx(raw, abs=0.051)
        as_percent = isinstance(raw, float) and value == pytest.approx(100 * raw, abs=0.051)
        assert as_stored or as_percent, f"{entry['source']}: {value} vs {raw}"
        checked += 1
    assert checked >= 40


def test_the_headline_numbers_are_the_reviewed_ones():
    h = rv.headline()
    assert h["s1"]["genuine_called_fake_pct"]["value"] == 94.8
    assert h["s1"]["english_eer"]["value"] == 0.85
    assert h["s1"]["codemix_line_eer"]["value"] == 58.42
    assert h["deployed"]["line_eer_mean"]["value"] == 3.27
    assert h["deployed"]["english_eer"]["value"] == 2.42
    assert h["final"]["new_tortoise"]["eer"]["value"] == 8.8
    assert h["final"]["new_tortoise"]["floor"]["value"] == 36.5
    assert h["ladders"]["final"]["caution_after"]["value"] == 3


def test_limits_cite_their_numbers_and_say_what_landed():
    data = rv.limits()
    assert [lim["key"] for lim in data["limits"]] == [
        "spontaneous",
        "chatterbox",
        "late",
        "carrier",
    ]
    for lim in data["limits"]:
        assert lim["tonight"]["what"] and isinstance(lim["tonight"]["landed"], bool)
        for cited in _cited(lim["numbers"]):
            assert cited["source"].startswith(("experiments/", "configs/"))


SCENE_PAGES = [
    "demo.html",
    "threat.html",
    "lanes.html",
    "live.html",
    "network.html",
    "trust.html",
    "limits.html",
    "index.html",
    "console.html",
    "phone_receiver.html",
    "stage.js",
]
#: Results that must reach a page through the API, never typed into it.
TYPED = ["94.8", "58.42", "36.5", "8.8%", "3.27", "0.83", "0.98", "0.38", "25%", "83%", "5 of 6"]


@pytest.mark.parametrize("page", SCENE_PAGES)
def test_no_scene_page_has_a_result_typed_in(page):
    html = (ROOT / "live_call" / "static" / page).read_text(encoding="utf-8")
    html = re.sub(
        r"<style>.*?</style>", "", html, flags=re.S
    )  # sizes and keyframes are not results
    found = [n for n in TYPED if n in html]
    assert not found, f"{page} holds {found}: read them from the results API instead"
