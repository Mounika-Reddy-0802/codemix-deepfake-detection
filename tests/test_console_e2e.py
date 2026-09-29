"""End-to-end checks of the live console in a real browser.

Every serious defect in the demo was found by driving a browser by hand: the warning
that never fired because the page read the wrong field, the room that accumulated
dead tracks until negotiation failed, the missing queue that scored zero windows
while looking healthy. None of those were visible to a unit test, because each of
them was in the wiring between parts that all worked on their own.

These tests place real calls against a running server with the real model, so they
are opt-in rather than part of the normal suite:

    DEMO_CLIPS_DIR=/path/to/clips python -m uvicorn live_call.server:app --port 8000
    pytest tests/test_console_e2e.py

They skip -- never fail -- when Playwright, Chrome or the server is absent, so CI
and a laptop without the checkpoints stay green.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

BASE = "http://127.0.0.1:8000"
#: Two low windows raise the caution at ~6 s and four the warning at ~10 s; this is
#: the budget for the whole sequence including connection setup.
WARNING_BUDGET_MS = 25_000


def _health() -> dict | None:
    try:
        with urllib.request.urlopen(f"{BASE}/api/health", timeout=5) as response:
            return json.load(response)
    except (urllib.error.URLError, OSError, TimeoutError):
        return None


@pytest.fixture(scope="module")
def server() -> dict:
    health = _health()
    if health is None:
        pytest.skip(f"no detection server on {BASE}")
    if not health["detector"]["model_loaded"]:
        pytest.skip("the model is not loaded")
    if not health.get("demo_clips"):
        pytest.skip("the server has no demo clips (set DEMO_CLIPS_DIR)")
    return health


@pytest.fixture(scope="module")
def browser(server):
    playwright_api = pytest.importorskip("playwright.sync_api")
    with playwright_api.sync_playwright() as playwright:
        try:
            launched = playwright.chromium.launch(
                channel="chrome", args=["--autoplay-policy=no-user-gesture-required"]
            )
        except Exception as err:  # noqa: BLE001 - no browser on this machine
            pytest.skip(f"cannot launch Chrome: {err}")
        yield launched
        launched.close()


def _place_call(browser, clip: str, room: str, hold_ms: int):
    """Dial, answer, hold, and hand back the page plus any javascript errors."""
    page = browser.new_page(viewport={"width": 1400, "height": 1000})
    errors: list[str] = []
    page.on("pageerror", lambda err: errors.append(str(err)))

    page.goto(f"{BASE}/demo?room={room}", wait_until="networkidle")
    for _ in range(40):  # both handsets must register before dialling
        with urllib.request.urlopen(f"{BASE}/api/lines", timeout=5) as response:
            lines = json.load(response)
        line = next((entry for entry in lines if entry["room"] == room), None)
        if line and len(line["parties"]) == 2:
            break
        page.wait_for_timeout(250)

    page.select_option("#clip", clip)
    page.click("#callBtn")
    page.wait_for_selector("#acceptBtn:not([disabled])", timeout=20_000)
    page.click("#acceptBtn")
    page.wait_for_timeout(hold_ms)
    return page, errors


def test_a_cloned_caller_warns_the_receiver_and_never_the_caller(browser):
    page, errors = _place_call(browser, "xtts_clone", "e2e-clone", hold_ms=0)
    try:
        page.wait_for_selector("#alertBox.show.likely_fake", timeout=WARNING_BUDGET_MS)
        assert "likely cloned" in page.inner_text("#alertBox").lower()
        assert page.inner_text("#verdictText") == "Likely cloned voice"

        # The whole point: the caller's side must carry no verdict. Checked on the
        # elements that could show one -- not on the panel text, which legitimately
        # contains the operator's clip selector ("XTTS-v2 cloned voice") and its
        # ground-truth tag. Those are demo controls, not something a real caller sees.
        assert page.locator("#callerPanel .alertbox").count() == 0
        assert page.locator("#callerPanel .vbig").count() == 0
        caller_state = page.inner_text("#callerState").lower()
        for leaked in ("cloned", "likely", "suspicious", "warning", "synthetic"):
            assert leaked not in caller_state, f"the caller's status line leaks {leaked!r}"

        assert not errors, f"javascript errors: {errors[:3]}"
    finally:
        page.close()


def test_a_genuine_caller_raises_no_alarm(browser):
    page, errors = _place_call(browser, "rvc_genuine", "e2e-genuine", hold_ms=18_000)
    try:
        assert page.inner_text("#verdictText") == "Genuine voice"
        assert "show" not in (page.get_attribute("#alertBox", "class") or "")
        assert page.inner_text("#cntFake").startswith("0"), "no window should be fake-leaning"
        assert not errors, f"javascript errors: {errors[:3]}"
    finally:
        page.close()


def test_every_window_is_scored_and_the_timeline_fills(browser):
    """Guards the bug where a broken room queue scored nothing while looking healthy."""
    page, errors = _place_call(browser, "rvc_genuine", "e2e-windows", hold_ms=14_000)
    try:
        scored = page.evaluate("() => points.length")
        assert scored >= 4, f"only {scored} windows scored in 14 s; expected one per 2 s"
        assert page.inner_text("#mWindows") == str(scored)
        assert not errors
    finally:
        page.close()


def test_hanging_up_clears_the_clock_on_both_handsets(browser):
    """A frozen duration under 'Call ended' reads as a call that is still running."""
    page, _ = _place_call(browser, "rvc_genuine", "e2e-hangup", hold_ms=6000)
    try:
        page.click("#callerEnd")
        page.wait_for_timeout(1500)
        assert page.inner_text("#callerTimer") == ""
        assert page.inner_text("#rxTimer") == ""
        assert "lasted" in page.inner_text("#callerState")
        assert page.inner_text("#rxPeer") == "no incoming call"
        assert page.is_enabled("#callBtn"), "a new call must be possible straight away"
    finally:
        page.close()
