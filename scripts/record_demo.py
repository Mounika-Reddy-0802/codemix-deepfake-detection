"""Record the backup demo video by driving the live console in a real browser.

The viva risk register lists demo flakiness as a top risk, mitigated by a recording
made in advance. This produces that recording from the real system -- real model,
real calls, real verdicts -- rather than a screen capture someone has to redo by
hand every time the UI changes.

    python scripts/record_demo.py --out C:/dfdata/demo_recordings

It needs the detection server running with demo clips:

    DEMO_CLIPS_DIR=/path/to/clips python -m uvicorn live_call.server:app --port 8000

Three calls are recorded in the order they should be presented: a genuine caller
that stays green, a cloned caller that raises the warning, and the unseen Tortoise
tool that is missed. The miss is deliberately included -- a recording that shows
only successes is worth less in front of an examiner than one that shows the limit.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8000"

#: (clip id, caption, seconds to hold after answering) in presentation order.
SCENES: tuple[tuple[str, str, int], ...] = (
    ("rvc_genuine", "A real human caller: the line stays green", 18),
    ("xtts_clone", "A cloned voice: the receiver is warned mid-call", 22),
    ("tortoise_clone", "An unseen cloning tool: this one is missed", 18),
)


def wait_for_two_parties(page, room: str, timeout_ms: int = 12_000) -> None:
    """Both handsets must be registered before dialling, or the call goes nowhere."""
    for _ in range(timeout_ms // 250):
        with urllib.request.urlopen(f"{BASE}/api/lines", timeout=5) as response:
            lines = json.load(response)
        line = next((entry for entry in lines if entry["room"] == room), None)
        if line and len(line["parties"]) == 2:
            return
        page.wait_for_timeout(250)
    raise RuntimeError(f"room {room} never registered both handsets")


def record(out_dir: Path, headed: bool = False) -> Path:
    from playwright.sync_api import sync_playwright

    out_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            channel="chrome",
            headless=not headed,
            args=["--autoplay-policy=no-user-gesture-required"],
        )
        context = browser.new_context(
            viewport={"width": 1600, "height": 1000},
            record_video_dir=str(out_dir),
            record_video_size={"width": 1600, "height": 1000},
        )
        page = context.new_page()

        for index, (clip, caption, hold) in enumerate(SCENES, start=1):
            room = f"recording-{index}"
            print(f"  scene {index}/{len(SCENES)}: {caption}")
            page.goto(f"{BASE}/demo?room={room}", wait_until="networkidle")
            wait_for_two_parties(page, room)
            page.wait_for_timeout(2500)  # let the viewer read the idle screen

            page.select_option("#clip", clip)
            page.wait_for_timeout(1500)
            page.click("#callBtn")
            page.wait_for_selector("#acceptBtn:not([disabled])", timeout=20_000)
            page.wait_for_timeout(2000)  # the phone rings on screen
            page.click("#acceptBtn")
            page.wait_for_timeout(hold * 1000)

            page.click("#callerEnd")
            page.wait_for_timeout(3000)

        context.close()  # the video is only written on close
        browser.close()

    videos = sorted(out_dir.glob("*.webm"), key=lambda p: p.stat().st_mtime)
    if not videos:
        raise RuntimeError("playwright wrote no video")
    final = out_dir / "live_call_demo_backup.webm"
    if final.exists():
        final.unlink()
    videos[-1].rename(final)
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="C:/dfdata/demo_recordings")
    parser.add_argument("--headed", action="store_true", help="show the browser while recording")
    args = parser.parse_args()

    try:
        with urllib.request.urlopen(f"{BASE}/api/health", timeout=10) as response:
            health = json.load(response)
    except Exception as err:  # noqa: BLE001 - a missing server is the common mistake
        print(f"the detection server is not answering on {BASE}: {err}", file=sys.stderr)
        return 1
    if not health["detector"]["model_loaded"]:
        print("the model has not finished loading; wait and re-run", file=sys.stderr)
        return 1
    if not health.get("demo_clips"):
        print("no demo clips: start the server with DEMO_CLIPS_DIR set", file=sys.stderr)
        return 1

    print(f"recording against {health['detector']['checkpoint']} on {health['detector']['device']}")
    video = record(Path(args.out), headed=args.headed)
    size_mb = video.stat().st_size / 1_048_576
    print(f"\nwrote {video}  ({size_mb:.1f} MB)")
    print("Play it before the viva; it is the fallback if the live demo misbehaves.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
