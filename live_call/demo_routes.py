"""The final-review demo's scene routes, added to the live-call server by ``create_app``.

``server.py`` serves the live system itself: calls, handsets, Twilio, the dashboard.
The routes here serve the demo's scenes on top of it, each in its own function so the
server reads as a list of what it offers:

- :func:`add_lane_routes`: scene 2, several detectors on one call (``live_call.lanes``);
- :func:`add_live_routes`: scene 3, the address and QR code the receiver's phone opens;
- :func:`add_results_routes`: scenes 1, 4 and 6, numbers read from committed results
  (``live_call.results_view``) and the stage player's prepared audio.

Every route scores audio through the live path (``StreamingScorer`` + ``VerdictEngine``)
or reads a committed result file; none holds a number of its own.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import socket
from collections.abc import Callable
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response

from live_call import results_view
from live_call.lanes import new_run_id, score_lanes

log = logging.getLogger("live_call")


def add_lane_routes(
    api: FastAPI,
    shared,
    find_clip: Callable[[str], dict],
    clip_audio: Callable[[dict], np.ndarray],
) -> None:
    """Scene 2: ``GET /api/lanes``, ``POST /api/replay_lanes/{id}``, ``POST /api/replay_lanes_stop``."""

    @api.get("/api/lanes")
    async def lanes() -> JSONResponse:
        lane_set = shared.lane_set()
        return JSONResponse(
            {
                "lanes": [lane.describe() for lane in lane_set.lanes],
                "problems": lane_set.problems(),
                "playing": shared.lanes_run,
            }
        )

    def lanes_or_400():
        lane_set = shared.lane_set()
        problems = lane_set.problems()
        if problems:
            raise HTTPException(400, "; ".join(problems))
        return lane_set.lanes

    @api.post("/api/replay_lanes/{clip_id}")
    async def replay_lanes(clip_id: str, realtime: bool = True) -> JSONResponse:
        """Score one prepared call through every lane, with the live path's windows and ladder.

        In real time (the default) the call plays at its own speed and every window and
        verdict change is pushed to the dashboard socket tagged with its lane and run;
        the response comes at once. With ``realtime=false`` the whole call is scored and
        the response carries each lane's verdict and window series.
        """
        lane_list = lanes_or_400()
        clip = find_clip(clip_id)
        audio = clip_audio(clip)
        run_id = new_run_id()
        meta = {
            "run": run_id,
            "clip": clip_id,
            "title": clip.get("title", clip_id),
            "description": clip.get("description", ""),
            "label": clip.get("label"),
            "tool": clip.get("tool"),
            "seconds": round(audio.size / 16000, 2),
        }
        executor = shared.detector.executor
        if not realtime:
            results = await score_lanes(audio, lane_list, executor, realtime=False)
            return JSONResponse(
                {
                    **meta,
                    "lanes": [
                        {**r.summary(), "windows": r.windows, "events": r.events} for r in results
                    ],
                }
            )

        if shared.lanes_task is not None and not shared.lanes_task.done():
            shared.lanes_task.cancel()  # the presenter picked another call: stop the old one
        described = [lane.describe() for lane in lane_list]
        shared.hub.publish({"type": "lanes_start", **meta, "lanes": described})

        def emit(message: dict) -> None:
            shared.hub.publish({**message, "run": run_id})

        async def play() -> None:
            try:
                results = await score_lanes(audio, lane_list, executor, realtime=True, emit=emit)
                shared.hub.publish(
                    {"type": "lanes_end", "run": run_id, "lanes": [r.summary() for r in results]}
                )
            except asyncio.CancelledError:
                shared.hub.publish({"type": "lanes_stopped", "run": run_id})
                raise
            except Exception as exc:  # shown on the page instead of failing silently
                log.exception("lanes run failed")
                shared.hub.publish({"type": "lanes_error", "run": run_id, "error": str(exc)})
            finally:
                if shared.lanes_run == run_id:
                    shared.lanes_run = None

        shared.lanes_run = run_id
        task = asyncio.create_task(play())
        shared.lanes_task = task
        shared.tasks.add(task)
        task.add_done_callback(shared.tasks.discard)
        return JSONResponse({**meta, "lanes": described})

    @api.post("/api/replay_lanes_stop")
    async def stop_lanes() -> JSONResponse:
        stopped = shared.lanes_task is not None and not shared.lanes_task.done()
        if stopped:
            shared.lanes_task.cancel()
        return JSONResponse({"stopped": stopped})


def lan_addresses() -> list[str]:
    """This laptop's addresses on the local network, best first (no packet is sent)."""
    found: list[str] = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 9))  # a documentation address: only picks the route
            found.append(probe.getsockname()[0])
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            address = info[4][0]
            if address not in found:
                found.append(address)
    except OSError:
        pass
    return [a for a in found if not a.startswith("127.")]


def add_live_routes(api: FastAPI, tunnel_file: Callable[[], Path]) -> None:
    """Scene 3: ``GET /api/tunnel`` and ``GET /api/qr``, for the receiver's phone."""

    @api.get("/api/tunnel")
    async def tunnel() -> JSONResponse:
        """Where the phone can reach this server: the https tunnel if one is open
        (``scripts/start_tunnel.py`` writes it), else the laptop's Wi-Fi address."""
        path = tunnel_file()
        url = path.read_text(encoding="utf-8").strip() if path.is_file() else ""
        port = int(os.environ.get("DFD_PORT", "8000"))
        return JSONResponse(
            {"url": url or None, "lan": [f"http://{a}:{port}" for a in lan_addresses()]}
        )

    @api.get("/api/qr")
    async def qr(text: str) -> Response:
        """A QR code (SVG) for a link, so the phone opens it without typing."""
        if not text.startswith(("http://", "https://")) or len(text) > 300:
            raise HTTPException(400, "only http(s) links up to 300 characters")
        try:
            import segno
        except ImportError as exc:
            raise HTTPException(404, "QR codes need the segno package") from exc
        buffer = io.BytesIO()  # a standalone SVG (with its namespace), so an <img> renders it
        segno.make(text, error="m").save(
            buffer, kind="svg", scale=8, border=2, dark="#06101f", light="#ffffff", xmldecl=False
        )
        return Response(buffer.getvalue(), media_type="image/svg+xml")


def add_results_routes(api: FastAPI, stage_dir: Callable[[], Path]) -> None:
    """Committed results for the scene pages, and the stage player's prepared audio."""

    @api.get("/api/headline")
    async def headline() -> JSONResponse:
        return JSONResponse(results_view.headline())

    @api.get("/api/real_calls")
    async def real_calls() -> JSONResponse:
        return JSONResponse(results_view.real_calls())

    @api.get("/api/limits")
    async def limits() -> JSONResponse:
        return JSONResponse(results_view.limits())

    def stage_index() -> dict | None:
        import json

        index = stage_dir() / "stage_player.json"
        return json.loads(index.read_text(encoding="utf-8")) if index.is_file() else None

    @api.get("/api/stage_player")
    async def stage_player() -> JSONResponse:
        """One genuine corpus call through each stage of a phone call, and its scores
        (prepared offline by ``scripts/prepare_stage_player.py``)."""
        data = stage_index()
        if data is None:
            return JSONResponse({"available": False, "stages": []})
        return JSONResponse({"available": True, **data})

    @api.get("/api/stage_player/{stage}/audio")
    async def stage_audio(stage: str) -> FileResponse:
        data = stage_index() or {}
        entry = next((s for s in data.get("stages", []) if s["key"] == stage), None)
        if entry is None or not (stage_dir() / entry["file"]).is_file():
            raise HTTPException(404, "unknown stage")
        return FileResponse(stage_dir() / entry["file"])
