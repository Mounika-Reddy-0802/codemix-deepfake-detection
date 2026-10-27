"""The final-review demo's scene routes, added to the live-call server by ``create_app``.

``server.py`` serves the live system itself: calls, handsets, Twilio, the dashboard.
The routes here serve the demo's scenes on top of it, each in its own function so the
server reads as a list of what it offers:

- :func:`add_lane_routes`: scene 2, several detectors on one call (``live_call.lanes``).

Every route scores audio through the live path (``StreamingScorer`` + ``VerdictEngine``)
or reads a committed result file; none holds a number of its own.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

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
