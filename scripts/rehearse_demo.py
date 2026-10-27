"""Rehearse the final-review demo against a running server: every call, every lane.

Each call in ``configs/demo_rehearsal.json`` is replayed through the server's exact live
path (``POST /api/replay/{id}``, played as fast as it can be scored) and, when the
three-lane view is configured, through every lane (``POST /api/replay_lanes/{id}``).
The observed peak verdict and the time of the first caution and of the first "likely
cloned" are printed beside the expected verdict:

    call  detector  expected  observed  caution s  likely s  result

The demo is shown only if this prints PASS on every row in every round; by default it
runs two rounds, the plan's "twice in a row". Exit status 0 means every row passed.

Run::

    python scripts/rehearse_demo.py --server http://localhost:8000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live_call import rehearsal  # noqa: E402


def run_round(client, plan: dict, with_lanes: bool) -> list[rehearsal.Row]:
    primary, lanes = {}, {}
    for call in plan["calls"]:
        cid = call["id"]
        r = client.post(f"/api/replay/{cid}", params={"realtime": "false", "wait": "true"})
        primary[cid] = r.json() if r.status_code == 200 else None
        if with_lanes:
            r = client.post(f"/api/replay_lanes/{cid}", params={"realtime": "false"})
            lanes[cid] = r.json()["lanes"] if r.status_code == 200 else []
    return rehearsal.rows(plan, primary, lanes if with_lanes else None)


def main() -> int:
    parser = argparse.ArgumentParser(description="Rehearse every demo call through the live path")
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument("--rounds", type=int, default=2, help="rounds that must all pass")
    parser.add_argument("--plan", default=str(rehearsal.REHEARSAL_FILE))
    parser.add_argument("--no-lanes", action="store_true", help="check the live model only")
    args = parser.parse_args()

    import httpx

    plan = rehearsal.load(args.plan)
    client = httpx.Client(base_url=args.server.rstrip("/"), timeout=600)
    try:
        health = client.get("/api/health").json()
    except httpx.HTTPError as exc:
        print(f"server not reachable at {args.server}: {exc}")
        return 2
    detector = health["detector"]
    print(
        f"server {args.server}: model {detector['checkpoint']}, threshold "
        f"{detector['threshold']:.3f} ({detector.get('threshold_source')}), "
        f"{health['demo_clips']} demo calls"
    )
    problems = []
    if detector["checkpoint"] != plan["primary"]["checkpoint"]:
        problems.append(
            f"server runs {detector['checkpoint']}, the demo expects {plan['primary']['checkpoint']}"
        )
    if detector.get("threshold_source") != plan["primary"]["threshold_file"]:
        problems.append(
            f"server threshold file is {detector.get('threshold_source')}, "
            f"the demo expects {plan['primary']['threshold_file']}"
        )
    problems += rehearsal.check_clips(client.get("/api/demo-clips").json(), plan)

    response = client.get("/api/lanes")  # absent on a server without the lanes view
    lanes_info = response.json() if response.status_code == 200 else {"lanes": []}
    roles = {lane["role"] for lane in lanes_info["lanes"]}
    wanted = {role for call in plan["calls"] for role in call.get("lanes", {})}
    with_lanes = not args.no_lanes and wanted <= roles
    if not args.no_lanes and not with_lanes:
        problems.append(
            "the three-lane view is not configured (DFD_LANES): lanes "
            f"{sorted(wanted - roles)} missing; rerun with --no-lanes to check the live model only"
        )
    problems += lanes_info.get("problems", [])
    if problems:
        print("\nNOT READY:")
        for p in problems:
            print("  -", p)
        return 1

    passed = 0
    for n in range(1, args.rounds + 1):
        start = time.time()
        result = run_round(client, plan, with_lanes)
        ok = all(r.ok for r in result)
        print(f"\nround {n} of {args.rounds} ({time.time() - start:.0f} s)")
        print(rehearsal.table(result))
        if not ok:
            failed = sum(not r.ok for r in result)
            print(
                f"\nFAIL: {failed} of {len(result)} rows. Do not show the demo until this passes."
            )
            return 1
        passed += 1
    print(f"\nPASS on every row, {passed} round{'s' if passed != 1 else ''} in a row.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
