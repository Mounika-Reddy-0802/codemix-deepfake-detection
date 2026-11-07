"""Does voice check v2 replace the voice check's detector? (P-042)

v2 is SSL-AASIST trained on the spontaneous recipe plus the generator study's training half:
IndicF5 and Chatterbox clones of the 31 ``gen_train`` speakers and their genuine clips
(``src/data/voice_check_v2``). The rule was written and committed before v2 was trained, and
is coded here unchanged. It is P-040's rule with two changes that follow from what v2 is:

* every quantity is measured **without the ``gen_train`` speakers**, for the reference and
  the candidates alike, so no voice v2 heard in training is judged;
* IndicF5 and Chatterbox are now tools v2 was trained on (other speakers), so instead of
  counting as unseen they get their own bar: v2 must catch **at least half** of each tool's
  clones (the reference catches 12.5% and 4.4%). Tortoise and RVC stay unseen and must not
  fall more than 10 points below the reference.

Each candidate at its own operating point, chosen on dev only by the unchanged rule (P-031),
as for the spontaneous recipe. Seed 1234 replaces the current detector only if **both** seeds
meet all five conditions::

    python -m src.reporting.voice_check_v2 --out experiments/results/voice_check_v2.json
"""

from __future__ import annotations

import json
from pathlib import Path

REFERENCE = "ssl_s1234_spont"  # the current voice check: frozen seed 1234 at 0.082
CANDIDATES = ("ssl_vcv2_s1234", "ssl_vcv2_s2025")
ADOPTED = "ssl_vcv2_s1234"  # the seed the voice check and every deployed model use

SPONTANEOUS_MIN = 97.3  # as P-040
READ_MIN = 99.0
LIVE_MIN = 11  # of the 12 genuine live calls
REPLAY_MIN = 5  # of the 6 genuine replays
XTTS_MIN = 98.0
NEW_TOOL_MIN = 50.0  # IndicF5 and Chatterbox clones caught, each, held-out speakers
UNSEEN_DROP_MAX = 10.0  # Tortoise and RVC: none more than this below the reference

GENUINE = (
    ("hiacc", "hiacc"),
    ("read_new_speakers", "new_speakers"),
    ("read_control", "control"),
)
NEW_TOOLS = (("indicf5", "cm07"), ("chatterbox", "cm09"))
#: (name, set key, rows) of the tools v2 never heard, on the G.711 line
UNSEEN = (
    ("tortoise_new", "new_speakers", ("spoof", "tortoise")),
    ("tortoise_cm04", "cm04", ("spoof", None)),
    ("rvc", "rvc_test", ("spoof", None)),
)


def measures(model: str, exclude: frozenset[str]) -> dict:
    """Every quantity the rule reads, for one model, without the ``exclude`` speakers."""
    from src.reporting import any_audio_accuracy as acc

    def pct(key, rows, expected):
        r = acc.accuracy(model, key, rows, expected, exclude)
        return None if r is None else r[0]

    out = {name: pct(key, ("bonafide", None), "bonafide") for name, key in GENUINE}
    out["xtts"] = pct("new_speakers", ("spoof", "xtts_v2"), "spoof")
    out.update({name: pct(key, ("spoof", None), "spoof") for name, key in NEW_TOOLS})
    out.update({name: pct(key, rows, "spoof") for name, key, rows in UNSEEN})
    calls = acc.call_counts(model) or {}
    out["genuine_live"] = calls.get(("genuine_live", "none"), (None, None))[0]
    out["genuine_replay"] = calls.get(("genuine_replay", "none"), (None, None))[0]
    return out


def conditions(m: dict, ref: dict) -> dict[str, bool | None]:
    """The five conditions for one candidate; None where a quantity is missing."""
    unseen = [n for n, _, _ in UNSEEN]
    needs = {
        "1_spontaneous": ["hiacc"],
        "2_read": ["read_new_speakers", "read_control", "genuine_live", "genuine_replay"],
        "3_new_tools": [n for n, _ in NEW_TOOLS],
        "4_unseen": unseen,
        "5_xtts": ["xtts"],
    }
    out: dict[str, bool | None] = {}
    for cond, keys in needs.items():
        missing = any(m.get(k) is None for k in keys)
        if cond == "4_unseen":
            missing = missing or any(ref.get(k) is None for k in keys)
        if missing:
            out[cond] = None
    if "1_spontaneous" not in out:
        out["1_spontaneous"] = m["hiacc"] >= SPONTANEOUS_MIN
    if "2_read" not in out:
        out["2_read"] = (
            m["read_new_speakers"] >= READ_MIN
            and m["read_control"] >= READ_MIN
            and m["genuine_live"] >= LIVE_MIN
            and m["genuine_replay"] >= REPLAY_MIN
        )
    if "3_new_tools" not in out:
        out["3_new_tools"] = all(m[n] >= NEW_TOOL_MIN for n, _ in NEW_TOOLS)
    if "4_unseen" not in out:
        out["4_unseen"] = all(m[n] >= ref[n] - UNSEEN_DROP_MAX for n in unseen)
    if "5_xtts" not in out:
        out["5_xtts"] = m["xtts"] >= XTTS_MIN
    return out


def decide(per_model: dict[str, dict]) -> dict:
    """Apply the rule to the reference's and both candidates' measures."""
    ref = per_model[REFERENCE]
    results = {c: conditions(per_model[c], ref) for c in CANDIDATES}
    values = [v for r in results.values() for v in r.values()]
    if any(v is None for v in values):
        decision = "incomplete: the current detector stays"
    elif all(values):
        decision = f"adopt {ADOPTED}"
    else:
        decision = "the current detector stays"
    return {
        "reference": REFERENCE,
        "measures": per_model,
        "conditions": results,
        "decision": decision,
    }


def main() -> None:
    """CLI: measure the reference and both candidates, apply the rule, write one JSON."""
    import argparse

    from src.data import generator_study as gs

    parser = argparse.ArgumentParser(description="Voice check v2, by the rule fixed before it")
    parser.add_argument("--pools", default=f"data/manifests/{gs.POOLS_OUT}")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    exclude = frozenset(gs.train_speakers(args.pools))
    report = decide({m: measures(m, exclude) for m in (REFERENCE, *CANDIDATES)})
    report["excluded_speakers"] = len(exclude)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for m, conds in report["conditions"].items():
        print(m, conds)
    print("DECISION:", report["decision"])


if __name__ == "__main__":
    main()
