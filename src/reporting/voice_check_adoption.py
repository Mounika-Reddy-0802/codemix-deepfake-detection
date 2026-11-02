"""Does the spontaneous recipe's SSL-AASIST replace the voice check's detector?

The rule was written before either candidate was measured in the voice check
(``docs/results/spontaneous_training_v1.md``, "Voice check", commit ``98d145c``) and is coded
here unchanged. It reads the stored scores the accuracy table reads
(``src/reporting/any_audio_accuracy.py``), each checkpoint at its own dev-chosen threshold.
Seed 1234 replaces the current detector only if both seeds meet all four conditions::

    python -m src.reporting.voice_check_adoption --out experiments/results/voice_check_adoption_v1.json
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

REFERENCE = "ssl_s1234_spont"  # the current voice check: frozen seed 1234 at 0.082
CANDIDATES = ("ssl_spontrain_s1234", "ssl_spontrain_s2025")
ADOPTED = "ssl_spontrain_s1234"  # the seed the voice check and every deployed model use

SPONTANEOUS_MIN = 97.3  # HiACC genuine judged real: the reference's 99.3% less 2 points
READ_MIN = 99.0  # read genuine judged real, new speakers and language control
LIVE_MIN = 11  # of the 12 genuine live calls
REPLAY_MIN = 5  # of the 6 genuine replays
UNSEEN_MEAN_MIN = 54.1  # the reference's 44.1% plus 10 points
UNSEEN_DROP_MAX = 10.0  # no unseen-tool row more than this below the reference
XTTS_MIN = 98.0

#: (name, set key, rows) of the five unseen-tool rows on the G.711 line
UNSEEN = (
    ("tortoise_new", "new_speakers", ("spoof", "tortoise")),
    ("tortoise_cm04", "cm04", ("spoof", None)),
    ("indicf5", "cm07", ("spoof", None)),
    ("chatterbox", "cm09", ("spoof", None)),
    ("rvc", "rvc_test", ("spoof", None)),
)
GENUINE = (
    ("hiacc", "hiacc"),
    ("read_new_speakers", "new_speakers"),
    ("read_control", "control"),
)


def measures(model: str) -> dict:
    """Every quantity the rule reads, for one column of the accuracy table."""
    from src.reporting import any_audio_accuracy as acc

    def pct(key, rows, expected):
        r = acc.accuracy(model, key, rows, expected)
        return None if r is None else r[0]

    out = {name: pct(key, ("bonafide", None), "bonafide") for name, key in GENUINE}
    out["xtts"] = pct("new_speakers", ("spoof", "xtts_v2"), "spoof")
    out.update({name: pct(key, rows, "spoof") for name, key, rows in UNSEEN})
    calls = acc.call_counts(model) or {}
    out["genuine_live"] = calls.get(("genuine_live", "none"), (None, None))[0]
    out["genuine_replay"] = calls.get(("genuine_replay", "none"), (None, None))[0]
    return out


def conditions(m: dict, ref: dict) -> dict[str, bool | None]:
    """The four conditions for one candidate; None where a quantity is missing."""
    names = [n for n, _, _ in UNSEEN]
    needs = {
        "1_spontaneous": ["hiacc"],
        "2_read": ["read_new_speakers", "read_control", "genuine_live", "genuine_replay"],
        "3_unseen": names,
        "4_xtts": ["xtts"],
    }
    out: dict[str, bool | None] = {}
    for cond, keys in needs.items():
        if any(m.get(k) is None for k in keys) or (
            cond == "3_unseen" and any(ref.get(k) is None for k in keys)
        ):
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
    if "3_unseen" not in out:
        out["3_unseen"] = mean(m[n] for n in names) >= UNSEEN_MEAN_MIN and all(
            m[n] >= ref[n] - UNSEEN_DROP_MAX for n in names
        )
    if "4_xtts" not in out:
        out["4_xtts"] = m["xtts"] >= XTTS_MIN
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
    names = [n for n, _, _ in UNSEEN]
    return {
        "reference": REFERENCE,
        "measures": per_model,
        "unseen_mean": {
            m: (None if any(v.get(n) is None for n in names) else mean(v[n] for n in names))
            for m, v in per_model.items()
        },
        "conditions": results,
        "decision": decision,
    }


def main() -> None:
    """CLI: measure the reference and both candidates, apply the rule, write one JSON."""
    import argparse

    parser = argparse.ArgumentParser(description="The voice check's detector, by the fixed rule")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = decide({m: measures(m) for m in (REFERENCE, *CANDIDATES)})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for m, conds in report["conditions"].items():
        print(m, conds)
    print("unseen mean:", {m: v and round(v, 1) for m, v in report["unseen_mean"].items()})
    print("DECISION:", report["decision"])


if __name__ == "__main__":
    main()
