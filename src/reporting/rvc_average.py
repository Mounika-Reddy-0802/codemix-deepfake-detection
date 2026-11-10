"""Voice check v2 averaged with the spontaneous recipe, to win back RVC (P-046).

Voice check v2 traded RVC for the two newest tools: against the spontaneous recipe's
SSL-AASIST alone on the same clips, RVC conversions caught fell from 85.8/89.0% to 73.5/77.6%
while IndicF5 and Chatterbox rose to 89-94% (``docs/results/voice_check_v2.md``). The
**average** of the two checkpoints of the same seed may keep both. It is built exactly as the
project's earlier two-checkpoint average (``src.inference.ensemble``): for every window and
every clip, the mean of the two models' P(genuine); its operating point chosen on the same dev
windows by the unchanged rule (P-031, ``src.inference.calibrate choose``), never on a test set
or a call. No training and no new scoring: every input is a score file both models already
wrote.

The rule, fixed before any averaged number exists, for each seed pair (v2 seed s with the
spontaneous recipe's seed s), measured exactly as voice check v2 was (``voice_check_v2
.measures``, the 31 ``gen_train`` speakers left out of the sets that hold them):

1. genuine speech kept to the v2 rule's bars: HiACC >= 97.3%, read speech (new speakers and the
   language control) >= 99.0%, >= 11 of 12 genuine live calls and >= 5 of 6 genuine replays;
2. XTTS-v2 clones caught >= 98.0%;
3. RVC conversions caught at least 5.0 points above v2 of the same seed;
4. IndicF5 and Chatterbox clones caught, each, at most 5.0 points below v2 of the same seed;
5. Tortoise (new speakers and CM04), each, at most 5.0 points below v2 of the same seed.

**recommended** if both pairs meet all five, **not recommended** otherwise, **incomplete** if a
quantity is missing. The average runs two models, so it doubles the voice check's scoring time
and memory; whatever the answer, the page is not switched before the 13 November review, and a
recommended average is put to the team after it::

    python -m src.reporting.rvc_average build      # dev point, averaged clips and calls
    python -m src.reporting.rvc_average judge --out experiments/results/rvc_average.json
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.reporting import voice_check_v2 as v2rule

SEEDS = (1234, 2025)
V2_CLIPS = "experiments/results/voice_check_v2/ssl_s{seed}"  # vcv2_s<seed>__<set>_scores.csv
SPONT_CLIPS = "experiments/results/spontaneous_training_v1/ssl_s{seed}"  # spont_s<seed>__...
OUT_CLIPS = "experiments/results/voice_check_v2_avg/ssl_s{seed}"  # avg_s<seed>__...
V2_DEV = "experiments/calibration_ssl_vcv2_s{seed}"
SPONT_DEV = "experiments/calibration_ssl_spontrain_s{seed}"
OUT_DEV = "experiments/calibration_ssl_vcv2avg_s{seed}"
V2_CALLS = "experiments/results/real_calls_v1/ssl_vcv2_s{seed}"
SPONT_CALLS = "experiments/results/real_calls_v1/ssl_spontrain_s{seed}"
OUT_CALLS = "experiments/results/real_calls_v1/ssl_vcv2avg_s{seed}"
THRESHOLD = "configs/threshold_ssl_aasist_vcv2avg_s{seed}.yaml"
CALIBRATION_REPORT = "experiments/results/threshold_calibration_ssl_vcv2avg_s{seed}.json"

RVC_GAIN_MIN = 5.0
KEEP_DROP_MAX = 5.0
KEPT = ("indicf5", "chatterbox", "tortoise_new", "tortoise_cm04")


def model(seed: int) -> str:
    return f"ssl_vcv2avg_s{seed}"


def logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Log-odds, the convention of every SSL-AASIST clip score file."""
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))


def build_clips(seed: int, root: Path = Path(".")) -> list[str]:
    """Average every set both models scored; writes ``avg_s<seed>__<set>_scores.csv``."""
    from src.inference.ensemble import average_clips

    a_dir = root / V2_CLIPS.format(seed=seed)
    b_dir = root / SPONT_CLIPS.format(seed=seed)
    out = root / OUT_CLIPS.format(seed=seed)
    out.mkdir(parents=True, exist_ok=True)
    done = []
    for a_path in sorted(a_dir.glob(f"vcv2_s{seed}__*_scores.csv")):
        stem = a_path.name[len(f"vcv2_s{seed}__") : -len("_scores.csv")]
        b_path = b_dir / f"spont_s{seed}__{stem}_scores.csv"
        if not b_path.is_file():
            continue
        avg = average_clips(pd.read_csv(a_path), pd.read_csv(b_path))
        avg["score"] = logit(avg["score"])
        avg.to_csv(out / f"avg_s{seed}__{stem}_scores.csv", index=False)
        done.append(stem)
    return done


def build_dev(seed: int) -> Path:
    """The averaged dev windows and the operating point the unchanged rule chooses on them."""
    from src.inference.ensemble import average_windows

    out = Path(OUT_DEV.format(seed=seed))
    out.mkdir(parents=True, exist_ok=True)
    name = "dev_spont_windows.csv"
    avg = average_windows(
        pd.read_csv(Path(V2_DEV.format(seed=seed)) / name),
        pd.read_csv(Path(SPONT_DEV.format(seed=seed)) / name),
    )
    avg.to_csv(out / name, index=False)
    cmd = [
        sys.executable,
        "-m",
        "src.inference.calibrate",
        "choose",
        "--dev",
        str(out / name),
        "--checkpoint-name",
        f"ssl_aasist_vcv2avg_s{seed}",
        "--out",
        THRESHOLD.format(seed=seed),
        "--report",
        CALIBRATION_REPORT.format(seed=seed),
    ]
    subprocess.run(cmd, check=True)
    return Path(THRESHOLD.format(seed=seed))


def build_calls(seed: int) -> Path:
    """The 30 calls' averaged windows and their verdicts at the average's point."""
    from src.inference import real_calls
    from src.inference.ensemble import average_windows
    from src.reporting.any_audio_accuracy import PROJ

    out = Path(OUT_CALLS.format(seed=seed))
    out.mkdir(parents=True, exist_ok=True)
    for condition in real_calls.CONDITIONS:
        name = f"windows_ssl_aasist_{condition}.csv"
        avg = average_windows(
            pd.read_csv(Path(V2_CALLS.format(seed=seed)) / name),
            pd.read_csv(Path(SPONT_CALLS.format(seed=seed)) / name),
        )
        avg.to_csv(out / name, index=False)
    saved = dict(real_calls.MODELS)
    real_calls.MODELS.clear()
    real_calls.MODELS["ssl_aasist"] = THRESHOLD.format(seed=seed)
    try:
        result = real_calls.report(PROJ / "data/real_calls/scored", out)
    finally:
        real_calls.MODELS.clear()
        real_calls.MODELS.update(saved)
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return out


def conditions(avg: dict, v2: dict) -> dict[str, bool | None]:
    """The five conditions for one seed pair; None where a quantity is missing."""
    need = ("hiacc", "read_new_speakers", "read_control", "genuine_live", "genuine_replay")
    need += ("xtts", "rvc", *KEPT)
    if any(avg.get(k) is None for k in need) or any(v2.get(k) is None for k in ("rvc", *KEPT)):
        return dict.fromkeys(
            ("1_genuine", "2_xtts", "3_rvc_gain", "4_new_tools_kept", "5_tortoise_kept")
        )
    return {
        "1_genuine": avg["hiacc"] >= v2rule.SPONTANEOUS_MIN
        and min(avg["read_new_speakers"], avg["read_control"]) >= v2rule.READ_MIN
        and avg["genuine_live"] >= v2rule.LIVE_MIN
        and avg["genuine_replay"] >= v2rule.REPLAY_MIN,
        "2_xtts": avg["xtts"] >= v2rule.XTTS_MIN,
        "3_rvc_gain": avg["rvc"] - v2["rvc"] >= RVC_GAIN_MIN,
        "4_new_tools_kept": all(avg[k] >= v2[k] - KEEP_DROP_MAX for k in ("indicf5", "chatterbox")),
        "5_tortoise_kept": all(
            avg[k] >= v2[k] - KEEP_DROP_MAX for k in ("tortoise_new", "tortoise_cm04")
        ),
    }


def decide(measures: dict[str, dict]) -> dict:
    """``measures`` holds ``ssl_vcv2avg_s<seed>`` and ``ssl_vcv2_s<seed>`` for both seeds."""
    conds = {
        seed: conditions(measures.get(model(seed), {}), measures.get(f"ssl_vcv2_s{seed}", {}))
        for seed in SEEDS
    }
    values = [v for c in conds.values() for v in c.values()]
    if any(v is None for v in values):
        answer = "incomplete"
    else:
        answer = "recommended" if all(values) else "not recommended"
    return {"conditions": {str(s): c for s, c in conds.items()}, "answer": answer}


def main() -> None:
    """CLI: ``build`` the averaged files, then ``judge`` them by the rule."""
    import argparse

    from src.data import generator_study as gs

    parser = argparse.ArgumentParser(description="v2 averaged with the spontaneous recipe")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    j = sub.add_parser("judge")
    j.add_argument("--pools", default=f"data/manifests/{gs.POOLS_OUT}")
    j.add_argument("--out", default="experiments/results/rvc_average.json")
    args = parser.parse_args()
    if args.cmd == "build":
        for seed in SEEDS:
            print(seed, "clips:", ", ".join(build_clips(seed)))
            print(seed, "operating point:", build_dev(seed))
            print(seed, "calls:", build_calls(seed))
        return
    exclude = frozenset(gs.train_speakers(args.pools))
    names = [m for s in SEEDS for m in (model(s), f"ssl_vcv2_s{s}")]
    measures = {m: v2rule.measures(m, exclude) for m in names}
    report = decide(measures)
    report["measures"] = measures
    report["excluded_speakers"] = len(exclude)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for seed, c in report["conditions"].items():
        print(seed, c)
    print("ANSWER:", report["answer"])


if __name__ == "__main__":
    main()
