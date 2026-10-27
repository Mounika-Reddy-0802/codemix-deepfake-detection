"""Score the real phone-call test (week 14) through the live verdict path.

The call plan (``CALL_PLAN``) has 30 calls between two fixed phones over the mobile
network, recorded on the receiving phone while the receiver stays muted, so each
recording holds only what reached the receiver:

- **genuine_live** (12): a team member speaks into the calling phone, reading script A or
  B, or talking freely;
- **genuine_replay** (6): the same member's own reading, played from a laptop speaker
  into the calling phone;
- **clone_replay** (12): clones of that member's voice saying the same script (XTTS-v2,
  the seen tool, and Chatterbox, the hardest unseen tool), played the same way.

Genuine and cloned replays share the loudspeaker path, so comparing them isolates the
synthesis; the live genuine calls measure what a real caller meets. Each recording is
scored twice: as recorded, and after a G.711 round trip at 8 kHz, which is what the live
system receives from its telephone carrier. Scoring uses ``src.inference.calibrate``,
the exact streaming path, and every call is replayed through ``VerdictEngine`` with each
model's frozen operating point. Nothing here chooses a threshold. Before scoring, every
replay recording is matched back to the file played into it (``check``), and the report
carries a diagnostic that treats each call's first window as silence.

Run::

    python -m src.inference.real_calls plan --out docs/real_calls/call_plan.csv
    python -m src.inference.real_calls check --recordings data/real_calls/recordings \\
        --playback data/real_calls/playback --out data/real_calls/windows/replay_check.csv
    python -m src.inference.real_calls prepare --recordings data/real_calls/recordings \\
        --out data/real_calls/scored
    python -m src.inference.real_calls score --audio data/real_calls/scored \\
        --deployed CKPT --final CKPT --out experiments/results/real_calls_v1
    python -m src.inference.real_calls report --audio data/real_calls/scored \\
        --windows experiments/results/real_calls_v1 --md experiments/results/real_calls_v1/report.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from live_call.verdict_engine import EngineConfig, State, VerdictEngine, load_config
from src.inference.streaming import EMA_ALPHA, SAMPLE_RATE
from src.utils.paths import DATA_ROOT_TOKEN

CONDITIONS = ("recorded", "g711")
#: Model name -> its frozen live operating point. SSL-AASIST's is chosen on the final
#: recipe's dev windows by the same rule (``src.inference.calibrate``), never on these calls.
MODELS = {
    "deployed": "configs/threshold.yaml",
    "final": "configs/threshold_final.yaml",
    "ssl_aasist": "configs/threshold_ssl_aasist.yaml",
}
#: Model name -> detector kind for ``src.inference.predict.load_score_fn``.
MODEL_KINDS = {"deployed": "adapter", "final": "adapter", "ssl_aasist": "ssl_aasist"}
SOURCE = "realcall"
NARROWBAND_SR = 8_000

PLAN_COLUMNS = ["call_id", "speaker", "kind", "tool", "script", "playback", "label"]


def _plan() -> list[dict]:
    rows: list[tuple[str, str, str, str]] = []
    for speaker in ("P1", "P2", "P3"):
        rows += [(speaker, "genuine_live", "none", s) for s in ("A", "B", "free", "free")]
    for speaker in ("P1", "P2", "P3"):
        rows += [(speaker, "genuine_replay", "none", s) for s in ("A", "B")]
    for tool in ("xtts_v2", "chatterbox"):
        for speaker in ("P1", "P2", "P3"):
            rows += [(speaker, "clone_replay", tool, s) for s in ("A", "B")]
    out = []
    for i, (speaker, kind, tool, script) in enumerate(rows, 1):
        playback = ""
        if kind == "genuine_replay":
            playback = f"genuine_{speaker}_{script}.wav"
        elif kind == "clone_replay":
            playback = f"{tool}_{speaker}_{script}.wav"
        label = "spoof" if kind == "clone_replay" else "bonafide"
        out.append(
            dict(
                zip(
                    PLAN_COLUMNS,
                    (f"C{i:02d}", speaker, kind, tool, script, playback, label),
                    strict=True,
                )
            )
        )
    return out


CALL_PLAN = pd.DataFrame(_plan(), columns=PLAN_COLUMNS)


def find_recordings(folder: Path, plan: pd.DataFrame = CALL_PLAN) -> dict[str, Path]:
    """Call id -> its recording (``C01.m4a``, any extension); calls without one are left out."""
    found: dict[str, Path] = {}
    for call_id in plan["call_id"]:
        matches = sorted(p for p in folder.glob(f"{call_id}.*") if p.is_file())
        if len(matches) > 1:
            raise ValueError(f"more than one recording for {call_id}: {matches}")
        if matches:
            found[call_id] = matches[0]
    return found


def telephone_leg(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """What the live system receives from its carrier: 8 kHz G.711 mu-law, back at ``sr``."""
    from src.data.channel_sim import apply_g711_ulaw
    from src.utils.audio_utils import resample

    narrow = apply_g711_ulaw(resample(audio, sr, NARROWBAND_SR))
    return resample(narrow, NARROWBAND_SR, sr)


def prepare(recordings: Path, out: Path, plan: pd.DataFrame = CALL_PLAN) -> pd.DataFrame:
    """Decode every recording to 16 kHz mono in both conditions; write one manifest each."""
    from src.utils.audio_utils import load_any, save_wav

    found = find_recordings(recordings, plan)
    rows = []
    for call in plan.itertuples(index=False):
        if call.call_id not in found:
            continue
        audio = load_any(str(found[call.call_id]), SAMPLE_RATE)
        versions = {"recorded": audio, "g711": telephone_leg(audio)}
        for condition, signal in versions.items():
            name = f"{condition}/{call.call_id}.wav"
            (out / condition).mkdir(parents=True, exist_ok=True)
            save_wav(str(out / name), signal, SAMPLE_RATE)
            rows.append(
                {
                    # resolved against the prepare folder, wherever it is on this machine
                    "filepath": f"{DATA_ROOT_TOKEN}/{name}",
                    "label": call.label,
                    "tool": call.tool,
                    "source": SOURCE,
                    "call_id": call.call_id,
                    "condition": condition,
                    "seconds": round(signal.size / SAMPLE_RATE, 1),
                }
            )
    table = pd.DataFrame(rows)
    for condition in CONDITIONS:
        part = table[table["condition"] == condition] if len(table) else table
        part.to_csv(out / f"manifest_{condition}.csv", index=False)
    missing = sorted(set(plan["call_id"]) - set(found))
    print(f"{len(found)} of {len(plan)} calls have a recording; missing: {missing or 'none'}")
    return table


#: Speech-activity frame for the replay check, in seconds.
ACTIVITY_HOP_S = 0.1


def speech_activity(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Smoothed 0/1 speech-activity track, one value per 100 ms.

    A frame is active when its level is within 25 dB of the recording's own loud frames
    (95th percentile), so the track survives a loudspeaker, a codec and the phone's gain
    control, which a waveform or a fine energy envelope does not.
    """
    hop = int(ACTIVITY_HOP_S * sr)
    n = audio.size // hop
    if n == 0:
        return np.zeros(0)
    level = 10 * np.log10(np.mean(audio[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-10)
    active = (level > np.percentile(level, 95) - 25).astype(float)
    return np.convolve(active, np.ones(3) / 3, mode="same")


def overlap_correlation(a: np.ndarray, b: np.ndarray, min_overlap: float = 0.7) -> float:
    """Best Pearson correlation of two tracks over every lag at which they overlap by at
    least ``min_overlap`` of the shorter one; -1 when no lag qualifies."""
    need = max(2, int(min_overlap * min(a.size, b.size)))
    best = -1.0
    for lag in range(-(b.size - need), a.size - need + 1):
        a0, b0 = max(0, lag), max(0, -lag)
        n = min(a.size - a0, b.size - b0)
        x, y = a[a0 : a0 + n], b[b0 : b0 + n]
        if n < need or x.std() < 1e-9 or y.std() < 1e-9:
            continue
        best = max(best, float(np.corrcoef(x, y)[0, 1]))
    return best


def check_replays(recordings: Path, playback: Path, plan: pd.DataFrame = CALL_PLAN) -> pd.DataFrame:
    """Does every replay recording hold the file the plan says was played into it?

    Each recorded replay call is matched against every playback file by the correlation of
    their speech-activity tracks; ``ok`` is true when the planned file is the best match.
    A swapped file or a mislabelled recording would put a clone under a genuine call's id.
    """
    from src.utils.audio_utils import load_any

    tracks = {
        p.name: speech_activity(load_any(str(p), SAMPLE_RATE))
        for p in sorted(playback.glob("*.wav"))
    }
    found = find_recordings(recordings, plan)
    rows = []
    for call in plan[plan["playback"] != ""].itertuples(index=False):
        if call.call_id not in found or call.playback not in tracks:
            continue
        track = speech_activity(load_any(str(found[call.call_id]), SAMPLE_RATE))
        scores = {name: overlap_correlation(track, t) for name, t in tracks.items()}
        others = [r for name, r in scores.items() if name != call.playback]
        best = max(scores, key=scores.get)
        rows.append(
            {
                "call_id": call.call_id,
                "planned": call.playback,
                "planned_r": round(scores[call.playback], 3),
                "best": best,
                "best_other_r": round(max(others), 3) if others else None,
                "ok": best == call.playback,
            }
        )
    columns = ["call_id", "planned", "planned_r", "best", "best_other_r", "ok"]
    return pd.DataFrame(rows, columns=columns)


def call_verdicts(windows: pd.DataFrame, config: EngineConfig) -> pd.DataFrame:
    """Replay each call's windows through the verdict engine; one row per call."""
    rows = []
    for filepath, group in windows.sort_values(["filepath", "window"]).groupby(
        "filepath", sort=False
    ):
        engine = VerdictEngine(config)
        first: dict[State, float] = {}
        for row in group.itertuples(index=False):
            smoothed = None if pd.isna(row.smoothed) else float(row.smoothed)
            for event in engine.update(float(row.end_seconds), smoothed):
                first.setdefault(event.state, float(event.at_seconds))
        engine.end_call()
        scored = group.dropna(subset=["smoothed"])
        peak = engine.summary.peak_state
        rows.append(
            {
                "call_id": Path(str(filepath)).stem,
                "windows": int(len(group)),
                "scored_windows": int(len(scored)),
                "fake_leaning_pct": round(
                    100 * float((scored["smoothed"] < config.threshold).mean()), 1
                )
                if len(scored)
                else None,
                "mean_score": round(float(scored["score"].mean()), 4) if len(scored) else None,
                "lowest_smoothed": round(float(scored["smoothed"].min()), 4)
                if len(scored)
                else None,
                "peak_state": peak.value,
                "caution": peak in (State.SUSPICIOUS, State.LIKELY_FAKE),
                "likely_cloned": peak == State.LIKELY_FAKE,
                "first_caution_s": first.get(State.SUSPICIOUS),
                "first_likely_cloned_s": first.get(State.LIKELY_FAKE),
            }
        )
    return pd.DataFrame(rows)


def wilson_interval(hits: int, n: int, z: float = 1.96) -> list[float]:
    """95% Wilson score interval for a share, in percent; it stays sensible at 0 or n of n."""
    if n == 0:
        return [0.0, 100.0]
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(100 * max(0.0, centre - half), 1), round(100 * min(1.0, centre + half), 1)]


def summarise(calls: pd.DataFrame) -> list[dict]:
    """Share of calls raising a caution and reaching 'likely cloned', per kind and tool.

    With a handful of calls per kind, each share comes with its 95% Wilson interval.
    """
    out = []
    for (kind, tool), group in calls.groupby(["kind", "tool"], sort=True):
        n = int(len(group))
        caution, likely = int(group["caution"].sum()), int(group["likely_cloned"].sum())
        out.append(
            {
                "kind": kind,
                "tool": tool,
                "calls": n,
                "caution_pct": round(100 * caution / n, 1),
                "caution_ci": wilson_interval(caution, n),
                "likely_cloned_pct": round(100 * likely / n, 1),
                "likely_cloned_ci": wilson_interval(likely, n),
                "median_first_caution_s": None
                if group["first_caution_s"].isna().all()
                else float(group["first_caution_s"].median()),
            }
        )
    return out


def by_speaker(calls: pd.DataFrame) -> list[dict]:
    """Calls warned and taken to 'likely cloned', per speaker and kind of call.

    Three speakers are too few for a rate per speaker; this shows whether the warnings on
    genuine calls are spread across the speakers or come from one of them.
    """
    out = []
    for (speaker, kind, tool), group in calls.groupby(["speaker", "kind", "tool"], sort=True):
        out.append(
            {
                "speaker": speaker,
                "kind": kind,
                "tool": tool,
                "calls": int(len(group)),
                "caution": int(group["caution"].sum()),
                "likely_cloned": int(group["likely_cloned"].sum()),
            }
        )
    return out


def window_eer(windows: pd.DataFrame, calls: pd.DataFrame, genuine_kind: str, tool: str) -> dict:
    """Window-level EER (%) of one tool's cloned calls against one kind of genuine call."""
    from src.training.metrics import compute_eer

    kinds = calls.set_index("call_id")[["kind", "tool"]]
    # the call plan, not the scoring manifest, says what each call was
    w = windows.dropna(subset=["score"]).drop(columns=["label", "tool"], errors="ignore")
    w["call_id"] = w["filepath"].map(lambda p: Path(str(p)).stem)
    w = w.join(kinds, on="call_id")
    real = w.loc[w["kind"] == genuine_kind, "score"].to_numpy(float)
    fake = w.loc[(w["kind"] == "clone_replay") & (w["tool"] == tool), "score"].to_numpy(float)
    if real.size == 0 or fake.size == 0:
        return {
            "genuine": genuine_kind,
            "tool": tool,
            "eer_pct": None,
            "windows": [real.size, fake.size],
        }
    scores = np.concatenate([real, fake])
    labels = np.concatenate([np.ones(real.size), np.zeros(fake.size)])
    eer, _ = compute_eer(scores, labels)
    return {
        "genuine": genuine_kind,
        "tool": tool,
        "eer_pct": round(100 * float(eer), 2),
        "windows": [int(real.size), int(fake.size)],
    }


#: Time-in-call bands (seconds, by window end) for :func:`score_by_phase`.
PHASES = ((0.0, 10.0, "0-10 s"), (10.0, 20.0, "10-20 s"), (20.0, float("inf"), "after 20 s"))


def score_by_phase(windows: pd.DataFrame, calls: pd.DataFrame) -> list[dict]:
    """Mean window score per kind of call and time in the call.

    The first window of a call spans the quiet before the first word, which passes the
    silence gate and is scored as speech, so a genuine call can look different in its first
    seconds than later; this shows where in a call the scores move. It describes the calls
    and chooses nothing.
    """
    kinds = calls.set_index("call_id")[["kind", "tool"]]
    w = windows.dropna(subset=["score"]).drop(columns=["label", "tool"], errors="ignore")
    w["call_id"] = w["filepath"].map(lambda p: Path(str(p)).stem)
    w = w.join(kinds, on="call_id", how="inner")
    out = []
    for (kind, tool), group in w.groupby(["kind", "tool"], sort=True):
        row = {"kind": kind, "tool": tool}
        for low, high, label in PHASES:
            band = group[(group["end_seconds"] > low) & (group["end_seconds"] <= high)]
            row[label] = round(float(band["score"].mean()), 3) if len(band) else None
        out.append(row)
    return out


def without_first_window(windows: pd.DataFrame, alpha: float = EMA_ALPHA) -> pd.DataFrame:
    """The same calls with each call's first window treated as silence, re-smoothed.

    A diagnostic for the early false alarms: replayed through the verdict engine, it shows
    how much of each verdict rests on the window that spans the quiet before the first
    word. It is not an operating rule, and nothing is chosen with it.
    """
    out = windows.copy()
    out.loc[out["window"] == 0, ["score", "smoothed"]] = np.nan
    smoothed = pd.Series(np.nan, index=out.index)
    for _, call in out.sort_values(["filepath", "window"]).groupby("filepath", sort=False):
        ema = None
        for idx, score in call["score"].items():
            if pd.isna(score):
                continue  # a skipped window carries no smoothed value, as in the live path
            ema = score if ema is None else alpha * score + (1 - alpha) * ema
            smoothed[idx] = ema
    out["smoothed"] = smoothed
    return out


def score_all(audio: Path, checkpoints: dict[str, str], out: Path, device: str = "cpu") -> None:
    """Window scores for every model and condition, resumable, via the calibration path."""
    from src.inference.calibrate import score_manifest
    from src.inference.predict import load_score_fn

    out.mkdir(parents=True, exist_ok=True)
    for name, checkpoint in checkpoints.items():
        score_fn = load_score_fn(checkpoint, kind=MODEL_KINDS.get(name, "adapter"), device=device)
        for condition in CONDITIONS:
            score_manifest(
                str(audio / f"manifest_{condition}.csv"),
                str(audio),
                str(out / f"windows_{name}_{condition}.csv"),
                score_fn,
            )


def report(audio: Path, windows_dir: Path, plan: pd.DataFrame = CALL_PLAN) -> dict:
    """Per-call verdicts and summaries for every scored model and condition."""
    result: dict = {"calls_recorded": 0, "models": {}}
    seconds = {}
    for condition in CONDITIONS:
        path = audio / f"manifest_{condition}.csv"
        if path.is_file() and path.stat().st_size > 1:
            manifest = pd.read_csv(path)
            seconds.update(dict(zip(manifest["call_id"], manifest["seconds"], strict=True)))
    result["calls_recorded"] = len(seconds)
    for name, threshold_file in MODELS.items():
        if not any((windows_dir / f"windows_{name}_{c}.csv").is_file() for c in CONDITIONS):
            continue  # a model that was not scored has no operating point to read either
        config = load_config(threshold_file)
        for condition in CONDITIONS:
            path = windows_dir / f"windows_{name}_{condition}.csv"
            if not path.is_file():
                continue
            windows = pd.read_csv(path)
            calls = plan.merge(call_verdicts(windows, config), on="call_id", how="inner")
            calls["seconds"] = calls["call_id"].map(seconds)
            calls.to_csv(windows_dir / f"calls_{name}_{condition}.csv", index=False)
            result["models"][f"{name}_{condition}"] = {
                "threshold": config.threshold,
                "ladder": [config.suspicious_after, config.fake_after],
                "summary": summarise(calls),
                "by_speaker": by_speaker(calls),
                "window_eer": [
                    window_eer(windows, calls, genuine, tool)
                    for genuine in ("genuine_replay", "genuine_live")
                    for tool in ("xtts_v2", "chatterbox")
                ],
                "by_phase": score_by_phase(windows, calls),
                "first_window_skipped": summarise(
                    plan.merge(
                        call_verdicts(without_first_window(windows), config),
                        on="call_id",
                        how="inner",
                    )
                ),
            }
    return result


def report_markdown(result: dict) -> str:
    """The results note: one summary table and one EER table per model and condition."""
    out = [
        "# Real phone-call test",
        "",
        "Written by `python -m src.inference.real_calls report`. Calls over a mobile",
        "network, recorded on the receiving phone (protocol, and what happened on the day:",
        "`docs/real_calls/PROTOCOL.md`). *recorded*: as the receiving phone recorded it;",
        "*g711*: after an 8 kHz G.711 round trip, as the live system's carrier delivers it.",
        "Counts are small; read every percentage with its number of calls.",
        "",
        f"Calls with a recording: {result['calls_recorded']}.",
        "",
    ]
    for key, entry in result["models"].items():
        out += [
            f"## {key} (threshold {entry['threshold']:.3f}, ladder {entry['ladder']})",
            "",
            "| Calls | Tool | n | Caution [95% CI] | Likely cloned [95% CI] | Median first caution (s) |",
            "|---|---|---|---|---|---|",
        ]
        for row in entry["summary"]:
            first = row["median_first_caution_s"]
            caution = f"{row['caution_pct']}% [{row['caution_ci'][0]}, {row['caution_ci'][1]}]"
            likely = (
                f"{row['likely_cloned_pct']}% "
                f"[{row['likely_cloned_ci'][0]}, {row['likely_cloned_ci'][1]}]"
            )
            out.append(
                f"| {row['kind']} | {row['tool']} | {row['calls']} | {caution} | {likely} | "
                f"{'-' if first is None else f'{first:.0f}'} |"
            )
        out += [
            "",
            "| Genuine side | Tool | Window EER | Windows (genuine, clone) |",
            "|---|---|---|---|",
        ]
        for row in entry["window_eer"]:
            eer = "-" if row["eer_pct"] is None else f"{row['eer_pct']}%"
            out.append(f"| {row['genuine']} | {row['tool']} | {eer} | {row['windows']} |")
        labels = [label for _, _, label in PHASES]
        out += [
            "",
            "Mean window score (P(genuine)) by time in the call:",
            "",
            "| Calls | Tool | " + " | ".join(labels) + " |",
            "|---|---|" + "---|" * len(labels),
        ]
        for row in entry.get("by_phase", []):
            cells = ["-" if row[label] is None else f"{row[label]:.3f}" for label in labels]
            out.append(f"| {row['kind']} | {row['tool']} | " + " | ".join(cells) + " |")
        if entry.get("by_speaker"):
            out += [
                "",
                "Per speaker (calls warned and taken to 'likely cloned'):",
                "",
                "| Speaker | Calls | Tool | n | Caution | Likely cloned |",
                "|---|---|---|---|---|---|",
            ]
            for row in entry["by_speaker"]:
                out.append(
                    f"| {row['speaker']} | {row['kind']} | {row['tool']} | {row['calls']} | "
                    f"{row['caution']} | {row['likely_cloned']} |"
                )
        if entry.get("first_window_skipped"):
            out += [
                "",
                "Diagnostic, chooses nothing: the same calls with each first window treated as",
                "silence (how much of a verdict rests on the quiet before the first word):",
                "",
                "| Calls | Tool | n | Caution | Likely cloned |",
                "|---|---|---|---|---|",
            ]
            for row in entry["first_window_skipped"]:
                out.append(
                    f"| {row['kind']} | {row['tool']} | {row['calls']} | "
                    f"{row['caution_pct']}% | {row['likely_cloned_pct']}% |"
                )
        out.append("")
    return "\n".join(out)


def main() -> None:
    """CLI: ``plan``, ``prepare``, ``check``, ``score`` or ``report``."""
    import argparse

    parser = argparse.ArgumentParser(description="Real phone-call test through the live path")
    sub = parser.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("plan")
    pl.add_argument("--out", default="docs/real_calls/call_plan.csv")
    pr = sub.add_parser("prepare")
    pr.add_argument("--recordings", required=True, help="folder with C01.*, C02.*, ...")
    pr.add_argument("--out", required=True)
    ck = sub.add_parser("check", help="does each replay recording hold its planned file?")
    ck.add_argument("--recordings", required=True)
    ck.add_argument("--playback", required=True, help="folder with the played .wav files")
    ck.add_argument("--out", required=True, help="CSV, one row per replay call")
    sc = sub.add_parser("score")
    sc.add_argument("--audio", required=True, help="the prepare output folder")
    sc.add_argument("--deployed", help="deployed adapter checkpoint")
    sc.add_argument("--final", help="final-recipe checkpoint")
    sc.add_argument("--ssl", help="SSL-AASIST state dict (final-recipe data)")
    sc.add_argument("--out", required=True)
    sc.add_argument("--device", default="cpu")
    rp = sub.add_parser("report")
    rp.add_argument("--audio", required=True)
    rp.add_argument("--windows", required=True)
    # the generated report; docs/results/real_calls_v1.md is the hand-written write-up
    rp.add_argument("--md", default="experiments/results/real_calls_v1/report.md")
    args = parser.parse_args()

    if args.cmd == "plan":
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        CALL_PLAN.to_csv(args.out, index=False)
        print(f"wrote {args.out} ({len(CALL_PLAN)} calls)")
    elif args.cmd == "prepare":
        prepare(Path(args.recordings), Path(args.out))
    elif args.cmd == "check":
        table = check_replays(Path(args.recordings), Path(args.playback))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)
        print(table.to_string(index=False))
        print(f"{int(table['ok'].sum())} of {len(table)} replay calls hold their planned file")
        if not table["ok"].all():
            raise SystemExit("a replay recording does not match its planned file")
    elif args.cmd == "score":
        given = {"deployed": args.deployed, "final": args.final, "ssl_aasist": args.ssl}
        checkpoints = {name: path for name, path in given.items() if path}
        if not checkpoints:
            parser.error("score needs at least one of --deployed, --final, --ssl")
        score_all(Path(args.audio), checkpoints, Path(args.out), args.device)
    else:
        result = report(Path(args.audio), Path(args.windows))
        (Path(args.windows) / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text(report_markdown(result) + "\n", encoding="utf-8")
        print(report_markdown(result))


if __name__ == "__main__":
    main()
