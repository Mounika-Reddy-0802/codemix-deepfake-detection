"""Every number the demo pages show, read from a committed result file with its source.

The final review's first rule: no number reaches a slide, a page or a web view unless
it is read from a results file in the repository. The demo pages therefore never hold
a figure in their HTML. They ask the server, and the server answers from here: each
value is returned as ``{"value": ..., "source": "<file>:<field>"}``, so a page can show
where every number came from, and ``tests/test_results_view.py`` can check each one
against its file.

Nothing is computed here that a results file does not already hold, except counts and
means that are plain sums over a committed per-call or per-recording table.
"""

from __future__ import annotations

import csv
import json
import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
REAL_CALLS = RESULTS / "real_calls_v1"

#: Call kinds in the real-call table, in the order the review shows them.
CALL_ROWS = (
    ("genuine_live", "none", "Genuine, live"),
    ("genuine_replay", "none", "Genuine, replayed"),
    ("clone_replay", "xtts_v2", "XTTS-v2 clones, replayed"),
    ("clone_replay", "chatterbox", "Chatterbox clones, replayed"),
)
#: The call chain, stage key -> label; the last stage is the genuine live calls themselves.
CHAIN_STAGES = (
    ("direct", "Direct recording"),
    ("g711", "Simulated G.711 line"),
    ("amr_nb", "Real AMR-NB codec, 12.2 kbit/s"),
    ("amr_nb_rec", "AMR-NB, then the call recorder"),
    ("amr_wb_rec", "AMR-WB (4G), then the recorder"),
    ("real_calls", "Genuine live calls as delivered"),
)
MODELS = {
    "deployed": {
        "label": "Deployed adapter",
        "colour": "#eb6834",
        "threshold_file": "configs/threshold.yaml",
    },
    "final": {
        "label": "Final recipe",
        "colour": "#1baf7a",
        "threshold_file": "configs/threshold_final.yaml",
    },
}


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


@lru_cache(maxsize=64)
def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def cite(value, path: Path, field: str) -> dict:
    """A number and where it was read from."""
    return {"value": value, "source": f"{_rel(path)}:{field}"}


def _threshold(file: str) -> dict:
    import yaml

    path = ROOT / file
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {
        "threshold": cite(float(raw["threshold"]), path, "threshold"),
        "caution_after": cite(int(raw.get("suspicious_after", 2)), path, "suspicious_after"),
        "likely_after": cite(int(raw.get("fake_after", 4)), path, "fake_after"),
    }


# --------------------------------------------------------------------------- #
# Headline numbers: the offline results the slides and pages already use
# --------------------------------------------------------------------------- #
def _systems_cell(system: str, variant: str, condition: str, set_: str) -> tuple[dict, int]:
    cells = _json(RESULTS / "systems_s1_s2_s3.json")["cells"]
    for i, c in enumerate(cells):
        if (c["system"], c["variant"], c["condition"], c["set"]) == (
            system,
            variant,
            condition,
            set_,
        ):
            return c, i
    raise KeyError(f"{system}/{variant}/{condition}/{set_} not in systems_s1_s2_s3.json")


def _floor_summary(detector: str, test_set: str) -> tuple[dict, int]:
    rows = _json(RESULTS / "floor_final_v1.json")["summary"]
    for i, r in enumerate(rows):
        if r["detector"] == detector and r["test_set"] == test_set:
            return r, i
    raise KeyError(f"{detector}/{test_set} not in floor_final_v1.json")


def _eer_vs_floor(detector: str, test_set: str) -> dict:
    row, i = _floor_summary(detector, test_set)
    path = RESULTS / "floor_final_v1.json"
    return {
        "eer": cite(round(100 * row["detector_eer"], 1), path, f"summary[{i}].detector_eer"),
        "floor": cite(round(100 * row["floor_eer"], 1), path, f"summary[{i}].floor_eer"),
        "seeds": cite(row["seeds"], path, f"summary[{i}].seeds"),
        "beats_floor": cite(row["beats_floor"], path, f"summary[{i}].beats_floor"),
    }


def _s1_false_alarm() -> dict:
    path = RESULTS / "classification_metrics.json"
    for i, run in enumerate(_json(path)["runs"]):
        if run["run"] == "stage1" and run["set"] == "eval_pool":
            rate = run["deployed"]["false_alarm_rate"]
            return cite(round(100 * rate, 1), path, f"runs[{i}].deployed.false_alarm_rate")
    raise KeyError("S1 eval_pool run missing from classification_metrics.json")


def _s1_false_alarm_counts() -> dict:
    """How the headline rate was measured: genuine clips, and how many were called fake."""
    path = RESULTS / "classification_metrics.json"
    for i, run in enumerate(_json(path)["runs"]):
        if run["run"] == "stage1" and run["set"] == "eval_pool":
            return {
                "genuine_clips": cite(run["bonafide"], path, f"runs[{i}].bonafide"),
                "called_fake": cite(
                    run["deployed"]["confusion"]["fp"], path, f"runs[{i}].deployed.confusion.fp"
                ),
                "threshold": cite(
                    round(run["deployed"]["threshold"], 3), path, f"runs[{i}].deployed.threshold"
                ),
            }
    raise KeyError("S1 eval_pool run missing from classification_metrics.json")


def _english_line(model: str = "deployed_s2") -> dict:
    """English speech over the phone line: the same English clips, clean and through G.711."""
    path = RESULTS / "english_channel_v1" / "summary.json"
    cells = _json(path)["cells"][model]
    return {
        "clean_eer": cite(cells["clean"]["eer"], path, f"cells.{model}.clean.eer"),
        "line_eer": cite(cells["channel"]["eer"], path, f"cells.{model}.channel.eer"),
        "clips": cite(cells["channel"]["clips"], path, f"cells.{model}.channel.clips"),
    }


def _calls_60s(file: str, key: str) -> dict:
    path = RESULTS / file
    v = _json(path)["validation"][key]["calls_60s"]
    field = f"validation.{key}.calls_60s"
    out = {
        "real_calls": cite(v["real_calls"], path, f"{field}.real_calls"),
        "real_alerted_pct": cite(
            v["real_calls_alerted_pct"], path, f"{field}.real_calls_alerted_pct"
        ),
    }
    if v.get("fake_calls"):
        out["fake_calls"] = cite(v["fake_calls"], path, f"{field}.fake_calls")
        out["fake_alerted_pct"] = cite(
            v["fake_calls_alerted_pct"], path, f"{field}.fake_calls_alerted_pct"
        )
        out["fake_likely_pct"] = cite(
            v["fake_calls_likely_fake_pct"], path, f"{field}.fake_calls_likely_fake_pct"
        )
    return out


def headline() -> dict:
    """The offline results, grouped by system, each with its source."""
    systems = RESULTS / "systems_s1_s2_s3.json"
    s1_en, i_en = _systems_cell("S1 English-only", "asvspoof", "clean", "english")
    s1_line, i_line = _systems_cell("S1 English-only", "asvspoof", "channel", "eval_pool")
    # the deployed adapter: S2 LoRA trained on XTTS + RVC through the phone line
    s2_en, i_s2en = _systems_cell("S2 LoRA", "xtts+rvc", "channel", "english")
    seeds = RESULTS / "seeds_v1.json"
    dep = _json(seeds)["deployed"]["sets"]
    timing = RESULTS / "encoder_spontaneous_v1" / "timing.json"
    t = _json(timing)
    return {
        "s1": {
            "english_eer": cite(s1_en["eer"], systems, f"cells[{i_en}].eer"),
            "codemix_line_eer": cite(s1_line["eer"], systems, f"cells[{i_line}].eer"),
            "genuine_called_fake_pct": _s1_false_alarm(),
            **_s1_false_alarm_counts(),
        },
        "english_line_deployed": _english_line(),
        "deployed": {
            "line_eer_mean": cite(
                round(dep["eval_pool"]["eer"]["mean"], 2), seeds, "deployed.sets.eval_pool.eer.mean"
            ),
            "line_eer_sd": cite(
                round(dep["eval_pool"]["eer"]["sd"], 2), seeds, "deployed.sets.eval_pool.eer.sd"
            ),
            "line_seeds": cite(
                dep["eval_pool"]["eer"]["n"], seeds, "deployed.sets.eval_pool.eer.n"
            ),
            "english_eer": cite(s2_en["eer"], systems, f"cells[{i_s2en}].eer"),
            "tortoise_eer_mean": cite(
                round(dep["cm04"]["eer"]["mean"], 2), seeds, "deployed.sets.cm04.eer.mean"
            ),
            "tortoise_floor": cite(
                round(dep["cm04"]["floor"], 2), seeds, "deployed.sets.cm04.floor"
            ),
            "tortoise_seeds_beating_floor": cite(
                dep["cm04"]["seeds_beating_floor"], seeds, "deployed.sets.cm04.seeds_beating_floor"
            ),
        },
        "final": {
            "new_tortoise": _eer_vs_floor("adapter", "new_tortoise"),
            "new_tortoise_amr": _eer_vs_floor("adapter", "new_tortoise_amr"),
            "indicf5": _eer_vs_floor("adapter", "cm07"),
            "chatterbox": _eer_vs_floor("adapter", "cm09"),
            "seen_tool": _eer_vs_floor("adapter", "eval_pool"),
        },
        "ssl": {
            "new_tortoise_amr": _eer_vs_floor("SSL-AASIST", "new_tortoise_amr"),
            "chatterbox": _eer_vs_floor("SSL-AASIST", "cm09"),
            "indicf5": _eer_vs_floor("SSL-AASIST", "cm07"),
            "indicf5_amr": _eer_vs_floor("SSL-AASIST", "cm07_amr"),
            "cpu_ms_per_hop": cite(t["ssl_aasist_cpu_ms"], timing, "ssl_aasist_cpu_ms"),
            "cpu_threads": cite(t["cpu_threads"], timing, "cpu_threads"),
            "wav2vec2_cpu_ms_per_hop": cite(
                t["wav2vec2_base_cpu_ms"], timing, "wav2vec2_base_cpu_ms"
            ),
        },
        "calls_60s_final": {
            "genuine_read": _calls_60s("threshold_calibration_final.json", "new_speakers_windows"),
            "tortoise": _calls_60s("threshold_calibration_final.json", "cm04_windows"),
            "indicf5": _calls_60s("threshold_calibration_final.json", "cm07_windows"),
            "chatterbox": _calls_60s("threshold_calibration_final.json", "cm09_windows"),
            "hiacc": _calls_60s("threshold_calibration_final.json", "hiacc_windows"),
        },
        "calls_60s_deployed": {
            "hiacc": _calls_60s("threshold_calibration_deployed_w14.json", "hiacc_windows"),
        },
        # SSL-AASIST at its own operating point, chosen on the same dev clips (#63)
        "calls_60s_ssl": {
            "tortoise": _calls_60s("threshold_calibration_ssl_aasist.json", "cm04_windows"),
            "indicf5": _calls_60s("threshold_calibration_ssl_aasist.json", "cm07_windows"),
            "chatterbox": _calls_60s("threshold_calibration_ssl_aasist.json", "cm09_windows"),
            "hiacc": _calls_60s("threshold_calibration_ssl_aasist.json", "hiacc_windows"),
        },
        "ladders": {
            "deployed": _threshold("configs/threshold.yaml"),
            "final": _threshold("configs/threshold_final.yaml"),
            "ssl": _threshold("configs/threshold_ssl_aasist.yaml"),
        },
    }


# --------------------------------------------------------------------------- #
# The 30 real mobile calls
# --------------------------------------------------------------------------- #
def _call_rows(model: str, version: str = "g711") -> list[dict]:
    return _csv(REAL_CALLS / f"calls_{model}_{version}.csv")


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    v = sorted(values)
    mid = len(v) // 2
    return v[mid] if len(v) % 2 else (v[mid - 1] + v[mid]) / 2


def real_call_table(
    version: str = "g711", models: tuple[str, ...] = ("deployed", "final")
) -> list[dict]:
    """The four rows of the real-call table, counted from the per-call CSVs.

    Per model: calls warned (caution or worse), calls taken to "likely cloned", and the
    median time of the first caution over the warned calls. The summary JSON's
    percentages are returned beside the counts, so a page and a test can see that the
    two agree. ``models`` may add ``ssl_aasist`` (the larger encoder, #63).
    """
    summary = _json(REAL_CALLS / "summary.json")["models"]
    rows = []
    for kind, tool, label in CALL_ROWS:
        out = {"kind": kind, "tool": tool, "label": label}
        for model in models:
            path = REAL_CALLS / f"calls_{model}_{version}.csv"
            calls = [
                r for r in _call_rows(model, version) if r["kind"] == kind and r["tool"] == tool
            ]
            warned = [r for r in calls if _truthy(r["caution"])]
            likely = [r for r in calls if _truthy(r["likely_cloned"])]
            firsts = [float(r["first_caution_s"]) for r in warned if r["first_caution_s"]]
            entry = next(
                (
                    s
                    for s in summary[f"{model}_{version}"]["summary"]
                    if s["kind"] == kind and s["tool"] == tool
                ),
                {},
            )
            src = _rel(path)
            out[model] = {
                "calls": {"value": len(calls), "source": f"{src}:kind={kind},tool={tool}"},
                "warned": {"value": len(warned), "source": f"{src}:caution"},
                "likely": {"value": len(likely), "source": f"{src}:likely_cloned"},
                "median_first_caution_s": {
                    "value": _median(firsts),
                    "source": f"{src}:first_caution_s",
                },
                "warned_pct": {
                    "value": entry.get("caution_pct"),
                    "source": f"{_rel(REAL_CALLS / 'summary.json')}:models.{model}_{version}.summary.caution_pct",
                },
                "likely_pct": {
                    "value": entry.get("likely_cloned_pct"),
                    "source": f"{_rel(REAL_CALLS / 'summary.json')}:models.{model}_{version}.summary.likely_cloned_pct",
                },
            }
        rows.append(out)
    return rows


def real_call_speakers(version: str = "g711") -> list[dict]:
    """Per speaker and model: genuine live calls warned and taken to likely cloned."""
    out = []
    for model in ("deployed", "final"):
        calls = [r for r in _call_rows(model, version) if r["kind"] == "genuine_live"]
        for speaker in sorted({r["speaker"] for r in calls}):
            mine = [r for r in calls if r["speaker"] == speaker]
            out.append(
                {
                    "model": model,
                    "speaker": speaker,
                    "calls": len(mine),
                    "warned": sum(_truthy(r["caution"]) for r in mine),
                    "likely": sum(_truthy(r["likely_cloned"]) for r in mine),
                }
            )
    return out


def call_chain() -> dict:
    """Mean P(genuine) per model at each stage of a phone call (the call-chain figure)."""
    chain = _csv(REAL_CALLS / "codec_chain.csv")
    stages = []
    for key, label in CHAIN_STAGES:
        entry = {"key": key, "label": label}
        for model in MODELS:
            if key == "real_calls":
                value, n, src = _live_call_mean(model)
            else:
                values = [
                    float(r["mean_p_genuine"])
                    for r in chain
                    if r["model"] == model and r["version"] == key
                ]
                value = sum(values) / len(values) if values else None
                n = len(values)
                src = f"{_rel(REAL_CALLS / 'codec_chain.csv')}:version={key},model={model}"
            entry[model] = {
                "value": None if value is None else round(value, 4),
                "n": n,
                "source": src,
            }
        stages.append(entry)
    models = {}
    for key, spec in MODELS.items():
        models[key] = {**spec, **_threshold(spec["threshold_file"])}
    return {"stages": stages, "models": models}


def _live_call_mean(model: str) -> tuple[float | None, int, str]:
    """Mean over the genuine live calls of each call's mean window P(genuine)."""
    path = REAL_CALLS / f"windows_{model}_g711.csv"
    plan = {r["call_id"]: r for r in _csv(ROOT / "docs" / "real_calls" / "call_plan.csv")}
    live = {cid for cid, r in plan.items() if r["kind"] == "genuine_live"}
    per_call: dict[str, list[float]] = {}
    for r in _csv(path):
        if not r.get("score"):
            continue
        call = Path(str(r["filepath"]).replace("\\", "/")).stem
        if call in live:
            per_call.setdefault(call, []).append(float(r["score"]))
    means = [sum(v) / len(v) for v in per_call.values() if v]
    value = sum(means) / len(means) if means else None
    return value, len(means), f"{_rel(path)}:score over genuine_live calls"


def real_calls() -> dict:
    """Everything the real-network scene shows."""
    summary = _json(REAL_CALLS / "summary.json")
    src = _rel(REAL_CALLS / "summary.json")
    final = summary["models"]["final_g711"]
    return {
        "calls_recorded": {"value": summary["calls_recorded"], "source": f"{src}:calls_recorded"},
        "table": real_call_table(),
        "speakers": real_call_speakers(),
        "window_eer_final": [
            {**row, "source": f"{src}:models.final_g711.window_eer"} for row in final["window_eer"]
        ],
        "chain": call_chain(),
        "note": "Recorded on the receiving handset, scored offline through this same path, "
        "at the thresholds fixed before the calls.",
    }


# --------------------------------------------------------------------------- #
# Scene 6: what it cannot do yet, and what was done about each limit
# --------------------------------------------------------------------------- #
#: Track A's overnight results: (doc, data files it may ship with), looked up at request
#: time so the limits scene picks them up the moment they are committed.
TONIGHT = {
    "spontaneous": ("docs/results/spontaneous_tradeoff_v1.md", "spontaneous_tradeoff_v1"),
    "late": ("docs/results/early_warning_v1.md", "early_warning_v1"),
}
#: The larger encoder's answer landed as new columns of the real-call study (#63).
SSL_CALLS = REAL_CALLS / "calls_ssl_aasist_g711.csv"
SSL_CHECKPOINT = RESULTS / "ssl_checkpoint_v1" / "summary.json"
SSL_CALIBRATION = RESULTS / "threshold_calibration_ssl_aasist.json"


EARLY = RESULTS / "early_warning_v1" / "summary.json"


def _early_row(model: str, gate: str, kind: str, tool: str) -> tuple[dict, str]:
    rows = _json(EARLY)["models"][model][gate]
    i = next(n for n, r in enumerate(rows) if r["kind"] == kind and r["tool"] == tool)
    return rows[i], f"models.{model}.{gate}[{i}]"


def _early_result() -> dict:
    """The stricter early-call gate on the final recipe: without and with it, as #63's
    follow-up measured it on the real calls (set on dev only, chooses nothing)."""
    out = {}
    for key, model, kind, tool, field in (
        ("genuine_warned_g711", "final_g711", "genuine_live", "none", "warned"),
        ("genuine_warned_recorded", "final_recorded", "genuine_live", "none", "warned"),
        ("xtts_first_caution_s", "final_g711", "clone_replay", "xtts_v2", "median_first_caution_s"),
    ):
        pair = {}
        for gate in ("without_gate", "with_gate"):
            row, at = _early_row(model, gate, kind, tool)
            pair[gate] = cite(row[field], EARLY, f"{at}.{field}")
        pair["calls"] = cite(row["calls"], EARLY, f"{at}.calls")
        out[key] = pair
    final = _json(EARLY)["models"]["final_g711"]
    out["early_windows"] = cite(final["early_windows"], EARLY, "models.final_g711.early_windows")
    out["early_windows_gated"] = cite(
        final["early_windows_gated"], EARLY, "models.final_g711.early_windows_gated"
    )
    return out


def _ssl_landed() -> dict:
    """SSL-AASIST on the 30 real calls and the held-out calls, beside the final recipe."""
    files = (SSL_CALLS, SSL_CHECKPOINT, SSL_CALIBRATION)
    out = {
        "doc": "docs/results/real_calls_v1.md",
        "landed": all(p.is_file() for p in files),
        "data": [_rel(p) for p in files if p.is_file()],
    }
    if not out["landed"]:
        return out
    table = {(r["kind"], r["tool"]): r for r in real_call_table(models=("final", "ssl_aasist"))}
    run = _json(SSL_CHECKPOINT)["runs"][0]
    out.update(
        {
            "title": "SSL-AASIST on the 30 real calls, at a threshold chosen on dev",
            "result": {
                "ssl": {
                    "chatterbox": table[("clone_replay", "chatterbox")]["ssl_aasist"],
                    "xtts": table[("clone_replay", "xtts_v2")]["ssl_aasist"],
                    "genuine_live": table[("genuine_live", "none")]["ssl_aasist"],
                },
                "final": {
                    "chatterbox": table[("clone_replay", "chatterbox")]["final"],
                    "xtts": table[("clone_replay", "xtts_v2")]["final"],
                    "genuine_live": table[("genuine_live", "none")]["final"],
                },
                "checkpoint_chatterbox_eer": cite(
                    round(100 * run["cm09"], 2), SSL_CHECKPOINT, "runs[0].cm09"
                ),
                "ladder": _threshold("configs/threshold_ssl_aasist.yaml"),
            },
            "caveat": "One checkpoint, and offline the weakest of the six SSL-AASIST trainings on "
            "Chatterbox, so it is the next candidate, not a replacement: a second checkpoint must agree.",
        }
    )
    return out


def _landed(key: str) -> dict:
    doc, stem = TONIGHT[key]
    path = ROOT / doc
    results = ROOT / "experiments" / "results"
    data = sorted(
        {p for p in results.rglob(f"{stem}*") if p.is_file()}
        | {p for p in (results / stem).glob("*") if p.is_file()}  # a results folder of its own
        | {p for p in (ROOT / "docs" / "results").glob(f"{stem}*.csv")}
    )
    out = {"doc": doc, "landed": path.is_file(), "data": [_rel(p) for p in data]}
    if path.is_file():
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
        title = next((ln.lstrip("# ") for ln in lines if ln.startswith("#")), doc)
        # the first prose paragraph, joined across its wrapped lines
        start = next(
            (
                i
                for i, ln in enumerate(lines)
                if ln and not ln.startswith(("#", "|", "-", "*", ">"))
            ),
            None,
        )
        body = []
        for ln in lines[start:] if start is not None else []:
            if not ln:
                break
            body.append(ln)
        paragraph = " ".join(body)
        # shown as plain text on the page: links keep their words, code and emphasis marks go
        paragraph = re.sub(r"\[([^\]]+)\]\([^)]*\)", lambda m: m.group(1), paragraph)
        paragraph = re.sub(r"[`*]", "", paragraph)
        if len(paragraph) > 360:
            paragraph = paragraph[:360].rsplit(" ", 1)[0] + " …"
        out.update({"title": title, "first_paragraph": paragraph})
    return out


def tradeoff_curve() -> dict | None:
    """The spontaneous-speech trade-off as chart rows, once Track A has committed its CSV.

    Columns are matched by name (a threshold column, a genuine-spontaneous warned share
    and a clone caught share), so a renamed column fails loudly in the test, not
    silently on the projector.
    """
    files = [p for p in _landed("spontaneous")["data"] if p.endswith(".csv")]
    if not files:
        return None
    # the sweep itself, not the per-call table it was counted from
    path = ROOT / next((f for f in files if f.endswith("/tradeoff.csv")), files[0])
    rows = _csv(path)
    if not rows:
        return None
    cols = list(rows[0])

    def pick(*needles: str, avoid: tuple[str, ...] = ()) -> str | None:
        for col in cols:
            low = col.lower()
            if all(n in low for n in needles) and not any(a in low for a in avoid):
                return col
        return None

    threshold = pick("threshold")
    genuine = pick("hiacc") or pick("spontaneous") or pick("genuine", avoid=("clone",))
    clones = pick("clone", avoid=("unseen",)) or pick("caught") or pick("spoof")
    unseen = pick("unseen")  # the tools no model trained on: the honest cost of a lower bar
    read = pick("read_genuine")
    if not (threshold and genuine and clones):
        return {
            "source": _rel(path),
            "columns": cols,
            "points": [],
            "error": "unrecognised columns",
        }
    points = []
    for r in rows:
        try:
            point = {
                "threshold": float(r[threshold]),
                "genuine_warned": float(r[genuine]),
                "clones_caught": float(r[clones]),
            }
            if unseen:
                point["unseen_caught"] = float(r[unseen])
            if read:
                point["read_genuine_warned"] = float(r[read])
            points.append(point)
        except (TypeError, ValueError):
            continue
    live = _threshold("configs/threshold_final.yaml")["threshold"]
    used = [c for c in (threshold, genuine, clones, unseen, read) if c]
    return {"source": _rel(path), "columns": used, "points": points, "live_threshold": live}


def limits() -> dict:
    """The four limits with their numbers, and Track A's overnight answer to each."""
    h = headline()
    calls = {(r["kind"], r["tool"]): r for r in real_call_table()}
    chatterbox_real = calls[("clone_replay", "chatterbox")]["final"]
    xtts_real = calls[("clone_replay", "xtts_v2")]["final"]
    likely_times = sorted(
        float(r["first_likely_cloned_s"])
        for r in _call_rows("final")
        if r["kind"] == "clone_replay" and r["tool"] == "xtts_v2" and r["first_likely_cloned_s"]
    )
    calls_src = _rel(REAL_CALLS / "calls_final_g711.csv")
    return {
        "limits": [
            {
                "key": "spontaneous",
                "title": "Spontaneous speech",
                "said": "Every genuine spontaneous Hindi-English call (HiACC, 60 s) is warned by both models. "
                "The training data is read speech; free conversation sounds different to the detector.",
                "numbers": {
                    "final_genuine_warned_pct": h["calls_60s_final"]["hiacc"]["real_alerted_pct"],
                    "final_calls": h["calls_60s_final"]["hiacc"]["real_calls"],
                    "deployed_genuine_warned_pct": h["calls_60s_deployed"]["hiacc"][
                        "real_alerted_pct"
                    ],
                },
                "tonight": {
                    "what": "A threshold sweep on the final recipe's existing scores that chooses nothing: "
                    "the share of genuine spontaneous calls warned against the share of clone calls caught.",
                    **_landed("spontaneous"),
                    "ssl": h["calls_60s_ssl"]["hiacc"],
                },
                "fix": "Spontaneous genuine speech in training. No threshold is chosen on HiACC.",
            },
            {
                "key": "chatterbox",
                "title": "Chatterbox, an unseen cloning tool",
                "said": "The final recipe is weakest on Chatterbox, a tool none of the models saw.",
                "numbers": {
                    "final_eer": h["final"]["chatterbox"]["eer"],
                    "final_floor": h["final"]["chatterbox"]["floor"],
                    "final_seeds": h["final"]["chatterbox"]["seeds"],
                    "final_calls_caught_pct": h["calls_60s_final"]["chatterbox"][
                        "fake_alerted_pct"
                    ],
                    "real_calls_likely": chatterbox_real["likely"],
                    "real_calls": chatterbox_real["calls"],
                    "ssl_eer": h["ssl"]["chatterbox"]["eer"],
                    "ssl_seeds": h["ssl"]["chatterbox"]["seeds"],
                },
                "tonight": {
                    "what": "The larger encoder (SSL-AASIST) trained on the final-recipe data, then the "
                    "30 real calls scored with it at a threshold fixed on development calls.",
                    **_ssl_landed(),
                    "held_out": h["calls_60s_ssl"]["chatterbox"],
                },
                "fix": "A larger encoder: SSL-AASIST already reaches a far lower Chatterbox EER offline.",
            },
            {
                "key": "late",
                "title": "Late warnings on real calls",
                "said": "On the real XTTS-v2 clone calls the first caution comes at a median of "
                "several seconds in, and a strong warning can take most of the call.",
                "numbers": {
                    "xtts_median_first_caution_s": xtts_real["median_first_caution_s"],
                    "xtts_likely_times_s": {
                        "value": likely_times,
                        "source": f"{calls_src}:first_likely_cloned_s",
                    },
                    "xtts_caught": xtts_real["likely"],
                    "xtts_calls": xtts_real["calls"],
                },
                "tonight": {
                    "what": "A stricter speech gate for the first seconds of a call, chosen on simulated "
                    "development calls only, then time-to-warn measured on the 30 real calls.",
                    **_landed("late"),
                    **(
                        {
                            "result": _early_result(),
                            "caveat": "The real calls are far quieter than the normalised dev clips, "
                            "so the gate also silences speech: it needs real development calls.",
                        }
                        if EARLY.is_file()
                        else {}
                    ),
                },
                "fix": "An early-call speech gate set relative to the caller's own level.",
            },
            {
                "key": "carrier",
                "title": "Not yet on a live phone number",
                "said": "The phone-carrier path (Twilio media streams, receiver-only alerts) is implemented "
                "and unit-tested; it has not yet run on live numbers.",
                "numbers": {},
                "tonight": {
                    "what": "Parked on purpose: the real-network evidence comes from the 30 recorded calls "
                    "scored through the same path.",
                    "doc": "live_call/README.md",
                    "landed": True,
                    "data": [],
                },
                "fix": "A carrier trial on live numbers, with consent, after the review.",
            },
        ],
        "tradeoff": tradeoff_curve(),
    }


__all__ = [
    "call_chain",
    "cite",
    "headline",
    "limits",
    "real_call_table",
    "real_calls",
    "tradeoff_curve",
]
