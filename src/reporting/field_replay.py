"""The field recordings (A4) and the corpus replay calls, scored by the voice check (reported).

Both are scored by the voice check's own path at its frozen operating point
(``src.inference.any_audio_eval score`` with ``configs/voice_check.yaml``, telephone front
end); nothing is chosen on them. Each recording is matched to its plan row by name, whatever
its audio format: a field take ``P2_F01.m4a`` to ``P2_F01`` in
``docs/field_validation/take_plan.csv``, a replay call ``R07.m4a`` to ``R07.wav`` in
``docs/real_calls/corpus_replay_plan.csv``. The label always comes from the plan, never from
the file. A recording that no plan row names stops the run (a typo would otherwise drop it
silently); a plan row with no recording is listed as missing::

    python -m src.reporting.field_replay manifest --field data/field_validation \\
        --replay data/real_calls/corpus_replay --out experiments/field_replay/manifest.csv
    python -m src.inference.any_audio_eval score experiments/field_replay/manifest.csv \\
        --front-end g711 --out experiments/field_replay/scored.csv
    python -m src.reporting.field_replay report experiments/field_replay/scored.csv \\
        --out experiments/results/field_replay.md
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

AUDIO = {".wav", ".m4a", ".mp3", ".aac", ".amr", ".3gp", ".ogg", ".opus", ".flac", ".webm"}
FIELD_PLAN = "docs/field_validation/take_plan.csv"
REPLAY_PLAN = "docs/real_calls/corpus_replay_plan.csv"
COLUMNS = ["filepath", "label", "tool", "set", "split", "kind", "take"]


def recordings(folder: str | Path | None) -> dict[str, Path]:
    """Upper-case file stem -> path, for every audio file under ``folder``; a stem twice stops."""
    out: dict[str, Path] = {}
    if folder is None or not Path(folder).is_dir():
        return out
    for path in sorted(Path(folder).rglob("*")):
        if path.suffix.lower() not in AUDIO:
            continue
        stem = path.stem.upper()
        if stem in out:
            raise ValueError(f"two recordings named {stem}: {out[stem]} and {path}")
        out[stem] = path
    return out


def _match(found: dict[str, Path], plan: pd.DataFrame, key: str, name: str) -> pd.DataFrame:
    ids = plan[key].astype(str).str.upper()
    unknown = sorted(set(found) - set(ids))
    if unknown:
        raise ValueError(f"{name}: recordings not in the plan: {', '.join(unknown)}")
    have = plan[ids.isin(found)].copy()
    have["filepath"] = [str(found[i]).replace("\\", "/") for i in ids[ids.isin(found)]]
    have["missing"] = False
    missing = plan[~ids.isin(found)].copy()
    missing["filepath"], missing["missing"] = "", True
    return pd.concat([have, missing], ignore_index=True)


def manifest(
    field_dir=None,
    replay_dir=None,
    field_plan: str = FIELD_PLAN,
    replay_plan: str = REPLAY_PLAN,
) -> tuple[pd.DataFrame, list[str]]:
    """The scoring manifest (labels from the plans) and the plan rows with no recording."""
    parts, missing = [], []
    field = recordings(field_dir)
    if field:
        plan = pd.read_csv(field_plan)
        m = _match(field, plan, "take_id", "field")
        m["take"], m["set"] = m["take_id"], "field_b" + m["batch"].astype(str)
        parts.append(m)
    replay = recordings(replay_dir)
    if replay:
        plan = pd.read_csv(replay_plan)
        plan["take"] = plan["file"].str.rsplit(".", n=1).str[0]
        m = _match(replay, plan, "take", "replay")
        m["set"] = "corpus_replay"
        parts.append(m)
    if not parts:
        raise ValueError("no recordings found")
    rows = pd.concat(parts, ignore_index=True)
    missing = sorted(rows.loc[rows["missing"], "take"])
    rows = rows[~rows["missing"]].assign(split="test")
    return rows[COLUMNS].reset_index(drop=True), missing


def _row(label: str, d: pd.DataFrame) -> str:
    from src.inference.real_calls import wilson_interval

    want = d["label"].str.lower().eq("bonafide").map({True: "real", False: "cloned"})
    ok = int((d["verdict"] == want).sum())
    short = int((d["verdict"] == "no speech").sum())
    errors = int(d["verdict"].astype(str).str.startswith("error").sum())
    lo, hi = wilson_interval(ok, len(d))
    note = "; ".join(
        s
        for s in (f"{short} too short" if short else "", f"{errors} unreadable" if errors else "")
        if s
    )
    return f"| {label} | {ok} of {len(d)} | {100 * ok / len(d):.1f} | [{lo}, {hi}] | {note} |"


def report(scored: pd.DataFrame, missing: list[str] | None = None) -> str:
    """Judged correctly per set and kind, with 95% Wilson intervals; telephone front end."""
    d = scored[scored["front_end"] == "g711"]
    lines = [
        "# Field recordings and corpus replay calls, scored by the voice check",
        "",
        "The voice check's own scoring at its frozen operating point, telephone front end; "
        "a recording with under 2 s of speech counts as not judged correctly. Brackets: 95% "
        "Wilson interval.",
        "",
        "| Set and kind | Judged correctly | % | 95% interval | Notes |",
        "|---|---|---|---|---|",
    ]
    for name, group in d.groupby("set", sort=True):
        for kind, g in group.groupby("kind", sort=False):
            lines.append(_row(f"{name}: {kind}", g))
        for label, text in (("bonafide", "all genuine"), ("spoof", "all cloned")):
            g = group[group["label"].str.lower() == label]
            if len(g):
                lines.append(_row(f"**{name}: {text}**", g))
    if missing:
        lines += ["", f"Planned but not recorded ({len(missing)}): {', '.join(missing)}."]
    return "\n".join(lines) + "\n"


def main() -> None:
    """CLI: ``manifest`` from the recordings and plans, then ``report`` the scored file."""
    import argparse

    parser = argparse.ArgumentParser(description="Field recordings and corpus replay calls")
    sub = parser.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("--field")
    m.add_argument("--replay")
    m.add_argument("--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("scored")
    r.add_argument("--field")
    r.add_argument("--replay")
    r.add_argument("--out")
    args = parser.parse_args()
    if args.cmd == "manifest":
        rows, missing = manifest(args.field, args.replay)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(args.out, index=False)
        print(rows.groupby(["set", "kind"]).size().to_string())
        print(f"missing ({len(missing)}):", ", ".join(missing) or "none")
        return
    missing = manifest(args.field, args.replay)[1] if (args.field or args.replay) else None
    text = report(pd.read_csv(args.scored), missing)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
