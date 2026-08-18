"""Rating sheet for the XTTS pilot: does romanising the transcripts cost quality?

The pilot's remaining question (W3-T5). Team decision P-014 fixed the corpus to
one script, Latin, so the Week-4 run will synthesise from transliterated Hinglish
rather than from Devanagari. What that decision cannot settle from a desk is
whether XTTS-v2 *pronounces* romanised Hindi as well as it pronounces Devanagari
-- its ``hi`` tokeniser was trained on Devanagari, and romanised input may come
out with an English accent.

So the sheet is an **A/B**: the same speaker saying the same sentence under the
same language tag, once from each script. Everything except the script is held
constant, because anything else varying would make a rating difference
uninterpretable -- the same reason ``build_pilot_jobs`` holds the reference voice
constant across cells.

Every row carries the **romanised** transcript, whichever clip it describes. That
is the point of P-014: a rater has to be able to read the target sentence to judge
whether the clip said it, and "sounds fluent" is not the same finding as "said the
right words". Blank ``script`` labels keep the rater from knowing which of the
pair is which until the sheet is scored.

Pure pandas -- no audio, no model, runs in CI.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

#: Columns the team fills in. One row per (clip, rater).
#:
#: ``pair_id`` and ``clip_id`` are deliberately absent: both identify which half
#: of a pair a clip is, which is the thing being hidden. ``read_along`` is shared
#: by a pair, so a rater can still tell that two clips are the same sentence --
#: that is fine and unavoidable, since they need the text to judge the words. What
#: they cannot tell is which one came from Devanagari.
RATING_COLUMNS = [
    "blind_id",
    "audio_path",
    "speaker",
    "language",
    "read_along",  # always romanised, so every rater can follow it
    "rater",
    "sounds_human_1_5",
    "said_the_right_words_1_5",
    "code_switch_natural_1_5",
    "notes",
]

#: Columns that identify the script, withheld from a blind sheet and kept in the
#: answer key instead.
BLIND_COLUMNS = ("script", "clip_id", "pair_id")


@dataclass(frozen=True)
class RatingConfig:
    """Who rates, and whether the script label is hidden while they do it."""

    raters: tuple[str, ...] = ("L", "M", "SK")
    blind: bool = True


def build_ab_sheet(
    roman_jobs: pd.DataFrame,
    deva_jobs: pd.DataFrame,
    roman_dir: str,
    deva_dir: str,
    config: RatingConfig | None = None,
) -> pd.DataFrame:
    """One row per (clip, rater) for the romanised/Devanagari A/B.

    The two job tables must describe the same sentences in the same order --
    ``deva_jobs`` is built from ``roman_jobs``'s ``transcript_source``, so a
    mismatch means the packs have drifted apart and the comparison is void.
    """
    cfg = config or RatingConfig()
    if len(roman_jobs) != len(deva_jobs):
        raise ValueError(
            f"packs describe different job counts ({len(roman_jobs)} vs {len(deva_jobs)}); "
            "the A/B is only meaningful on matched sentences"
        )
    mismatched = (
        roman_jobs["transcript_source"].astype(str).values != deva_jobs["transcript"].astype(str).values
    ).sum()
    if mismatched:
        raise ValueError(
            f"{mismatched} job(s) do not share a source sentence between the two packs; "
            "rebuild the Devanagari pack from the romanised pack's transcript_source"
        )

    rows: list[dict[str, object]] = []
    for pair_id, (roman, deva) in enumerate(
        zip(roman_jobs.itertuples(index=False), deva_jobs.itertuples(index=False), strict=True),
        start=1,
    ):
        read_along = str(roman.transcript)  # romanised for both halves of the pair
        for script, job, directory in (
            ("roman", roman, roman_dir),
            ("devanagari", deva, deva_dir),
        ):
            name = str(job.output_path).rsplit("/", 1)[-1]
            rows.append(
                {
                    "pair_id": pair_id,
                    "clip_id": f"{pair_id:02d}{'a' if script == 'roman' else 'b'}",
                    "audio_path": f"{directory}/{name}",
                    "speaker": str(job.speaker),
                    "language": str(job.language),
                    "read_along": read_along,
                    "script": script,
                    "sounds_human_1_5": "",
                    "said_the_right_words_1_5": "",
                    "code_switch_natural_1_5": "",
                    "notes": "",
                }
            )

    sheet = pd.DataFrame(rows)
    # Shuffle first, THEN number. `clip_id` encodes the script in its a/b suffix,
    # so numbering before the shuffle would leave the answer sitting in the id --
    # and a rater who notices that every "a" sounds alike has un-blinded the whole
    # sheet. `blind_id` is assigned in shuffled order and carries no signal; it is
    # also what the staged audio files are named after, since the pack folder in a
    # raw path (`pilot_roman/` vs `pilot_deva/`) gives the game away just as
    # plainly as a label would. Deterministic seed: reproducible, not guessable.
    sheet = sheet.sort_values("clip_id", kind="stable").reset_index(drop=True)
    sheet = sheet.sample(frac=1.0, random_state=1234).reset_index(drop=True)
    sheet["blind_id"] = [f"{i:02d}" for i in range(1, len(sheet) + 1)]

    out: list[dict[str, object]] = []
    for row in sheet.itertuples(index=False):
        for rater in cfg.raters:
            record = {c: getattr(row, c, "") for c in RATING_COLUMNS if c != "rater"}
            record["rater"] = rater
            if not cfg.blind:
                for column in BLIND_COLUMNS:
                    record[column] = getattr(row, column)
            out.append(record)

    columns = list(RATING_COLUMNS) + ([] if cfg.blind else list(BLIND_COLUMNS))
    return pd.DataFrame(out, columns=columns)


def answer_key(sheet: pd.DataFrame, roman_jobs: pd.DataFrame | None = None) -> pd.DataFrame:
    """The ``blind_id`` -> script mapping withheld from a blind sheet.

    Written to a separate file so the sheet can be scored without the rater ever
    having seen which clip was which.
    """
    if "script" in sheet.columns:
        source = sheet.drop_duplicates("blind_id")
        return (
            source[["blind_id", "pair_id", "clip_id", "script"]]
            .sort_values("blind_id", kind="stable")
            .reset_index(drop=True)
        )
    raise ValueError(
        "answer_key needs the unblinded sheet; build it once with "
        "RatingConfig(blind=False) to obtain the key, and once blind for the raters"
    )


SCORE_COLUMNS = ("sounds_human_1_5", "said_the_right_words_1_5", "code_switch_natural_1_5")


def summarise_ab(sheet: pd.DataFrame, key: pd.DataFrame | None = None) -> dict:
    """Mean scores per script. Blank rows are counted, never guessed at."""
    frame = sheet.copy()
    if key is not None and "script" not in frame.columns:
        frame = frame.merge(key[["blind_id", "script"]], on="blind_id", how="left")

    filled = frame[frame["sounds_human_1_5"].astype(str).str.strip() != ""]
    summary: dict = {
        "rows": int(len(frame)),
        "rated": int(len(filled)),
        "unrated": int(len(frame) - len(filled)),
        "complete": bool(len(filled) == len(frame)) if len(frame) else False,
    }
    if "script" not in frame.columns or filled.empty:
        return summary

    for script, group in filled.groupby("script"):
        summary[str(script)] = {
            column: round(float(pd.to_numeric(group[column], errors="coerce").mean()), 2)
            for column in SCORE_COLUMNS
        }
    return summary


def main() -> None:
    """CLI: ``python -m src.data.pilot_rating --roman-pack ... --deva-pack ...``."""
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Build the XTTS pilot A/B rating sheet")
    parser.add_argument("--roman-pack", required=True, help="pack dir holding the romanised run")
    parser.add_argument("--deva-pack", required=True, help="pack dir holding the Devanagari run")
    parser.add_argument("--sheet", default="docs/qa/pilot_script_rating_sheet.csv")
    parser.add_argument("--key", default="docs/qa/pilot_script_answer_key.csv")
    parser.add_argument("--show-script", action="store_true", help="do not blind the sheet")
    parser.add_argument(
        "--stage-dir",
        default=None,
        help="copy both runs into one folder under neutral names (required for a blind sheet)",
    )
    args = parser.parse_args()

    roman_pack, deva_pack = Path(args.roman_pack), Path(args.deva_pack)
    roman_jobs = pd.read_csv(roman_pack / "generation_jobs.csv")
    deva_jobs = pd.read_csv(deva_pack / "generation_jobs.csv")

    # Built twice from the same deterministic shuffle: once unblinded to obtain
    # the key, once as the raters will see it.
    unblinded = build_ab_sheet(
        roman_jobs,
        deva_jobs,
        roman_dir=str(roman_pack / "outputs"),
        deva_dir=str(deva_pack / "outputs"),
        config=RatingConfig(blind=False),
    )
    key = answer_key(unblinded)

    sheet = build_ab_sheet(
        roman_jobs,
        deva_jobs,
        roman_dir=str(roman_pack / "outputs"),
        deva_dir=str(deva_pack / "outputs"),
        config=RatingConfig(blind=not args.show_script),
    )

    # Blinding the column is not enough: the clip lives under `pilot_roman/` or
    # `pilot_deva/`, and the *path* gives the answer away just as plainly as a
    # label would. Stage both runs into one folder under `clip_id` names so the
    # rater has nothing to go on but the audio.
    if args.stage_dir:
        import shutil

        stage = Path(args.stage_dir)
        stage.mkdir(parents=True, exist_ok=True)
        staged = 0
        for row in sheet.drop_duplicates("blind_id").itertuples(index=False):
            source = Path(row.audio_path)
            if not source.is_file():
                print(f"  [missing] {source}")
                continue
            shutil.copy2(source, stage / f"clip_{row.blind_id}.wav")
            staged += 1
        sheet["audio_path"] = [f"{stage}/clip_{b}.wav" for b in sheet["blind_id"]]
        print(f"staged {staged} clip(s) under neutral names -> {stage}")
    elif not args.show_script:
        print("  [warning] blind sheet without --stage-dir: the pack folder in")
        print("            audio_path reveals the script. Pass --stage-dir.")

    Path(args.sheet).parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(args.sheet, index=False)
    print(f"wrote {len(sheet)} rows -> {args.sheet}")

    key.to_csv(args.key, index=False)
    print(f"wrote the answer key -> {args.key}  (do not open it before rating)")
    print(json.dumps(summarise_ab(sheet, key), indent=2))
    print("\nnext: L, M and SK each rate every row, then score with summarise_ab")


if __name__ == "__main__":
    main()
