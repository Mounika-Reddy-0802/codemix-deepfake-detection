"""Every condition for voice check v2 beside the current detector, without the training speakers.

The decision is ``src.reporting.voice_check_v2`` (five conditions, fixed before training). This
table adds what the rule reports but does not judge: clip counts, the AMR-NB rows and the
replayed clones. Every column leaves out the 31 ``gen_train`` speakers wherever they occur
(the sets built from the 61 cloned speakers, over G.711 and AMR-NB alike), so no voice v2 heard
in training is counted, and the reference is measured on exactly the same clips. The spontaneous
recipe alone (decision D1 part b, not adopted, P-040) is shown beside them, so the table also
says what the IndicF5 and Chatterbox rows added on top of that recipe.

    python -m src.reporting.voice_check_v2_table --out experiments/results/voice_check_v2_table.md
"""

from __future__ import annotations

from pathlib import Path

from src.reporting.voice_check_v2 import CANDIDATES, REFERENCE

#: the spontaneous recipe without the generator study's rows: v2 minus what P-042 added
RECIPE_ALONE = ("ssl_spontrain_s1234", "ssl_spontrain_s2025")
MODELS = (REFERENCE, *RECIPE_ALONE, *CANDIDATES)

#: the sets that hold any of the 61 cloned speakers, G.711 and AMR-NB
POOL_SETS = frozenset(
    {"new_speakers", "cm04", "cm07", "cm09", "new_speakers_amr", "cm07_amr", "cm04_amr"}
)
#: what each column heard in training (always other speakers than the ones measured), from
#: the manifests: XTTS-v2 and RVC (CM02's speaker-held-out half, P-025) in every recipe here,
#: IndicF5 and Chatterbox in v2 only. Tortoise is in no training set.
HEARD = "XTTS-v2 and RVC in every column, IndicF5 and Chatterbox in v2 only; Tortoise in none"


def rows(exclude: frozenset[str]) -> list[tuple[str, dict[str, str]]]:
    """(condition, {model: cell}) for every clip-level condition and every call kind."""
    from src.reporting import any_audio_accuracy as acc

    models = MODELS
    out = []
    for label, key, which, expected in acc.CONDITIONS:
        if key is None:
            out.append((f"**{label}**", dict.fromkeys(models, "")))
            continue
        cells = {}
        for m in models:
            r = acc.accuracy(m, key, which, expected, exclude if key in POOL_SETS else frozenset())
            cells[m] = "-" if r is None else f"{r[0]:.1f} (n={r[1]})"
        out.append((label.replace(" (unseen)", "").replace(" (seen tool)", ""), cells))
    out.append(
        ("**Real recordings (as captured on the receiving phone)**", dict.fromkeys(models, ""))
    )
    counts = {m: acc.call_counts(m) or {} for m in models}
    for kind, tool, text in acc.CALL_KINDS:
        cells = {}
        for m in models:
            c = counts[m].get((kind, tool))
            cells[m] = "-" if c is None else f"{c[0]}/{c[1]}"
        out.append((text, cells))
    return out


def table(exclude: frozenset[str]) -> str:
    """The markdown table, one column per model."""
    from src.reporting import any_audio_accuracy as acc

    models = MODELS
    lines = [
        "| Condition | " + " | ".join(models) + " |",
        "|---|" + "---|" * len(models),
    ]
    lines += [
        f"| {label} | " + " | ".join(c[m] for m in models) + " |" for label, c in rows(exclude)
    ]
    thr = ", ".join(f"{m} {acc.threshold(m):.3f}" for m in models)
    lines += [
        "",
        f"Percent of clips judged correctly (genuine as real, clones as cloned); the "
        f"{len(exclude)} `gen_train` speakers are left out of every column. Operating points, "
        f"each chosen on dev only: {thr}. Heard in training (other speakers): {HEARD}.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    """CLI: print the table and optionally write it."""
    import argparse

    from src.data import generator_study as gs

    parser = argparse.ArgumentParser(description="Voice check v2 against the current detector")
    parser.add_argument("--pools", default=f"data/manifests/{gs.POOLS_OUT}")
    parser.add_argument("--out")
    args = parser.parse_args()
    text = table(frozenset(gs.train_speakers(args.pools)))
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
