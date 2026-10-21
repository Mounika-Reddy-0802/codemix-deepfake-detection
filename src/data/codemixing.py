"""How code-mixed is an utterance, and a language-controlled test set built on it.

MUCS transcripts write Hindi in Devanagari and English in Latin script, so each word's
language can be read from its script. From that:

- ``token_languages`` tags each word ``"hi"`` (Devanagari), ``"en"`` (Latin letters) or
  drops it (numbers, punctuation);
- ``cmi`` is the Code-Mixing Index of Das and Gambäck (2014),
  ``100 * (1 - max(n_hi, n_en) / n_tagged)``: 0 for a single-language utterance, 50 for
  an even mix;
- ``english_share`` is the fraction of tagged words that are English.

The question these answer is whether the English-trained detector fails *because of*
code-mixing or because of the recording domain. ``control_manifest`` builds the test
for it: genuine clips from the same speakers, the same lecture recordings, in three
language conditions -- all English, all Hindi, and mixed -- so that speaker and
recording are held fixed and only the language changes. It draws only from speakers
the ``control`` pool reserves for evaluation (``src.data.speaker_pools_ext``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Clip length kept by preprocessing everywhere else in the project.
MIN_SECONDS, MAX_SECONDS = 2.0, 10.0
#: A single-language clip must have at least this many tagged words to count.
MIN_WORDS = 3
#: A "mixed" control clip has at least this CMI (roughly one word in five switched).
MIXED_CMI = 20.0
CONDITIONS = ("all English", "all Hindi", "mixed")


def token_languages(text: str) -> list[str]:
    """``"hi"`` or ``"en"`` for every word that carries script; others are dropped."""
    tags = []
    for token in str(text).split():
        if any("ऀ" <= ch <= "ॿ" for ch in token):
            tags.append("hi")
        elif any(ch.isascii() and ch.isalpha() for ch in token):
            tags.append("en")
    return tags


def cmi(text: str) -> float:
    """Code-Mixing Index in [0, 50]; 0 when the utterance has no tagged words."""
    tags = token_languages(text)
    if not tags:
        return 0.0
    return 100.0 * (1.0 - max(tags.count("hi"), tags.count("en")) / len(tags))


def english_share(text: str) -> float:
    """Fraction of tagged words that are English; 0 when there are none."""
    tags = token_languages(text)
    return tags.count("en") / len(tags) if tags else 0.0


def switch_count(text: str) -> int:
    """Number of language changes between consecutive tagged words."""
    tags = token_languages(text)
    return sum(a != b for a, b in zip(tags, tags[1:], strict=False))


def annotate(clips: pd.DataFrame) -> pd.DataFrame:
    """Add ``n_words``, ``cmi``, ``english_share`` and ``switches`` from ``transcript``."""
    out = clips.copy()
    text = out["transcript"].astype(str)
    out["n_words"] = text.map(lambda t: len(token_languages(t)))
    out["cmi"] = text.map(cmi)
    out["english_share"] = text.map(english_share)
    out["switches"] = text.map(switch_count)
    return out


def condition_of(row: pd.Series) -> str | None:
    """Which control condition a clip belongs to, if any."""
    if row["n_words"] < MIN_WORDS:
        return None
    if row["english_share"] == 1.0:
        return "all English"
    if row["english_share"] == 0.0:
        return "all Hindi"
    if row["cmi"] >= MIXED_CMI:
        return "mixed"
    return None


def control_manifest(
    clip_index: pd.DataFrame, control_speakers: set[str], seed: int = 1234
) -> pd.DataFrame:
    """Genuine MUCS clips of the control speakers in three language conditions.

    For every speaker with at least one all-English clip, all of their all-English
    clips are taken, plus as many all-Hindi and as many mixed clips from the same
    speaker (fewer if they have fewer). Speaker and recording domain are therefore the
    same across conditions; only the language changes. Deterministic given ``seed``.
    """
    clips = clip_index[clip_index["source"].astype(str) == "mucs2021"]
    clips = clips[clips["speaker"].astype(str).isin(control_speakers)]
    clips = clips[clips["duration_seconds"].between(MIN_SECONDS, MAX_SECONDS)]
    clips = annotate(clips)
    clips["condition"] = clips.apply(condition_of, axis=1)
    clips = clips[clips["condition"].notna()]

    rng = np.random.default_rng(seed)
    picked = []
    for _speaker, group in clips.groupby(clips["speaker"].astype(str), sort=True):
        english = group[group["condition"] == "all English"]
        if english.empty:
            continue
        picked.append(english)
        for condition in ("all Hindi", "mixed"):
            pool = group[group["condition"] == condition].sort_values("utt_id")
            n = min(len(english), len(pool))
            if n:
                picked.append(pool.iloc[np.sort(rng.choice(len(pool), n, replace=False))])
    if not picked:
        return pd.DataFrame(
            columns=[
                "filepath",
                "label",
                "language",
                "speaker",
                "source",
                "tool",
                "condition",
                "split",
                "utt_id",
                "language_condition",
            ]
        )
    chosen = pd.concat(picked, ignore_index=True)
    return pd.DataFrame(
        {
            "filepath": chosen["wav_path"],
            "label": "bonafide",
            "language": "hi-en",
            "speaker": chosen["speaker"].astype(str),
            "source": "mucs2021",
            "tool": "none",
            "condition": "clean",
            "split": "eval",
            "utt_id": chosen["utt_id"],
            "start_seconds": chosen["start_seconds"],
            "end_seconds": chosen["end_seconds"],
            "language_condition": chosen["condition"],
            "cmi": chosen["cmi"].round(2),
            "english_share": chosen["english_share"].round(3),
        }
    )


def main() -> None:
    """CLI: write the language-control source manifest from the frozen extension pools."""
    import argparse

    from src.data.speaker_pools_ext import EXT_PATH

    parser = argparse.ArgumentParser(description="Build the language-controlled test set")
    parser.add_argument("--clip-index", default="data/manifests/clip_index.csv")
    parser.add_argument("--pools-ext", default=EXT_PATH)
    parser.add_argument("--out", default="data/manifests/mucs_language_control_source.csv")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    pools = pd.read_csv(args.pools_ext)
    control = set(pools.loc[pools["pool"] == "control", "speaker"].astype(str))
    manifest = control_manifest(pd.read_csv(args.clip_index), control, seed=args.seed)
    manifest.to_csv(args.out, index=False)
    by = manifest.groupby("language_condition")["speaker"].agg(["size", "nunique"])
    print(f"wrote {args.out}: {len(manifest)} clips, {manifest['speaker'].nunique()} speakers")
    print(by.rename(columns={"size": "clips", "nunique": "speakers"}).to_string())


if __name__ == "__main__":
    main()
