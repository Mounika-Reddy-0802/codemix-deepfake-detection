"""Extension pools from the MUCS speakers the frozen pools never used.

The frozen ``speaker_pools.csv`` uses 50 of MUCS's 520 speakers (25 train, 10
adaptation, 15 eval). The other 470 are carved here, once and before any audio is
generated for them, into three disjoint pools:

- ``eval_ext`` -- extra test speakers, cloned for evaluation only, so speaker-level
  intervals stop resting on 15 people;
- ``control`` -- speakers with at least one all-English utterance, held out for the
  language-controlled test (``src.data.codemixing.control_manifest``), evaluation only;
- ``adapt_ext`` -- everyone else, available as extra *genuine* adaptation speakers.

``eval_ext`` and ``control`` are evaluation-only: ``src.training.config_guard`` rejects
any training row from them, exactly as for the frozen eval pool. The original pools
are not touched; this file sits beside them and is frozen and hashed the same way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import speaker_pools as sp
from src.data.codemixing import MAX_SECONDS, MIN_SECONDS, MIN_WORDS, annotate

EXT_PATH = "data/manifests/speaker_pools_ext.csv"
#: Pools in the extension file that must never reach a training manifest.
EVAL_ONLY = ("eval_ext", "control")


def carve_extension(
    clip_index: pd.DataFrame,
    frozen: pd.DataFrame,
    n_eval: int = 60,
    min_clips: int = 40,
    seed: int = 1234,
) -> pd.DataFrame:
    """``speaker, pool, source`` for every MUCS speaker outside the frozen pools.

    ``eval_ext`` is a seeded random draw of ``n_eval`` speakers with at least
    ``min_clips`` usable clips (enough for cloning references and a genuine test side).
    ``control`` is every remaining speaker with an all-English clip; ``adapt_ext`` is
    the rest.
    """
    mucs = clip_index[clip_index["source"].astype(str) == "mucs2021"].copy()
    mucs["speaker"] = mucs["speaker"].astype(str)
    used = set(frozen["speaker"].astype(str))
    mucs = mucs[~mucs["speaker"].isin(used)]
    usable = mucs[mucs["duration_seconds"].between(MIN_SECONDS, MAX_SECONDS)]

    counts = usable.groupby("speaker").size()
    eligible = sorted(counts[counts >= min_clips].index)
    if len(eligible) < n_eval:
        raise ValueError(f"only {len(eligible)} speakers have {min_clips}+ clips, need {n_eval}")
    rng = np.random.default_rng(seed)
    eval_ext = set(rng.choice(eligible, n_eval, replace=False).tolist())

    tagged = annotate(usable)
    english = tagged[(tagged["english_share"] == 1.0) & (tagged["n_words"] >= MIN_WORDS)]
    control = set(english["speaker"]) - eval_ext

    rows = []
    for speaker in sorted(set(mucs["speaker"])):
        pool = (
            "eval_ext" if speaker in eval_ext else "control" if speaker in control else "adapt_ext"
        )
        rows.append({"speaker": speaker, "pool": pool, "source": "mucs2021"})
    return pd.DataFrame(rows)


def evaluation_only_speakers(path: str = EXT_PATH) -> set[str]:
    """Speakers of the extension's evaluation-only pools; empty if the file is absent."""
    from pathlib import Path

    if not Path(path).is_file():
        return set()
    frame = pd.read_csv(path)
    return set(frame.loc[frame["pool"].isin(EVAL_ONLY), "speaker"].astype(str))


def main() -> None:
    """CLI: carve and freeze ``speaker_pools_ext.csv``; refuses to overwrite a frozen file."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Carve extension speaker pools")
    parser.add_argument("--clip-index", default="data/manifests/clip_index.csv")
    parser.add_argument("--frozen", default="data/manifests/speaker_pools.csv")
    parser.add_argument("--out", default=EXT_PATH)
    parser.add_argument("--n-eval", type=int, default=60)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if Path(args.out).is_file():
        raise SystemExit(f"{args.out} is already frozen; delete it deliberately to re-carve")
    index = pd.read_csv(args.clip_index)
    frame = carve_extension(index, pd.read_csv(args.frozen), n_eval=args.n_eval, seed=args.seed)
    record = sp.freeze(frame, args.out)
    print(record.describe())
    print(frame["pool"].value_counts().to_string())


if __name__ == "__main__":
    main()
