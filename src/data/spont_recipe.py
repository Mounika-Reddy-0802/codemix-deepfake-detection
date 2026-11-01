"""The spontaneous recipe: the final recipe plus spontaneous speakers (plan task A2.5).

Every Hindi-English training set so far is read lecture speech, and the final recipe calls
43.9% of HiACC's and 37.5% of IndicVoices' genuine spontaneous clips fake
(``docs/results/indicvoices_data_v1.md``). This recipe adds the ``iv_train`` speakers of
IndicVoices (extempore Hindi): their genuine clips **and** XTTS-v2 clones of the same
speakers, so speaking style cannot become the shortcut, each through both telephone codecs
as in the final recipe. The dev set gains the ``iv_dev`` speakers (genuine and clones,
G.711), so model selection can see spontaneous speech. Everything else is the final
recipe: same base manifest, start, schedule.

Only the project's standard manifest columns are kept: the IndicVoices clip manifests
carry transcripts and demographics that must not reach a tracked file.

Training on these rows is team decision D1; until it is recorded, ``config_guard`` rejects
every IndicVoices row in training (the source is evaluation-only).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data import final_recipe

COLUMNS = ["filepath", "label", "language", "speaker", "source", "tool", "condition", "split",
           "utt_id"]  # fmt: skip
TRAIN_OUT = "codemix_adapt_train_spont"
DEV_OUT = "codemix_adapt_dev_spont_channel20"
#: the test speakers' genuine clips and clones, one scoring manifest per codec (evaluation only)
TEST_OUT = {"g711": "score_iv_test_channel20", "amr": "score_iv_test_amr"}
DEV_BASE = "codemix_adapt_dev_channel20"
#: the IndicVoices parts: (genuine manifest, clone manifest) per codec
GENUINE = {"g711": "indicvoices_genuine_channel20", "amr": "indicvoices_genuine_amr"}
CLONES = {"g711": "iv_xtts_{pool}_channel20", "amr": "iv_xtts_{pool}_amr"}


def pool_rows(frame: pd.DataFrame, pool: str, split: str) -> pd.DataFrame:
    """Rows of one IndicVoices pool, standard columns only, relabelled with ``split``."""
    rows = frame[frame["speaker"].astype(str).str.startswith("iv_")].copy()
    if "split" in rows and pool in set(rows["split"]):
        rows = rows[rows["split"] == pool]
    rows["split"] = split
    return rows.reindex(columns=COLUMNS)


def build(manifests_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The training and dev manifests of the spontaneous recipe."""
    mdir = Path(manifests_dir)

    def read(stem: str) -> pd.DataFrame:
        return pd.read_csv(mdir / f"{stem}.csv", dtype={"speaker": str})

    train = [read(final_recipe.OUT).reindex(columns=COLUMNS)]
    for codec in ("g711", "amr"):
        train.append(pool_rows(read(GENUINE[codec]), "iv_train", "train"))
        train.append(pool_rows(read(CLONES[codec].format(pool="train")), "iv_train", "train"))
    dev = [read(DEV_BASE).reindex(columns=COLUMNS)]
    dev.append(pool_rows(read(GENUINE["g711"]), "iv_dev", "dev"))
    dev.append(pool_rows(read(CLONES["g711"].format(pool="dev")), "iv_dev", "dev"))
    return pd.concat(train, ignore_index=True), pd.concat(dev, ignore_index=True)


def build_test(manifests_dir: str) -> dict[str, pd.DataFrame]:
    """The ``iv_test`` speakers' genuine clips and clones, per codec, for scoring only."""
    mdir = Path(manifests_dir)
    out = {}
    for codec, stem in TEST_OUT.items():
        genuine = pd.read_csv(mdir / f"{GENUINE[codec]}.csv", dtype={"speaker": str})
        clones = pd.read_csv(
            mdir / f"{CLONES[codec].format(pool='test')}.csv", dtype={"speaker": str}
        )
        out[stem] = pd.concat(
            [pool_rows(genuine, "iv_test", "eval"), pool_rows(clones, "iv_test", "eval")],
            ignore_index=True,
        )
    return out


def check_test(tests: dict[str, pd.DataFrame], pools: pd.DataFrame) -> None:
    """Only ``iv_test`` speakers in the scoring manifests, and every clone's speaker genuine too."""
    role = dict(zip(pools["speaker"].astype(str), pools["pool"], strict=True))
    for name, frame in tests.items():
        speakers = set(frame["speaker"].astype(str))
        if any(role.get(s) != "iv_test" for s in speakers):
            raise ValueError(f"{name}: a speaker outside iv_test")
        fakes = set(frame.loc[frame["label"] == "spoof", "speaker"].astype(str))
        if not fakes <= set(frame.loc[frame["label"] == "bonafide", "speaker"].astype(str)):
            raise ValueError(f"{name}: clones of a speaker with no genuine clips")


def check(train: pd.DataFrame, dev: pd.DataFrame, pools: pd.DataFrame) -> None:
    """The pools' roles hold: no dev or test speaker in training, no test speaker in dev."""
    role = dict(zip(pools["speaker"].astype(str), pools["pool"], strict=True))
    train_iv = {s for s in train["speaker"].astype(str) if s.startswith("iv_")}
    dev_iv = {s for s in dev["speaker"].astype(str) if s.startswith("iv_")}
    bad_train = sorted(s for s in train_iv if role.get(s) != "iv_train")
    bad_dev = sorted(s for s in dev_iv if role.get(s) != "iv_dev")
    if bad_train or bad_dev:
        raise ValueError(f"pool roles broken: train {bad_train[:3]}, dev {bad_dev[:3]}")
    if train["filepath"].duplicated().any():
        raise ValueError("a clip appears twice in the training manifest")


def main() -> None:
    """CLI: write the spontaneous recipe's training and dev manifests."""
    parser = argparse.ArgumentParser(description="Build the spontaneous-recipe manifests")
    parser.add_argument("--manifests", default="data/manifests")
    args = parser.parse_args()
    train, dev = build(args.manifests)
    pools = pd.read_csv(Path(args.manifests) / "indicvoices_pools.csv")
    check(train, dev, pools)
    tests = build_test(args.manifests)
    check_test(tests, pools)
    outputs = {TRAIN_OUT: train, DEV_OUT: dev, **tests}
    for name, frame in outputs.items():
        frame.to_csv(Path(args.manifests) / f"{name}.csv", index=False)
    for name, frame in outputs.items():
        print(
            f"{name}: {len(frame)} clips, {frame['speaker'].nunique()} speakers, "
            f"labels {frame['label'].value_counts().to_dict()}"
        )


if __name__ == "__main__":
    main()
