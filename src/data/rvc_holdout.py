"""Speaker-held-out split of CM02 for the XTTS+RVC adapter ablation (W9-T1, P-025).

Scoring CM02 showed voice conversion is a blind spot: the XTTS-trained adapter
calls 76-83% of RVC fakes real. The obvious fix is to let the adapter see RVC --
but CM02 is the only RVC data there is, so training on it and then scoring on it
would test the adapter on what it memorised. This splits CM02 so both can happen.

**The rule, enforced here rather than trusted:** a conversion carries two
identities -- the *target* voice it imitates and the *source* recording whose
words, room and microphone it keeps. Every conversion in the training half has
*both* its target and its source speaker absent from the test half. The 25
train-pool speakers are partitioned into two halves (the 12 target voices
alternated 6/6, the other speakers alternated too, over sorted ids so every
machine gets the same split -- P-016); a conversion is kept only if both of its
speakers fall in the same half. Cross-half conversions are dropped, not guessed.

**Fakes never enter alone.** Each half also carries the real source segments its
conversions were made from. Without them, every train-pool voice in the training
set would be fake, and the adapter could learn "these speakers are spoof" instead
of "this is converted".

Pure pandas; the audio steps live in ``normalise_bundle`` and ``train``.
"""

from __future__ import annotations

import pandas as pd

from src.data.scoring_manifests import COLUMNS, portable, qa_passed

RVC_DIR = "rvc_cm02/rvc_converted_wavs/rvc_outputs"
SOURCE_DIR = "interim/rvc_gate/bonafide_source"


class HoldoutError(AssertionError):
    """Raised when the two halves would share a speaker."""


def speaker_halves(jobs: pd.DataFrame) -> tuple[set[str], set[str]]:
    """Partition every speaker CM02 touches into two deterministic halves."""
    targets = sorted(set(jobs["target_speaker"].astype(str)))
    everyone = set(jobs["target_speaker"].astype(str)) | set(jobs["source_speaker"].astype(str))
    others = sorted(everyone - set(targets))
    half_a = set(targets[0::2]) | set(others[0::2])
    return half_a, everyone - half_a


def _half(jobs: pd.DataFrame, ok: set[str], half: set[str], split: str) -> pd.DataFrame:
    tgt, src = jobs["target_speaker"].astype(str), jobs["source_speaker"].astype(str)
    kept = jobs[tgt.isin(half) & src.isin(half) & jobs["clip"].isin(ok)]
    fakes = pd.DataFrame(
        {
            "filepath": [portable(RVC_DIR, c) for c in kept["clip"]],
            "label": "spoof",
            "speaker": kept["target_speaker"].astype(str).to_list(),
            "tool": "rvc",
            "source": "cm02",
        }
    )
    sources = kept.drop_duplicates("source_utt_id")
    reals = pd.DataFrame(
        {
            "filepath": [portable(SOURCE_DIR, f"{u}.wav") for u in sources["source_utt_id"]],
            "label": "bonafide",
            "speaker": sources["source_speaker"].astype(str).to_list(),
            "tool": "none",
            "source": "mucs2021",
        }
    )
    frame = pd.concat([fakes, reals], ignore_index=True)
    frame["language"] = "hi-en"
    frame["condition"] = "clean"
    frame["split"] = split
    return frame[COLUMNS]


def check_disjoint(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Raise if any speaker appears on both sides."""
    shared = set(train["speaker"].astype(str)) & set(test["speaker"].astype(str))
    if shared:
        raise HoldoutError(f"speakers on both sides of the RVC holdout: {sorted(shared)}")


def holdout_split(jobs: pd.DataFrame, qa: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, test) manifests: QA-passed conversions + their real sources, per half."""
    frame = jobs.copy()
    frame["clip"] = frame["output_path"].map(lambda p: str(p).replace("\\", "/").split("/")[-1])
    ok = set(qa_passed(qa)["clip"])
    half_a, half_b = speaker_halves(frame)
    train, test = _half(frame, ok, half_a, "train"), _half(frame, ok, half_b, "eval")
    check_disjoint(train, test)
    return train, test


def main() -> None:
    """CLI: ``python -m src.data.rvc_holdout`` -- writes the two halves to data/manifests/."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Speaker-held-out split of CM02")
    parser.add_argument("--jobs", default="data/manifests/rvc_generation_jobs.csv")
    parser.add_argument("--qa", default="docs/qa/rvc_generation_qa.csv")
    parser.add_argument("--out-dir", default="data/manifests")
    args = parser.parse_args()

    train, test = holdout_split(pd.read_csv(args.jobs), pd.read_csv(args.qa))
    summary = {}
    for name, frame in (("rvc_holdout_train", train), ("rvc_holdout_test", test)):
        frame.to_csv(f"{args.out_dir}/{name}.csv", index=False)
        summary[name] = {
            "spoof": int((frame.label == "spoof").sum()),
            "bonafide": int((frame.label == "bonafide").sum()),
            "speakers": int(frame.speaker.nunique()),
            "target_voices": int(frame.loc[frame.label == "spoof", "speaker"].nunique()),
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
