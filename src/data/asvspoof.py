"""Build Stage-1 manifests from the ASVspoof 2019 LA official protocols.

Stage-1 trains on **ASVspoof 2019 LA and nothing else** (P-001). That is what makes
the English -> Hindi/Tamil -> code-mixed gap a measurement of generalisation rather
than a measurement of how much Indic data leaked into training, so this module
deliberately reads only the corpus's own protocol files and never merges anything
else in.

**The official splits are used verbatim**, not re-derived. ASVspoof ships
speaker-disjoint train/dev/eval partitions with the eval set holding 13 attacks
(A07-A19) that never appear in training -- that unseen-attack design is the point
of the corpus, and any re-split would quietly destroy it while still producing
plausible-looking numbers.

Protocol line format (``ASVspoof2019.LA.cm.<split>.tr[ln].txt``)::

    LA_0079 LA_T_1138215 - - bonafide
    speaker utt_id       - attack label

Audio lives at ``LA/ASVspoof2019_LA_<split>/flac/<utt_id>.flac``.

The attack id is carried through as ``tool`` so the eval split can be sliced per
attack later, and because a manifest that cannot say which attack a row came from
cannot support an attack-wise results table.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.build_manifests import MANIFEST_COLUMNS

#: The corpus's own partitions. Filenames differ: train uses .trn, dev/eval .trl.
PROTOCOLS = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
}

#: Every clip is English read speech, and every row is real audio rather than a
#: clone we generated, so these are constant across the corpus.
LANGUAGE = "en"
SOURCE = "asvspoof2019_la"


class ProtocolError(FileNotFoundError):
    """Raised when the corpus is not laid out the way the protocols expect."""


def protocol_path(la_root: str | Path, split: str) -> Path:
    """Locate one protocol file inside an extracted LA tree."""
    if split not in PROTOCOLS:
        raise KeyError(f"unknown split {split!r}; expected one of {sorted(PROTOCOLS)}")
    return Path(la_root) / "ASVspoof2019_LA_cm_protocols" / PROTOCOLS[split]


def audio_dir(la_root: str | Path, split: str) -> Path:
    """Locate the flac directory for one split."""
    return Path(la_root) / f"ASVspoof2019_LA_{split}" / "flac"


def read_protocol(la_root: str | Path, split: str) -> pd.DataFrame:
    """Parse one protocol file into ``speaker, utt_id, attack, label`` rows."""
    path = protocol_path(la_root, split)
    if not path.is_file():
        raise ProtocolError(f"protocol not found: {path}")
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        rows.append(
            {
                "speaker": parts[0],
                "utt_id": parts[1],
                # "-" marks bonafide; keep the attack id for per-attack slicing.
                "attack": parts[3],
                "label": parts[4].lower(),
            }
        )
    return pd.DataFrame(rows)


def build_manifest(
    la_root: str | Path, split: str, condition: str = "clean", require_audio: bool = False
) -> pd.DataFrame:
    """Turn one official split into a manifest in the project schema.

    ``require_audio`` checks every flac exists. It is off by default because the
    eval split alone is 71,237 files and stat-ing them all is slow; turn it on when
    a manifest is frozen for a paper table.
    """
    protocol = read_protocol(la_root, split)
    flac = audio_dir(la_root, split)
    if not flac.is_dir():
        raise ProtocolError(f"audio directory not found: {flac}")

    frame = pd.DataFrame(
        {
            "filepath": [str(flac / f"{u}.flac") for u in protocol["utt_id"]],
            "label": protocol["label"],
            "language": LANGUAGE,
            "speaker": protocol["speaker"],
            "source": SOURCE,
            # Real speech carries no tool; a spoof carries the attack that made it.
            "tool": protocol["attack"].where(protocol["attack"] != "-", "none"),
            "condition": condition,
            "split": split,
        },
        columns=MANIFEST_COLUMNS,
    )

    if require_audio:
        missing = [p for p in frame["filepath"] if not Path(p).is_file()]
        if missing:
            raise ProtocolError(
                f"{len(missing)} flac file(s) named in the {split} protocol are absent, "
                f"e.g. {missing[0]}"
            )
    return frame


def summarise(frame: pd.DataFrame) -> dict:
    """Counts a reviewer will ask for, and that catch a mis-built manifest."""
    if frame.empty:
        return {"rows": 0}
    attacks = sorted(a for a in frame["tool"].unique() if a != "none")
    return {
        "rows": int(len(frame)),
        "speakers": int(frame["speaker"].nunique()),
        "bonafide": int((frame["label"] == "bonafide").sum()),
        "spoof": int((frame["label"] == "spoof").sum()),
        "attacks": len(attacks),
        "attack_ids": attacks,
    }


def main() -> None:
    """CLI: ``python -m src.data.asvspoof --la-root <DATA_ROOT>/raw/asvspoof2019_LA/LA``."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Build Stage-1 manifests from ASVspoof 2019 LA")
    parser.add_argument("--la-root", required=True, help="the extracted LA/ directory")
    parser.add_argument("--out-dir", default="data/manifests")
    parser.add_argument("--prefix", default="asvspoof")
    parser.add_argument("--splits", default="train,dev,eval")
    parser.add_argument(
        "--require-audio", action="store_true", help="verify every flac exists (slow)"
    )
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames = {}
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        frame = build_manifest(args.la_root, split, require_audio=args.require_audio)
        destination = out / f"{args.prefix}_{split}.csv"
        frame.to_csv(destination, index=False)
        frames[split] = frame
        print(f"{split:5s} -> {destination}")
        print("      " + json.dumps(summarise(frame)))

    # The corpus's own guarantee, re-checked rather than trusted: no speaker may
    # appear in two partitions, or Stage-1's generalisation claim is void.
    names = list(frames)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = set(frames[a]["speaker"]) & set(frames[b]["speaker"])
            status = "OK" if not shared else f"LEAK: {len(shared)} shared"
            print(f"speaker disjoint {a} vs {b}: {status}")

if __name__ == "__main__":
    main()
