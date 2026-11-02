"""Download the IndicVoices clips of the frozen speaker pools (plan task A2.3).

Reads only the 40 frozen row groups (``indicvoices_row_groups.csv``), each a whole
parquet row group fetched over HTTP, and keeps the usable extempore clips (2-15 s) of the
200 pooled speakers (``indicvoices_pools.csv``). Every kept clip is decoded to 16 kHz
mono, level-normalised to -23 dBFS like every other bundle in the project, and written as
``clips/<utt_id>.wav``. Nothing else of the corpus is stored.

The manifest has the project's columns; ``split`` is the speaker's pool (``iv_train``,
``iv_dev``, ``iv_test``), ``language`` is ``hi`` and the original transcript is kept
in ``text`` for the clone jobs. The run is resumable: a row group whose clips are all on
disk is skipped. Usage (``HF_TOKEN`` from ``.env``)::

    python data/indicvoices_run.py src.data.indicvoices_download \\
        --out D:/dfdata/indicvoices_bundle/clean
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.indicvoices_pools import GROUPS_PATH, POOLS_PATH, PREFIX, usable

SR = 16_000
TARGET_DBFS = -23.0
MANIFEST = "data/manifests/indicvoices_genuine.csv"
KEEP = ["speaker_id", "scenario", "duration", "text", "gender", "age_group", "area", "state"]
COLUMNS = ["filepath", "label", "language", "speaker", "source", "tool", "condition", "split",
           "utt_id", "text", "gender", "age_group", "area", "state"]  # fmt: skip


def utt_id(speaker: str, shard: str, row: int) -> str:
    """A stable clip id: speaker, shard number and row within the shard."""
    number = shard.split("-")[1] if "-" in shard else shard
    tag = "v" if shard.startswith("valid") else "t"
    return f"{PREFIX}{speaker}_{tag}{number}_{row:05d}"


def decode(audio_bytes: bytes) -> np.ndarray:
    """Decode one stored clip to 16 kHz mono float32."""
    import soundfile as sf

    from src.utils.audio_utils import resample, to_mono

    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    audio = to_mono(audio)
    return audio if sr == SR else np.asarray(resample(audio, sr, SR), dtype=np.float32)


def select_rows(table: pd.DataFrame, pools: pd.DataFrame) -> pd.DataFrame:
    """The rows of one row group to keep: usable clips of pooled speakers, with their pool."""
    pool_of = dict(zip(pools["speaker"], pools["pool"], strict=True))
    table = table.assign(speaker=PREFIX + table["speaker_id"].astype(str))
    keep = usable(table) & table["speaker"].isin(pool_of)
    return table[keep].assign(split=lambda d: d["speaker"].map(pool_of))


def manifest_rows(kept: pd.DataFrame, shard: str, first_row: int) -> pd.DataFrame:
    """Manifest rows for the kept clips of one row group (positions are within the group)."""
    rows = []
    for position, row in kept.iterrows():
        uid = utt_id(str(row["speaker_id"]), shard, first_row + int(position))
        rows.append(
            {
                "filepath": f"${{DATA_ROOT}}/clips/{uid}.wav",
                "label": "bonafide",
                "language": "hi",
                "speaker": row["speaker"],
                "source": "indicvoices",
                "tool": "none",
                "condition": "clean",
                "split": row["split"],
                "utt_id": uid,
                **{k: row.get(k) for k in ("text", "gender", "age_group", "area", "state")},
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def download(out: str | Path, token: str | None, groups_path: str = GROUPS_PATH) -> pd.DataFrame:
    """Fetch every frozen row group, write the kept clips, return the manifest."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    from src.data.indicvoices_inventory import REPO_ID
    from src.utils.audio_utils import rms_normalize, save_wav

    clips_dir = Path(out) / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    pools = pd.read_csv(POOLS_PATH)
    groups = pd.read_csv(groups_path)
    fs = HfFileSystem(token=token)
    parts = []
    for i, g in enumerate(groups.itertuples(index=False), 1):
        path = f"datasets/{REPO_ID}/hindi/{g.shard}"
        with fs.open(path, "rb", block_size=2**22) as handle:
            parquet = pq.ParquetFile(handle)
            meta = parquet.read_row_group(int(g.group), columns=KEEP).to_pandas()
            kept = select_rows(meta, pools)
            rows = manifest_rows(kept, g.shard, int(g.first_row))
            todo = [p for p in rows["utt_id"] if not (clips_dir / f"{p}.wav").is_file()]
            if todo:
                audio = parquet.read_row_group(int(g.group), columns=["audio_filepath"])
                stored = audio.column("audio_filepath").to_pylist()
                for position, uid in zip(kept.index, rows["utt_id"], strict=True):
                    if uid in todo:
                        wav = rms_normalize(decode(stored[int(position)]["bytes"]), TARGET_DBFS)
                        save_wav(str(clips_dir / f"{uid}.wav"), wav, SR)
                del audio, stored
        parts.append(rows)
        print(f"[{i}/{len(groups)}] {g.shard} group {g.group}: {len(rows)} clips", flush=True)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    """CLI: download the pooled clips and write the manifest."""
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Download the pooled IndicVoices clips")
    parser.add_argument("--out", required=True, help="bundle root; clips go to <out>/clips")
    parser.add_argument("--manifest", default=MANIFEST)
    args = parser.parse_args()
    manifest = download(args.out, os.environ.get("HF_TOKEN"))
    manifest.to_csv(args.manifest, index=False)
    print(manifest.groupby("split")["speaker"].agg(["nunique", "size"]).to_string())


if __name__ == "__main__":
    main()
