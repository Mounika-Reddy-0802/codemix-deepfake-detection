"""Inventory of an IndicVoices language subset, read from metadata columns only.

IndicVoices (AI4Bharat, CC BY 4.0, gated) holds read, extempore and conversational
speech; the Hindi subset alone is 82 parquet shards of about 0.5-0.8 GB each. Choosing
speaker pools needs only who speaks, for how long and in which style, so this module
reads the parquet **metadata columns** over HTTP range requests and never the audio
column: a few megabytes per shard instead of the whole corpus.

The dataset card is gated, so column names are not assumed. The speaker and duration
columns are matched against known spellings, and the speaking-style column is found
from its values (the low-cardinality text column whose values name read, extempore or
conversational speech).

Each shard's metadata is cached as its own parquet file, so an interrupted run resumes
where it stopped. The per-speaker table stays under ``data/`` (ignored by git); only the
aggregate summary is written as a result file.

Usage::

    python -m src.data.indicvoices_inventory --subset hindi \\
        --cache data/indicvoices/meta --out data/indicvoices/hindi_speakers.csv \\
        --summary experiments/results/indicvoices_inventory_hindi.json

Needs ``HF_TOKEN`` in the environment from an account that has accepted the dataset's
terms on its Hugging Face page. This is an inventory: it does not decide whether the
corpus is used for training (that is a recorded team decision).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.data.download import hf_token

if TYPE_CHECKING:  # import only for type checkers, never at runtime
    import pandas as pd
    import pyarrow as pa

REPO_ID = "ai4bharat/IndicVoices"
LICENCE = "CC BY 4.0"

SPEAKER_COLUMNS = ("speaker_id", "speaker", "spk_id", "speakerid", "speaker_name")
DURATION_COLUMNS = ("duration", "audio_duration", "duration_s", "duration_sec", "length")
#: words that identify the speaking-style values; matched case-insensitively
STYLE_WORDS = ("read", "extempore", "conversation")
#: a text column with more distinct values than this is not a category
MAX_CATEGORY_VALUES = 60


def metadata_columns(schema: pa.Schema) -> list[str]:
    """Return the columns that carry metadata, not audio.

    Binary columns, lists (sample arrays) and structs holding binary or list fields
    (the ``datasets`` Audio feature) are left out, so reading the returned columns
    never downloads audio.
    """
    import pyarrow as pa

    def is_audio(dtype: pa.DataType) -> bool:
        if pa.types.is_binary(dtype) or pa.types.is_large_binary(dtype):
            return True
        if pa.types.is_list(dtype) or pa.types.is_large_list(dtype):
            return True
        if pa.types.is_struct(dtype):
            return any(is_audio(dtype.field(i).type) for i in range(dtype.num_fields))
        return False

    return [f.name for f in schema if not is_audio(f.type)]


def find_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    """Return the first column whose lower-cased name is one of ``candidates``."""
    lowered = {c.lower(): c for c in columns}
    for name in candidates:
        if name in lowered:
            return lowered[name]
    return None


def find_style_column(df: pd.DataFrame) -> str | None:
    """Return the text column whose values name the speaking style, or ``None``.

    The column must be a category (at most ``MAX_CATEGORY_VALUES`` distinct values)
    and at least two of read / extempore / conversation must appear in its values.
    """
    best, best_hits = None, 1
    for col in df.columns:
        if df[col].dtype != object:
            continue
        values = df[col].dropna().astype(str).str.lower().unique()
        if len(values) == 0 or len(values) > MAX_CATEGORY_VALUES:
            continue
        hits = sum(any(word in v for v in values) for word in STYLE_WORDS)
        if hits > best_hits:
            best, best_hits = col, hits
    return best


def style_of(value: Any) -> str:
    """Map a raw style value to ``read``, ``extempore``, ``conversation`` or ``other``."""
    text = str(value).lower()
    for word in STYLE_WORDS:
        if word in text:
            return word
    return "other"


def speaker_table(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str | None]]:
    """Aggregate clip metadata into one row per speaker.

    Returns the table (clips, hours in total and per style, and the first value of
    every other category column, such as gender or district) and the columns used.
    """
    import pandas as pd

    cols = list(df.columns)
    speaker = find_column(cols, SPEAKER_COLUMNS)
    duration = find_column(cols, DURATION_COLUMNS)
    if speaker is None or duration is None:
        raise ValueError(f"speaker or duration column not found among {cols}")
    style = find_style_column(df)

    work = pd.DataFrame(
        {
            "speaker": df[speaker].astype(str),
            "hours": pd.to_numeric(df[duration], errors="coerce").fillna(0.0) / 3600.0,
            "style": df[style].map(style_of) if style else "unknown",
        }
    )
    table = work.groupby("speaker").agg(clips=("hours", "size"), hours=("hours", "sum"))
    per_style = work.pivot_table(
        index="speaker", columns="style", values="hours", aggfunc="sum", fill_value=0.0
    ).add_prefix("hours_")
    table = table.join(per_style)

    skip = {speaker, duration, style}
    for col in cols:
        if col in skip or df[col].dtype != object:
            continue
        if df[col].nunique(dropna=True) > MAX_CATEGORY_VALUES:
            continue  # free text (transcripts, file names), not a speaker attribute
        table[col] = df.groupby(df[speaker].astype(str))[col].first()
    used = {"speaker": speaker, "duration": duration, "style": style}
    return table.reset_index().sort_values("hours", ascending=False), used


def summarise(table: pd.DataFrame, used: dict[str, str | None]) -> dict[str, Any]:
    """Aggregate figures for the result file: no speaker ids, no per-person rows."""
    styles = [c for c in table.columns if c.startswith("hours_")]
    out: dict[str, Any] = {
        "repo_id": REPO_ID,
        "licence": LICENCE,
        "columns_used": used,
        "speakers": int(len(table)),
        "clips": int(table["clips"].sum()),
        "hours": round(float(table["hours"].sum()), 2),
        "hours_by_style": {c[6:]: round(float(table[c].sum()), 2) for c in styles},
        "speakers_by_style": {c[6:]: int((table[c] > 0).sum()) for c in styles},
    }
    if "hours_extempore" in table.columns:
        minutes = table["hours_extempore"] * 60.0
        out["speakers_with_extempore_minutes"] = {
            f">={m}": int((minutes >= m).sum()) for m in (2, 5, 10, 20)
        }
    for col in table.columns:
        if table[col].dtype == object and col != "speaker":
            out.setdefault("category_counts", {})[col] = {
                str(k): int(v) for k, v in table[col].value_counts().head(40).items()
            }
    return out


def list_shards(subset: str, token: str | None = None) -> list[str]:
    """Return the repo paths of every parquet shard of ``subset`` (train and valid)."""
    from huggingface_hub import HfApi

    files = HfApi(token=token).list_repo_files(REPO_ID, repo_type="dataset")
    prefix = f"{subset.lower()}/"
    return sorted(f for f in files if f.startswith(prefix) and f.endswith(".parquet"))


def read_shard_metadata(path: str, token: str | None) -> pa.Table:
    """Read only the metadata columns of one shard over HTTP range requests."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem(token=token)
    with fs.open(f"datasets/{REPO_ID}/{path}", "rb", block_size=2**20) as handle:
        parquet = pq.ParquetFile(handle)
        columns = metadata_columns(parquet.schema_arrow)
        return parquet.read(columns=columns)


def build_inventory(
    subset: str, cache: Path, max_shards: int | None = None, token: str | None = None
) -> pd.DataFrame:
    """Read (or reuse the cached) metadata of every shard and concatenate it."""
    import pandas as pd
    import pyarrow.parquet as pq

    token = token or hf_token()
    if not token:
        raise SystemExit(
            "HF_TOKEN is not set: accept the IndicVoices terms on its Hugging Face page, "
            "create a read token and put it in .env as HF_TOKEN."
        )
    cache.mkdir(parents=True, exist_ok=True)
    shards = list_shards(subset, token)[:max_shards]
    frames = []
    for i, path in enumerate(shards, 1):
        cached = cache / Path(path).name
        if not cached.exists():
            table = read_shard_metadata(path, token)
            tmp = cached.with_suffix(".part")
            pq.write_table(table, tmp)
            tmp.replace(cached)
        frame = pd.read_parquet(cached)
        frame["shard"] = Path(path).name
        frames.append(frame)
        print(f"[{i}/{len(shards)}] {path}: {len(frame)} clips", flush=True)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    """CLI: build the per-speaker table and the aggregate summary for one subset."""
    import argparse

    parser = argparse.ArgumentParser(description="IndicVoices metadata-only inventory")
    parser.add_argument("--subset", default="hindi")
    parser.add_argument("--cache", type=Path, default=Path("data/indicvoices/meta"))
    parser.add_argument("--out", type=Path, default=Path("data/indicvoices/hindi_speakers.csv"))
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("experiments/results/indicvoices_inventory_hindi.json"),
    )
    parser.add_argument("--max-shards", type=int, default=None)
    args = parser.parse_args()

    clips = build_inventory(args.subset, args.cache, args.max_shards)
    table, used = speaker_table(clips)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    summary = summarise(table, used)
    summary["subset"] = args.subset
    summary["shards_read"] = int(clips["shard"].nunique())
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("speakers", "hours", "hours_by_style")}, indent=2))


if __name__ == "__main__":
    main()
