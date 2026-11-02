"""Tests for src/data/indicvoices_inventory.py: column discovery and aggregation.

No network: the shard reader needs a token and is exercised manually with
``python -m src.data.indicvoices_inventory --max-shards 1``.
"""

import pandas as pd
import pytest

from src.data import indicvoices_inventory as inv


def _clips() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Speaker_ID": ["s1", "s1", "s1", "s2", "s2", "s3"],
            "duration": [600.0, 1200.0, 360.0, 3600.0, 36.0, 120.0],
            "scenario": ["Extempore", "Extempore", "Read", "Conversation", "Read", "Read"],
            "gender": ["female", "female", "female", "male", "male", "male"],
            "text": ["a", "b", "c", "d", "e", "f"],
        }
    )


def test_columns_are_found_by_name_and_by_value() -> None:
    df = _clips()
    assert inv.find_column(list(df.columns), inv.SPEAKER_COLUMNS) == "Speaker_ID"
    assert inv.find_column(list(df.columns), inv.DURATION_COLUMNS) == "duration"
    assert inv.find_style_column(df) == "scenario"


def test_free_text_is_not_a_style_column() -> None:
    df = pd.DataFrame({"text": [f"read sentence {i}" for i in range(100)]})
    assert inv.find_style_column(df) is None


def test_style_values_map_to_three_styles() -> None:
    assert inv.style_of("Extempore") == "extempore"
    assert inv.style_of("conversational") == "conversation"
    assert inv.style_of("READ") == "read"
    assert inv.style_of("monologue") == "other"


def test_speaker_table_sums_hours_per_style() -> None:
    table, used = inv.speaker_table(_clips())
    s1 = table.set_index("speaker").loc["s1"]
    assert s1["clips"] == 3
    assert s1["hours_extempore"] == pytest.approx(0.5)
    assert s1["hours_read"] == pytest.approx(0.1)
    assert s1["gender"] == "female"
    assert used == {"speaker": "Speaker_ID", "duration": "duration", "style": "scenario"}
    assert table.iloc[0]["speaker"] == "s2"  # most hours first


def test_summary_holds_aggregates_only() -> None:
    table, used = inv.speaker_table(_clips())
    summary = inv.summarise(table, used)
    assert summary["speakers"] == 3
    assert summary["hours"] == round(5916 / 3600, 2)
    assert summary["speakers_by_style"]["read"] == 3
    assert summary["speakers_with_extempore_minutes"][">=20"] == 1
    assert "s1" not in str(summary)  # no speaker ids in the result file


def test_audio_columns_are_never_read() -> None:
    pa = pytest.importorskip("pyarrow")
    schema = pa.schema(
        [
            ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
            ("samples", pa.list_(pa.float32())),
            ("raw", pa.large_binary()),
            ("speaker_id", pa.string()),
            ("duration", pa.float64()),
        ]
    )
    assert inv.metadata_columns(schema) == ["speaker_id", "duration"]


def test_missing_token_stops_with_instructions(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="HF_TOKEN"):
        inv.build_inventory("hindi", tmp_path)
