"""Clip and hour counting: the arithmetic behind the datasheet."""

from __future__ import annotations

import csv

from src.reporting import datasets as ds


def _manifest(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_a_tally_keeps_clips_hours_and_distinct_speakers():
    tally = ds.Tally()
    tally.add(3600.0, "spk1")
    tally.add(1800.0, "spk2")
    tally.add(1800.0, "spk1")
    assert tally.as_dict() == {
        "clips": 3,
        "hours": 2.0,
        "speakers": 2,
        "unmeasured_clips": 0,
    }


def test_audio_that_is_not_on_this_machine_is_counted_but_never_estimated():
    """CM01 lives in a private archive; its clips must not get an invented duration."""
    tally = ds.Tally()
    tally.add(3600.0, "spk1")
    tally.add(None, "spk1")  # archived elsewhere: counted, not measured
    result = tally.as_dict()
    assert result["clips"] == 2
    assert result["unmeasured_clips"] == 1
    assert result["hours"] == 1.0, "the unmeasured clip must not add invented time"


def test_the_duration_cache_survives_a_round_trip(tmp_path):
    cache = tmp_path / "cache.csv"
    ds.save_cache({"b": 2.5, "a": 1.25}, cache)
    assert ds.load_cache(cache) == {"a": 1.25, "b": 2.5}


def test_a_missing_cache_is_empty_rather_than_an_error(tmp_path):
    assert ds.load_cache(tmp_path / "nothing.csv") == {}


def test_stems_are_read_from_either_slash_style():
    assert ds._stem("${DATA_ROOT}/clips/abc_0001.wav") == "abc_0001"
    assert ds._stem(r"C:\dfdata\raw\LA_E_1234.flac") == "LA_E_1234"


def test_a_manifest_is_split_into_real_and_fake_with_hours(tmp_path):
    root = tmp_path
    (root / "data" / "manifests").mkdir(parents=True)
    (root / "experiments" / "results").mkdir(parents=True)

    _manifest(
        root / "data" / "manifests" / "clip_index.csv",
        [
            {"utt_id": "r1", "speaker": "s1", "source": "mucs2021", "duration_seconds": "3600"},
            {"utt_id": "r2", "speaker": "s2", "source": "hiacc", "duration_seconds": "1800"},
        ],
    )
    _manifest(
        root / "data" / "manifests" / "codemix_eval.csv",
        [
            {
                "filepath": "x/r1.wav",
                "label": "bonafide",
                "speaker": "s1",
                "tool": "none",
                "utt_id": "r1",
            },
            {
                "filepath": "x/f1.wav",
                "label": "spoof",
                "speaker": "s1",
                "tool": "xtts_v2",
                "utt_id": "r1",
            },
            {
                "filepath": "x/f2.wav",
                "label": "spoof",
                "speaker": "s2",
                "tool": "tortoise",
                "utt_id": "r2",
            },
        ],
    )

    report = ds.summarise(root=root, data_root=root / "no-audio-here")

    assert report["corpora"]["mucs2021"] == {
        "clips": 1,
        "hours": 1.0,
        "speakers": 1,
        "unmeasured_clips": 0,
    }
    assert report["corpora"]["hiacc"]["hours"] == 0.5

    entry = report["manifests"]["codemix_eval"]
    assert entry["clips"] == 3
    assert entry["bonafide"]["clips"] == 1
    assert entry["spoof"]["clips"] == 2
    assert entry["spoof_per_real"] == 2.0
    assert "XTTS-v2 (CM01 / CM03)" in entry["by_tool"]
    assert "Tortoise-TTS (CM04, held out)" in entry["by_tool"]


def test_tool_totals_only_count_spoof_clips(tmp_path):
    root = tmp_path
    (root / "data" / "manifests").mkdir(parents=True)
    _manifest(
        root / "data" / "manifests" / "codemix_eval.csv",
        [
            {
                "filepath": "a.wav",
                "label": "bonafide",
                "speaker": "s1",
                "tool": "none",
                "utt_id": "u1",
            },
            {"filepath": "b.wav", "label": "spoof", "speaker": "s1", "tool": "rvc", "utt_id": "u2"},
        ],
    )
    report = ds.summarise(root=root, data_root=root / "none")
    assert "bonafide (no tool)" not in report["attack_tools"]
    assert report["attack_tools"]["RVC v2 (CM02)"]["clips"] == 1


def test_a_manifest_that_does_not_exist_is_skipped_not_invented(tmp_path):
    (tmp_path / "data" / "manifests").mkdir(parents=True)
    report = ds.summarise(root=tmp_path, data_root=tmp_path / "none")
    assert report["manifests"] == {}
