"""Tests for the CM02 reporting layer: what leaves Kaggle and enters git.

Two of these guard mistakes this project has already made once. Commit f60f989
gitignored the ASVspoof manifests because they carried one machine's absolute
paths, and the RVC metadata is written full of ``/kaggle/working/...`` for the
same reason — so :func:`portable_records` is pinned here rather than trusted. And
the ADR log's own rule is that entries are appended, never rewritten, so a
repeated Kaggle run must not be able to leave two P-020 rows behind.

Pure stdlib, pandas and tmp_path: no audio, no Praat, no RVC checkout.
"""

from __future__ import annotations

import json

import pandas as pd

from src.data import rvc_report

TAXONOMY = """# Attack taxonomy

| Attack ID | Generation Type | Model | Base Data | Samples (target) | No. Spks | Split | Seen by |
|---|---|---|---|---|---|---|---|
| CM01 | TTS (zero-shot clone) | XTTS-v2 | MUCS train pool | ~4,000 | 20-25 | Train | S3 |
| CM02 | VC | RVC (per-speaker models) | MUCS train pool | ~1,500 | 10-15 | Train | S3 |
"""

DECISIONS = """# Problems & decisions (ADR log)

| ID | Decision | Why |
|----|----------|-----|
| **P-019** | XTTS-v2 compresses pitch range | measured |
"""


# --------------------------------------------------------------------------- #
# Portable metadata
# --------------------------------------------------------------------------- #
def test_data_paths_are_rebased_on_the_data_root_token() -> None:
    rows = rvc_report.portable_records(
        [{"reference_wav": "/kaggle/working/dataroot/raw/mucs2021/train/rec.wav"}],
        root="/kaggle/working/dataroot",
    )
    assert rows[0]["reference_wav"] == "${DATA_ROOT}/raw/mucs2021/train/rec.wav"


def test_a_kaggle_output_path_keeps_its_name_and_loses_the_machine() -> None:
    """The converted clip is not in the repo and never will be; the name identifies it."""
    rows = rvc_report.portable_records(
        [{"output_path": "/kaggle/working/rvc_outputs/rvc_445718_00001.wav"}]
    )
    assert rows[0]["output_path"] == "rvc_445718_00001.wav"


def test_no_absolute_machine_path_survives_into_the_written_metadata(tmp_path) -> None:
    destination = tmp_path / "meta.jsonl"
    written = rvc_report.write_portable_metadata(
        [
            {
                "output_path": "/kaggle/working/rvc_outputs/a.wav",
                "reference_wav": "/kaggle/working/dataroot/raw/mucs2021/train/b.wav",
                "speaker": "445718",
            }
        ],
        str(destination),
        root="/kaggle/working/dataroot",
    )
    assert written == 1
    text = destination.read_text(encoding="utf-8")
    assert "/kaggle/working" not in text
    assert json.loads(text)["speaker"] == "445718"


def test_records_without_paths_are_untouched() -> None:
    rows = rvc_report.portable_records([{"speaker": "1", "seed": 4201}])
    assert rows == [{"speaker": "1", "seed": 4201}]


# --------------------------------------------------------------------------- #
# The ADR log and the taxonomy
# --------------------------------------------------------------------------- #
def test_a_decision_is_appended_once(tmp_path) -> None:
    log = tmp_path / "decisions.md"
    log.write_text(DECISIONS, encoding="utf-8")
    entry = "| **P-020** | RVC toolchain | three defects |"
    assert rvc_report.append_decision(entry, str(log)) is True
    assert rvc_report.append_decision(entry, str(log)) is False
    assert log.read_text(encoding="utf-8").count("P-020") == 1


def test_appending_a_decision_leaves_the_existing_ones_alone(tmp_path) -> None:
    log = tmp_path / "decisions.md"
    log.write_text(DECISIONS, encoding="utf-8")
    rvc_report.append_decision("| **P-020** | new | why |", str(log))
    assert "| **P-019** | XTTS-v2 compresses pitch range | measured |" in log.read_text()


def test_taxonomy_row_takes_measured_counts(tmp_path) -> None:
    table = tmp_path / "taxonomy.md"
    table.write_text(TAXONOMY, encoding="utf-8")
    assert rvc_report.update_taxonomy_row("CM02", "**1,487**", "12", str(table)) is True
    line = [ln for ln in table.read_text().splitlines() if "CM02" in ln][0]
    assert "**1,487**" in line
    assert "~1,500" not in line
    assert "RVC (per-speaker models)" in line


def test_updating_one_taxonomy_row_does_not_touch_another(tmp_path) -> None:
    table = tmp_path / "taxonomy.md"
    table.write_text(TAXONOMY, encoding="utf-8")
    rvc_report.update_taxonomy_row("CM02", "1,487", "12", str(table))
    assert "| CM01 | TTS (zero-shot clone) | XTTS-v2 | MUCS train pool | ~4,000 |" in (
        table.read_text()
    )


def test_an_unknown_attack_id_is_reported_not_invented(tmp_path) -> None:
    table = tmp_path / "taxonomy.md"
    table.write_text(TAXONOMY, encoding="utf-8")
    assert rvc_report.update_taxonomy_row("CM99", "1", "1", str(table)) is False


# --------------------------------------------------------------------------- #
# Job screening column names
# --------------------------------------------------------------------------- #
def test_the_screen_reads_an_rvc_job_tables_target_as_the_speaker(tmp_path) -> None:
    """CM01 tables call it ``speaker``; an RVC job names two speakers."""
    jobs = pd.DataFrame(
        [
            {
                "target_speaker": "445718",
                "transcript": "x" * 40,
                "language": "hi",
                "output_path": "out/rvc_445718_00001.wav",
            }
        ]
    )
    report = rvc_report.screen_rvc_jobs(jobs, str(tmp_path))
    assert report.loc[0, "speaker"] == "445718"
    assert report.loc[0, "reason"] == "missing"


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #
def _context() -> dict:
    return {
        "accelerator": "GPU T4 x2",
        "run_date": "2026-09-05",
        "branch": "week8-krishna-rvc-generation",
        "commit": "d9f3eea",
        "rvc_commit": "81eed5e",
        "n_targets": 12,
        "train_epochs": 100,
        "conversions_per_target": 125,
        "written": 1487,
        "failures": 0,
        "total_audio_hours": 2.4,
        "train_pool_speakers": 25,
        "sample_rate": "40k",
        "f0_method": "rmvpe",
        "pitch_floor": 60.0,
        "pitch_ceiling": 400.0,
        "model_mb": 55.0,
        "index_rows": 52825,
        "index_speakers": 520,
        "per_target": [
            {
                "speaker": "445718",
                "train_clips": 127,
                "train_seconds": 771.5,
                "epochs": 100,
                "written": 125,
                "qa_pass": "125/125",
            }
        ],
        "qa": {
            "clips": 1487,
            "passed": 1487,
            "failed": 0,
            "pass_rate": 100.0,
            "median_chars_per_sec": 13.2,
            "failures": {},
        },
        "pitch": {
            "converted": {"median_f0_iqr_hz": 40.6, "median_f0_hz": 112.0, "usable": 1480},
            "real_source": {"median_f0_iqr_hz": 41.3, "median_f0_hz": 111.0, "usable": 1480},
            "retention_pct": 98.3,
            "verdict": "P-019's prediction HOLDS: converted clips keep 98.3%",
        },
        "flag_rate": 0,
        "flag_silent": 0,
        "flag_clipped": 0,
        "flag_short": 0,
        "flag_missing": 0,
    }


def test_the_results_document_states_the_measured_counts() -> None:
    text = rvc_report.render_results_doc(_context())
    assert "**1487**" in text
    assert "40.6 Hz" in text
    assert "41.3 Hz" in text
    assert rvc_report.MODEL_MANIFEST_PATH in text


def test_the_results_document_says_the_audio_is_not_committed() -> None:
    """A public repo, cloned voices of identifiable speakers: this must be explicit."""
    text = rvc_report.render_results_doc(_context())
    assert "not** committed" in text
    assert "gitignored" in text


def test_the_qa_document_carries_the_verdict_and_the_pitch_table() -> None:
    text = rvc_report.render_qa_doc(_context())
    assert "usable as generated" in text
    assert "HOLDS" in text
    assert "25–29 Hz" in text


def test_a_failing_screen_produces_a_failing_verdict() -> None:
    ctx = _context()
    ctx["qa"] = {
        "clips": 100,
        "passed": 70,
        "failed": 30,
        "pass_rate": 70.0,
        "median_chars_per_sec": 40.0,
        "failures": {"speech too fast / truncated (44 chars/s)": 30},
    }
    text = rvc_report.render_qa_doc(ctx)
    assert "does NOT clear the bar" in text


def test_the_decision_entry_names_all_three_defects() -> None:
    entry = rvc_report.decision_entry(_context())
    assert entry.startswith("| **P-020** |")
    assert "raw/" in entry
    assert "clip_index.csv" in entry
    assert "PYTHONSAFEPATH" in entry


def test_the_job_table_is_made_portable_before_it_is_committed(tmp_path) -> None:
    """`!data/manifests/*.csv` is tracked on purpose; a Kaggle path there is junk."""
    jobs = pd.DataFrame(
        [
            {
                "job_id": 1,
                "target_speaker": "445718",
                "source_wav": "/kaggle/working/dataroot/raw/mucs2021/train/rec.wav",
                "output_path": "/kaggle/working/rvc_outputs/rvc_445718_00001.wav",
            }
        ]
    )
    out = tmp_path / "jobs.csv"
    frame = rvc_report.write_portable_jobs(jobs, str(out), root="/kaggle/working/dataroot")
    assert frame.loc[0, "source_wav"] == "${DATA_ROOT}/raw/mucs2021/train/rec.wav"
    assert frame.loc[0, "output_path"] == "rvc_445718_00001.wav"
    assert "/kaggle/" not in out.read_text(encoding="utf-8")
    assert frame.loc[0, "target_speaker"] == "445718"
