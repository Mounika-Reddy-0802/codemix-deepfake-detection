"""Tests for the W4-T2 RVC attack (CM02): job table, firewall, toolchain plumbing.

CM02 is the harder of the two seen attack families, and the thing that makes it
usable is not the audio quality but the bookkeeping around it:

- **the pool firewall on both endpoints** -- a converted clip carries the target's
  speaker id, but the source voice survives in the prosody, so an eval-pool source
  would leak an evaluation voice into training just as surely as an eval-pool
  target would;
- **no speaker converted into themselves**, which is a re-encode, not an attack;
- **no repeated (target, source clip) pair**, or the detector learns the utterance
  instead of the conversion artefact;
- **reproducibility** -- two machines must build the same 1,500 jobs from the same
  frozen manifest (P-016).

The pieces that shell out to the RVC checkout are tested through their inputs and
their preflight: :func:`_write_filelist` is the stage upstream has no CLI for, so
it is reproduced in this repo and has to be checked here rather than trusted.

Pure pandas and tmp_path -- no audio, no torch, no RVC checkout.
"""

import json
import os

import pandas as pd
import pytest

from src.data import rvc_generation as rvc
from src.data.ethics_gate import EthicsGateError
from src.data.pilot_jobs import PilotError


def _pools(n_train: int = 4) -> pd.DataFrame:
    rows = [{"speaker": f"trn{i}", "pool": "train", "source": "mucs2021"} for i in range(n_train)]
    rows += [{"speaker": f"evl{i}", "pool": "eval", "source": "mucs2021"} for i in range(2)]
    rows += [{"speaker": f"adp{i}", "pool": "adaptation", "source": "mucs2021"} for i in range(2)]
    return pd.DataFrame(rows)


def _index(n_train: int = 4, clips_per_speaker: int = 60, seconds: float = 6.0) -> pd.DataFrame:
    rows = []
    for prefix, count in (("trn", n_train), ("evl", 2), ("adp", 2)):
        for i in range(count):
            for j in range(clips_per_speaker):
                rows.append(
                    {
                        "utt_id": f"{prefix}{i}_{j:04d}",
                        "speaker": f"{prefix}{i}",
                        "source": "mucs2021",
                        "wav_path": f"/data/raw/mucs2021/{prefix}{i}.wav",
                        "start_seconds": j * seconds,
                        "end_seconds": (j + 1) * seconds,
                        "duration_seconds": seconds,
                        "transcript": f"clip {j} spoken by {prefix}{i}",
                    }
                )
    return pd.DataFrame(rows)


def _config(**kwargs) -> rvc.RVCConfig:
    defaults = {"n_targets": 3, "conversions_per_target": 10, "min_training_seconds": 60.0}
    return rvc.RVCConfig(**{**defaults, **kwargs})


# --------------------------------------------------------------------------- #
# Target selection
# --------------------------------------------------------------------------- #
def test_targets_come_only_from_the_train_pool() -> None:
    targets = rvc.select_target_speakers(_index(), _pools(), _config(n_targets=99))
    assert set(targets["speaker"]) == {"trn0", "trn1", "trn2", "trn3"}


def test_targets_below_the_audio_threshold_are_dropped() -> None:
    # 60 clips x 6 s = 360 s per speaker; a 400 s threshold excludes everyone.
    targets = rvc.select_target_speakers(_index(), _pools(), _config(min_training_seconds=400.0))
    assert targets.empty


def test_target_order_is_stable_across_ties() -> None:
    # Every speaker has identical total audio, so only the tie-break decides order.
    index, pools, config = _index(), _pools(), _config(n_targets=99)
    first = rvc.select_target_speakers(index, pools, config)
    second = rvc.select_target_speakers(index.sample(frac=1, random_state=7), pools, config)
    assert list(first["speaker"]) == list(second["speaker"])


def test_training_clips_are_capped_at_the_configured_budget() -> None:
    clips = rvc.gather_training_clips(_index(), "trn0", _config(max_training_seconds=30.0))
    assert clips["duration_seconds"].sum() <= 30.0
    assert list(clips["utt_id"]) == [f"trn0_{j:04d}" for j in range(5)]


def test_training_clips_keep_one_clip_even_under_an_impossible_cap() -> None:
    clips = rvc.gather_training_clips(_index(), "trn0", _config(max_training_seconds=0.5))
    assert len(clips) == 1


# --------------------------------------------------------------------------- #
# Job table + the firewall
# --------------------------------------------------------------------------- #
def test_job_table_hits_the_requested_size() -> None:
    jobs = rvc.build_rvc_jobs(_index(), _pools(), config=_config())
    assert len(jobs) == 30
    assert jobs["target_speaker"].value_counts().to_dict() == {"trn0": 10, "trn1": 10, "trn2": 10}
    assert list(jobs.columns) == rvc.JOB_COLUMNS


def test_every_endpoint_is_a_train_pool_speaker() -> None:
    jobs = rvc.build_rvc_jobs(_index(), _pools(), config=_config())
    train = {"trn0", "trn1", "trn2", "trn3"}
    assert set(jobs["target_speaker"]) <= train
    assert set(jobs["source_speaker"]) <= train
    assert set(jobs["pool"]) == {"train"}


def test_no_speaker_is_converted_into_themselves() -> None:
    jobs = rvc.build_rvc_jobs(_index(), _pools(), config=_config())
    assert not (jobs["target_speaker"] == jobs["source_speaker"]).any()


def test_no_source_clip_is_used_twice_for_one_target() -> None:
    jobs = rvc.build_rvc_jobs(_index(), _pools(), config=_config())
    assert not jobs[["target_speaker", "source_utt_id"]].duplicated().any()


def test_sources_are_spread_across_speakers_not_taken_from_one() -> None:
    jobs = rvc.build_rvc_jobs(_index(), _pools(), config=_config())
    per_target = jobs.groupby("target_speaker")["source_speaker"].nunique()
    assert (per_target == 3).all()  # the other three train-pool speakers, each used


def test_targets_read_different_windows_of_the_source_pool() -> None:
    """Twelve targets over the same 125 sentences would be one corpus, not twelve."""
    jobs = rvc.build_rvc_jobs(_index(), _pools(), config=_config())
    per_target = jobs.groupby("target_speaker")["source_utt_id"].apply(set)
    assert per_target["trn0"] != per_target["trn1"]


def test_job_table_is_identical_on_a_reshuffled_manifest() -> None:
    index, pools, config = _index(), _pools(), _config()
    first = rvc.build_rvc_jobs(index, pools, config=config)
    second = rvc.build_rvc_jobs(index.sample(frac=1, random_state=3), pools, config=config)
    pd.testing.assert_frame_equal(first, second)


def test_out_dir_reaches_the_output_paths() -> None:
    jobs = rvc.build_rvc_jobs(_index(), _pools(), out_dir="/kaggle/working/rvc", config=_config())
    assert jobs["output_path"].str.startswith("/kaggle/working/rvc/rvc_").all()


def test_short_source_pool_is_reported_not_silently_truncated(capsys) -> None:
    jobs = rvc.build_rvc_jobs(
        _index(clips_per_speaker=4),
        _pools(),
        config=_config(conversions_per_target=50, min_training_seconds=20.0),
    )
    # Three other train-pool speakers with four clips each: 12 available, 50 asked.
    assert jobs.groupby("target_speaker").size().max() == 12
    assert "[short]" in capsys.readouterr().out


def test_no_qualifying_target_is_an_error_not_an_empty_table() -> None:
    with pytest.raises(PilotError, match="fit an RVC model"):
        rvc.build_rvc_jobs(_index(), _pools(), config=_config(min_training_seconds=1e6))


def test_no_usable_source_clip_is_an_error() -> None:
    with pytest.raises(PilotError, match="nothing to convert"):
        rvc.build_rvc_jobs(_index(), _pools(), config=_config(min_source_seconds=1e6))


# --------------------------------------------------------------------------- #
# The invariants, checked directly
# --------------------------------------------------------------------------- #
def test_an_eval_pool_source_is_refused() -> None:
    jobs = rvc.build_rvc_jobs(_index(), _pools(), config=_config())
    jobs.loc[0, "source_speaker"] = "evl0"
    with pytest.raises(PilotError, match="non-train-pool source_speaker"):
        rvc.assert_rvc_invariants(jobs, _pools())


def test_an_eval_pool_target_is_refused() -> None:
    jobs = rvc.build_rvc_jobs(_index(), _pools(), config=_config())
    jobs.loc[0, "target_speaker"] = "evl0"
    with pytest.raises(PilotError, match="non-train-pool target_speaker"):
        rvc.assert_rvc_invariants(jobs, _pools())


def test_self_conversion_is_refused() -> None:
    jobs = rvc.build_rvc_jobs(_index(), _pools(), config=_config())
    jobs.loc[0, "source_speaker"] = jobs.loc[0, "target_speaker"]
    with pytest.raises(PilotError, match="into themselves"):
        rvc.assert_rvc_invariants(jobs, _pools())


def test_duplicate_target_source_pair_is_refused() -> None:
    jobs = rvc.build_rvc_jobs(_index(), _pools(), config=_config())
    jobs.loc[1, ["target_speaker", "source_speaker", "source_utt_id"]] = jobs.loc[
        0, ["target_speaker", "source_speaker", "source_utt_id"]
    ].values
    with pytest.raises(PilotError, match="duplicate"):
        rvc.assert_rvc_invariants(jobs, _pools())


def test_summary_reports_the_shape_of_the_run() -> None:
    summary = rvc.rvc_job_summary(rvc.build_rvc_jobs(_index(), _pools(), config=_config()))
    assert "30 RVC job(s) across 3 target(s)" in summary
    assert "per target: min 10, max 10" in summary


# --------------------------------------------------------------------------- #
# The ethics gate -- checked here as well as in test_ethics_gate.py
# --------------------------------------------------------------------------- #
def test_training_is_blocked_without_a_signed_note(closed_ethics_gate, tmp_path) -> None:
    with pytest.raises(EthicsGateError):
        rvc.train_speaker_model("trn0", _index(), str(tmp_path), str(tmp_path))


def test_conversion_is_blocked_without_a_signed_note(closed_ethics_gate, tmp_path) -> None:
    jobs = rvc.build_rvc_jobs(_index(), _pools(), config=_config())
    inferencer = rvc.RVCInferencer(repo_dir=str(tmp_path), python="python", config=_config())
    with pytest.raises(EthicsGateError):
        rvc.convert_batch(jobs, inferencer, {}, metadata_path=str(tmp_path / "meta.jsonl"))


# --------------------------------------------------------------------------- #
# Toolchain preflight + the stage upstream has no CLI for
# --------------------------------------------------------------------------- #
def test_asset_preflight_names_the_missing_file_and_the_fix(tmp_path) -> None:
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "train.py").write_text("")
    with pytest.raises(rvc.RVCError) as excinfo:
        rvc.assert_rvc_assets(str(tmp_path))
    message = str(excinfo.value)
    assert "assets/hubert_base/config.json" in message
    assert "logs/mute/0_gt_wavs" in message  # not optional: filelist.txt references it
    assert "hf download lj1995/VoiceConversionWebUI" in message


def test_asset_preflight_rejects_a_non_rvc_directory(tmp_path) -> None:
    with pytest.raises(rvc.RVCError, match="does not look like an RVC-WebUI checkout"):
        rvc.assert_rvc_assets(str(tmp_path))


def _fake_experiment(tmp_path, n_slices: int = 3):
    """A checkout and an experiment dir with all four extraction stages present."""
    repo = tmp_path / "rvc"
    exp = repo / "logs" / "exp"
    for sub in ("0_gt_wavs", "3_feature768", "2a_f0", "2b-f0nsf"):
        (exp / sub).mkdir(parents=True)
    for i in range(n_slices):
        (exp / "0_gt_wavs" / f"0_{i}.wav").write_text("")
        (exp / "3_feature768" / f"0_{i}.npy").write_text("")
        (exp / "2a_f0" / f"0_{i}.wav.npy").write_text("")
        (exp / "2b-f0nsf" / f"0_{i}.wav.npy").write_text("")
    template = repo / "configs" / "v1" / "40k.json"
    template.parent.mkdir(parents=True)
    template.write_text(json.dumps({"train": {"log_interval": 200}, "speaker_info": ["stale"]}))
    return repo, exp


def test_filelist_lists_every_slice_plus_the_two_mute_lines(tmp_path) -> None:
    repo, exp = _fake_experiment(tmp_path, n_slices=3)
    kept = rvc._write_filelist(exp, repo, rvc.RVCConfig(), seed=0)
    lines = (exp / "filelist.txt").read_text(encoding="utf-8").splitlines()
    assert kept == 3
    assert len(lines) == 5
    assert sum("logs/mute" in line.replace("\\", "/") for line in lines) == 2
    assert all(line.endswith("|0") for line in lines)  # single-speaker id


def test_filelist_takes_the_v1_template_for_40k_even_on_v2(tmp_path) -> None:
    """``configs/v2/40k.json`` does not exist upstream; ``click_train`` knows this."""
    repo, exp = _fake_experiment(tmp_path)
    rvc._write_filelist(exp, repo, rvc.RVCConfig(sample_rate="40k", version="v2"), seed=0)
    written = json.loads((exp / "config.json").read_text(encoding="utf-8"))
    assert written["train"]["log_interval"] == 200
    assert "speaker_info" not in written  # single-speaker model


def test_filelist_is_the_same_for_the_same_seed(tmp_path) -> None:
    repo, exp = _fake_experiment(tmp_path, n_slices=6)
    rvc._write_filelist(exp, repo, rvc.RVCConfig(), seed=11)
    first = (exp / "filelist.txt").read_text(encoding="utf-8")
    rvc._write_filelist(exp, repo, rvc.RVCConfig(), seed=11)
    assert (exp / "filelist.txt").read_text(encoding="utf-8") == first


def test_empty_extraction_output_fails_before_training_starts(tmp_path) -> None:
    repo, exp = _fake_experiment(tmp_path, n_slices=0)
    with pytest.raises(rvc.RVCError, match="nothing to train on"):
        rvc._write_filelist(exp, repo, rvc.RVCConfig(), seed=0)


def test_inferencer_without_a_known_checkout_says_so(monkeypatch) -> None:
    monkeypatch.delenv("RVC_REPO", raising=False)
    monkeypatch.setattr(rvc, "_LAST_RVC_REPO", None)
    with pytest.raises(rvc.RVCError, match="no RVC checkout known"):
        rvc.load_rvc_inferencer()


def test_conversion_command_disables_the_index_when_there_is_none(tmp_path) -> None:
    inferencer = rvc.RVCInferencer(repo_dir=str(tmp_path), python="python", config=_config())
    model = rvc.RVCModel(
        speaker="trn0",
        experiment="rvc_trn0",
        model_path="/w/rvc_trn0.pth",
        index_path="",
        repo_dir=str(tmp_path),
        epochs=20,
        n_clips=5,
        train_seconds=300.0,
        sample_rate="40k",
        version="v2",
    )
    command = rvc._convert_command(inferencer, model, tmp_path / "in", tmp_path / "out")
    assert "--index" not in command
    assert command[command.index("--index-rate") + 1] == "0"
    assert "rvc_trn0" in model.describe() and "no index" in model.describe()


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def test_stats_count_targets_and_sources_separately() -> None:
    records = [
        {
            "tool": "rvc",
            "speaker": "trn0",
            "source_speaker": "trn1",
            "source_utt_id": f"trn1_{i:04d}",
            "pool": "train",
            "language": "hi",
        }
        for i in range(3)
    ] + [
        {
            "tool": "rvc",
            "speaker": "trn1",
            "source_speaker": "trn2",
            "source_utt_id": "trn2_0000",
            "pool": "train",
            "language": "hi",
        }
    ]
    stats = rvc.rvc_generation_stats(records)
    assert stats["total"] == 4
    assert stats["n_targets"] == 2
    assert stats["n_source_speakers"] == 2
    assert stats["per_target_min"] == 1
    assert stats["per_target_max"] == 3
    assert stats["unique_source_clips"] == 4
    assert stats["by_pool"] == {"train": 4}


def test_metadata_reconciliation_is_shared_with_the_cm01_log(tmp_path) -> None:
    """CM01 and CM02 logs are read by one code path, so this must hold for both."""
    log = tmp_path / "meta.jsonl"
    gone = tmp_path / "gone.wav"
    kept = tmp_path / "kept.wav"
    kept.write_text("")
    log.write_text(
        json.dumps({"output_path": str(gone), "tool": "rvc"})
        + "\n"
        + json.dumps({"output_path": str(kept), "tool": "rvc"})
        + "\n",
        encoding="utf-8",
    )
    assert rvc.rewrite_metadata(str(log)) == 1
    assert [r["output_path"] for r in rvc.read_metadata(str(log))] == [str(kept)]


# --------------------------------------------------------------------------- #
# Stage subprocess environment
# --------------------------------------------------------------------------- #
def test_stage_env_puts_the_checkout_root_on_the_path(tmp_path) -> None:
    """Without this the first stage dies on ``No module named 'infer'``."""
    env = rvc._stage_env(str(tmp_path))
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(tmp_path.resolve())


def test_stage_env_disables_the_script_directory_prepend(tmp_path) -> None:
    """``<repo>/train/train.py`` shadows the ``train`` package if it is on the path.

    The symptom is a circular import of a partially initialised ``train``, three
    stages in and after the wavs have been staged.
    """
    assert rvc._stage_env(str(tmp_path))["PYTHONSAFEPATH"] == "1"


def test_stage_env_keeps_an_existing_pythonpath_without_duplicating_the_root(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path.resolve()), "/some/where"]))
    entries = rvc._stage_env(str(tmp_path))["PYTHONPATH"].split(os.pathsep)
    assert entries == [str(tmp_path.resolve()), "/some/where"]
