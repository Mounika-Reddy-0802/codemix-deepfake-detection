"""Tests for generation-at-scale job assembly + stats (Week 3, owner L)."""

from src.data import spoof_generation as sg


def test_build_clone_jobs_pairs_and_tags_pool() -> None:
    refs = {"spk_a": "a.wav", "spk_b": "b.wav"}
    pools = {"spk_a": "adaptation", "spk_b": "eval"}
    transcripts = [("spk_a", "hello dost"), ("spk_b", "kya haal hai"), ("spk_a", "chalo")]
    jobs = sg.build_clone_jobs(refs, transcripts, pools, out_dir="out")
    assert len(jobs) == 3
    assert jobs[0].reference_wav == "a.wav"
    assert jobs[1].pool == "eval"
    assert all(j.tool == "xtts_v2" for j in jobs)
    assert len({j.output_path for j in jobs}) == 3  # unique paths


def test_build_clone_jobs_skips_missing_ref_or_pool() -> None:
    refs = {"spk_a": "a.wav"}
    pools = {"spk_a": "adaptation"}
    transcripts = [("spk_a", "ok"), ("spk_unknown", "skip me")]
    jobs = sg.build_clone_jobs(refs, transcripts, pools, out_dir="out")
    assert len(jobs) == 1
    assert jobs[0].speaker == "spk_a"


def test_build_clone_jobs_respects_target() -> None:
    refs = {"s": "s.wav"}
    pools = {"s": "eval"}
    transcripts = [("s", f"line {i}") for i in range(100)]
    jobs = sg.build_clone_jobs(refs, transcripts, pools, out_dir="o", n_target=10)
    assert len(jobs) == 10


def test_generation_stats() -> None:
    records = [
        {"tool": "xtts_v2", "language": "hi", "pool": "adaptation", "speaker": "a"},
        {"tool": "xtts_v2", "language": "hi", "pool": "adaptation", "speaker": "a"},
        {"tool": "xtts_v2", "language": "ta", "pool": "eval", "speaker": "b"},
    ]
    stats = sg.generation_stats(records)
    assert stats["total"] == 3
    assert stats["by_tool"] == {"xtts_v2": 3}
    assert stats["by_language"] == {"hi": 2, "ta": 1}
    assert stats["n_speakers"] == 2
    assert stats["per_speaker_max"] == 2


def test_generation_stats_empty() -> None:
    stats = sg.generation_stats([])
    assert stats["total"] == 0
    assert stats["n_speakers"] == 0
