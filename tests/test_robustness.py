"""The robustness sweep's transforms: the pre-registered list, and that each does what it says."""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest

from src.data import robustness as rb

SR = rb.SAMPLE_RATE
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _voice(seconds: float = 2.0, seed: int = 0) -> np.ndarray:
    """A voiced-like test signal: a 140 Hz pulse train shaped by two formants, plus breath."""
    t = np.arange(int(seconds * SR)) / SR
    rng = np.random.default_rng(seed)
    pulse = np.sign(np.sin(2 * np.pi * 140 * t)) * 0.1
    formants = 0.2 * np.sin(2 * np.pi * 700 * t) + 0.1 * np.sin(2 * np.pi * 1_200 * t)
    return (
        pulse
        + formants * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
        + 0.01 * rng.standard_normal(t.size)
    ).astype(np.float32)


def test_the_list_is_the_pre_registered_one() -> None:
    assert [t.name for t in rb.TRANSFORMS] == [
        "mp3_32k",
        "mp3_64k",
        "opus_12k",
        "white_10db",
        "white_20db",
        "babble_10db",
        "babble_20db",
        "reverb",
        "tempo_up_5",
        "tempo_down_5",
        "pitch_up_1",
        "pitch_down_1",
        "aac_recorder",
        "g711_second",
    ]
    after = {t.name for t in rb.TRANSFORMS if t.where == "after"}
    assert after == {"aac_recorder", "g711_second"}
    assert {t.where for t in rb.TRANSFORMS} == {"before", "after"}


def test_white_noise_lands_at_its_snr() -> None:
    from src.data.channel_sim import measured_snr_db

    clean = _voice()
    noisy = rb.BY_NAME["white_10db"].apply(clean, np.random.default_rng(1), None)
    assert measured_snr_db(clean, noisy) == pytest.approx(10.0, abs=0.3)


def test_babble_needs_its_voices_and_lands_at_its_snr() -> None:
    from src.data.channel_sim import measured_snr_db

    clean = _voice()
    with pytest.raises(ValueError, match="babble"):
        rb.BY_NAME["babble_20db"].apply(clean, np.random.default_rng(1), None)
    voices = _voice(3.0, seed=5) + _voice(3.0, seed=6)
    mixed = rb.BY_NAME["babble_20db"].apply(clean, np.random.default_rng(1), voices)
    assert measured_snr_db(clean, mixed) == pytest.approx(20.0, abs=0.3)


def test_reverb_is_seeded_keeps_length_and_level() -> None:
    clean = _voice()
    a = rb.reverb(clean, rb.clip_rng("x.wav", "reverb"), None)
    b = rb.reverb(clean, rb.clip_rng("x.wav", "reverb"), None)
    c = rb.reverb(clean, rb.clip_rng("y.wav", "reverb"), None)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert a.size == clean.size
    assert np.sqrt(np.mean(a**2)) == pytest.approx(np.sqrt(np.mean(clean**2)), rel=1e-3)
    impulse = rb.room_impulse(np.random.default_rng(0))
    assert impulse[0] == 1.0 and impulse.size == int(rb.REVERB_RT60_S * SR)


def test_babble_never_uses_the_clips_own_speaker() -> None:
    pool = pd.DataFrame(
        {
            "filepath": [f"s{i}_{j}.wav" for i in range(8) for j in range(2)],
            "speaker": [f"s{i}" for i in range(8) for _ in range(2)],
        }
    )
    used: list[str] = []

    def load(path: str) -> np.ndarray:
        used.append(path.split("_")[0])
        return np.ones(SR, dtype=np.float32)

    mix = rb.babble_for("clip.wav", "s3", pool, load)
    assert len(used) == rb.BABBLE_VOICES and "s3" not in used and len(set(used)) == len(used)
    assert mix.size == SR


def test_through_line_with_no_transform_is_the_bundles_clip() -> None:
    pytest.importorskip("librosa")  # resampling, not installed in CI
    from src.data.channel_bundle import clip_seed
    from src.data.channel_sim import ChannelConfig, simulate_channel

    clean = _voice()
    bundle = simulate_channel(
        clean, ChannelConfig(codec="g711", snr_db=20.0, seed=clip_seed("a.wav", 1234))
    )
    assert np.array_equal(rb.through_line(clean, "a.wav", None), bundle)
    after = rb.through_line(clean, "a.wav", rb.BY_NAME["g711_second"])
    assert after.size == bundle.size and not np.array_equal(after, bundle)


@needs_ffmpeg
@pytest.mark.parametrize("name", ["mp3_32k", "opus_12k", "aac_recorder"])
def test_codecs_round_trip_to_the_same_length(name: str) -> None:
    clean = _voice()
    out = rb.BY_NAME[name].apply(clean, np.random.default_rng(0), None)
    assert abs(out.size - clean.size) <= 0.06 * SR  # encoder padding, at most 60 ms
    assert np.max(np.abs(out)) > 0.05


@needs_ffmpeg
def test_tempo_changes_length_and_pitch_keeps_it() -> None:
    clean = _voice(4.0)
    faster = rb.BY_NAME["tempo_up_5"].apply(clean, np.random.default_rng(0), None)
    higher = rb.BY_NAME["pitch_up_1"].apply(clean, np.random.default_rng(0), None)
    assert faster.size == pytest.approx(clean.size / 1.05, rel=0.02)
    assert higher.size == pytest.approx(clean.size, rel=0.02)


def test_the_plan_is_the_frozen_clip_list() -> None:
    plan = pd.read_csv(
        rb.Path(__file__).resolve().parents[1] / "data/manifests/robustness_plan.csv"
    )
    assert plan.equals(rb.build_plan(rb.Path(__file__).resolve().parents[1]).astype(plan.dtypes))
    counts = plan.groupby(["set", "label", "tool"]).size().to_dict()
    assert counts == {
        ("indicf5", "bonafide", "none"): 300,
        ("indicf5", "spoof", "indicf5"): 300,
        ("new_speakers", "bonafide", "none"): 300,
        ("new_speakers", "spoof", "tortoise"): 222,
        ("new_speakers", "spoof", "xtts_v2"): 222,
    }
