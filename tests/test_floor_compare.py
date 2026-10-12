"""Tests for the paired detector-versus-floor comparison (src/training/floor_compare.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.training import floor_compare as fc


def _two_class_speakers(n_speakers: int = 10, per_class: int = 20, seed: int = 0):
    rng = np.random.default_rng(seed)
    labels, speakers = [], []
    for s in range(n_speakers):
        labels += [1] * per_class + [0] * per_class
        speakers += [f"spk{s}"] * (2 * per_class)
    labels = np.asarray(labels)
    return labels, np.asarray(speakers), rng


def test_perfect_detector_beats_a_random_floor():
    labels, speakers, rng = _two_class_speakers()
    detector = labels + 0.01 * rng.standard_normal(len(labels))  # separates perfectly
    floor = rng.standard_normal(len(labels))  # chance
    stats = fc.speaker_bootstrap(labels, speakers, detector, floor, n_boot=300)
    assert stats["eer_a"] == 0.0
    assert stats["diff_ci"][1] < 0
    assert fc.verdict(stats["diff_ci"]) == fc.BEATS_FLOOR
    assert stats["n_speakers"] == 10


def test_resamples_are_paired_so_a_monotone_copy_has_zero_difference():
    # EER depends only on the ranking, so a monotone transform of the same scores has
    # the same EER on ANY set of clips. If the two systems were resampled separately,
    # the differences would scatter; paired, every one is exactly zero.
    labels, speakers, rng = _two_class_speakers()
    scores = labels + rng.standard_normal(len(labels))
    stats = fc.speaker_bootstrap(labels, speakers, scores, 3.0 * scores + 7.0, n_boot=200)
    assert stats["diff"] == 0.0
    assert stats["diff_ci"] == [0.0, 0.0]
    assert fc.verdict(stats["diff_ci"]) == fc.INCONCLUSIVE


def test_worse_detector_is_called_worse():
    labels, speakers, rng = _two_class_speakers()
    floor = labels + 0.01 * rng.standard_normal(len(labels))
    detector = rng.standard_normal(len(labels))
    stats = fc.speaker_bootstrap(labels, speakers, detector, floor, n_boot=300)
    assert stats["diff_ci"][0] > 0
    assert fc.verdict(stats["diff_ci"]) == fc.WORSE_THAN_FLOOR


def test_speaker_interval_is_wider_than_one_from_independent_clips():
    # Speaker effects make clips within a speaker correlated: the speaker bootstrap
    # must reflect that and not report the narrow interval clip resampling would.
    rng = np.random.default_rng(1)
    labels, speakers = [], []
    scores = []
    for s in range(12):
        shift = rng.normal(0, 1.5)  # a per-speaker offset on the fakes only
        for y in (1, 0):
            n = 40
            labels += [y] * n
            speakers += [f"s{s}"] * n
            scores += list(rng.normal(1.0 if y else shift, 1.0, n))
    labels, speakers, scores = np.asarray(labels), np.asarray(speakers), np.asarray(scores)
    floor = rng.standard_normal(len(labels))
    by_speaker = fc.speaker_bootstrap(labels, speakers, scores, floor, n_boot=400)
    # clip-level: give every clip its own "speaker"
    by_clip = fc.speaker_bootstrap(
        labels, np.arange(len(labels)).astype(str), scores, floor, n_boot=400
    )
    width = lambda ci: ci[1] - ci[0]  # noqa: E731
    assert width(by_speaker["eer_a_ci"]) > width(by_clip["eer_a_ci"])


def test_single_class_speakers_are_skipped_not_fatal():
    labels = np.array([1] * 10 + [0] * 10)
    speakers = np.array(["a"] * 10 + ["b"] * 10)  # each speaker holds one class
    scores = labels.astype(float)
    stats = fc.speaker_bootstrap(labels, speakers, scores, scores, n_boot=100, seed=3)
    assert stats["n_boot_skipped"] > 0
    assert stats["n_boot_used"] + stats["n_boot_skipped"] == 100


def test_a_resample_set_that_always_loses_a_class_raises():
    labels = np.array([1] * 5 + [0] * 5)
    speakers = np.array(["only"] * 10)
    with pytest.raises(ValueError):
        # one speaker holding both classes never loses one, so make it single-class
        fc.speaker_bootstrap(np.ones(10, dtype=int), speakers, labels, labels, n_boot=20)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        fc.speaker_bootstrap(
            np.array([1, 0]), np.array(["a", "a"]), np.array([0.1, 0.2]), np.array([0.1])
        )


def test_p_value_is_small_for_a_clear_win_and_large_for_a_tie():
    labels, speakers, rng = _two_class_speakers()
    good = labels + 0.01 * rng.standard_normal(len(labels))
    noise = rng.standard_normal(len(labels))
    win = fc.speaker_bootstrap(labels, speakers, good, noise, n_boot=300)
    tie = fc.speaker_bootstrap(labels, speakers, noise, noise, n_boot=300)
    assert win["p_value"] < 0.01
    assert tie["p_value"] == 1.0


def test_verdict_boundaries():
    assert fc.verdict([-0.05, -0.01]) == fc.BEATS_FLOOR
    assert fc.verdict([0.01, 0.05]) == fc.WORSE_THAN_FLOOR
    assert fc.verdict([-0.01, 0.01]) == fc.INCONCLUSIVE
    assert fc.verdict([0.0, 0.0]) == fc.INCONCLUSIVE


def test_per_speaker_eer_handles_a_one_class_speaker():
    labels = np.array([1, 1, 0, 0, 1, 1])
    speakers = np.array(["a", "a", "a", "a", "b", "b"])
    scores = np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.6])
    table = fc.per_speaker_eer(labels, speakers, scores).set_index("speaker")
    assert table.loc["a", "eer"] == 0.0
    assert np.isnan(table.loc["b", "eer"])
    assert table.loc["b", "n_spoof"] == 0


def test_binary_labels():
    out = fc.binary_labels(pd.Series(["bonafide", "spoof", "Bonafide", "SPOOF"]))
    assert out.tolist() == [1, 0, 1, 0]


def test_pair_scores_joins_by_filepath_not_position():
    test_rows = pd.DataFrame(
        {
            "filepath": ["x", "y", "z"],
            "label": ["bonafide", "spoof", "spoof"],
            "speaker": ["1", "1", "2"],
        }
    )
    detector = pd.DataFrame({"filepath": ["z", "x", "y"], "score": [0.3, 0.9, 0.2]})
    paired = fc.pair_scores(test_rows, np.array([0.5, 0.6, 0.7]), detector)
    assert paired.set_index("filepath").loc["x", "detector"] == 0.9
    assert paired.set_index("filepath").loc["z", "floor"] == 0.7


def test_pair_scores_refuses_different_clip_sets():
    test_rows = pd.DataFrame(
        {"filepath": ["x", "y"], "label": ["bonafide", "spoof"], "speaker": ["1", "1"]}
    )
    detector = pd.DataFrame({"filepath": ["x", "q"], "score": [0.9, 0.2]})
    with pytest.raises(ValueError):
        fc.pair_scores(test_rows, np.array([0.5, 0.6]), detector)


def _fake_audio(monkeypatch, unreadable: str | None = None):
    """Replace the wav loader: bonafide clips are loud, spoofs quiet; one path can fail."""
    from src.utils import audio_utils

    def load_wav(path, target_sr=16_000):
        if unreadable and unreadable in path:
            raise OSError("unreadable")
        rng = np.random.default_rng(abs(hash(path)) % (2**32))
        amp = 0.5 if "bona" in path else 0.05
        return (amp * rng.standard_normal(target_sr)).astype(np.float32), target_sr

    monkeypatch.setattr(audio_utils, "load_wav", load_wav)


def test_extract_reports_which_rows_it_kept(monkeypatch):
    from src.data import lowlevel_cue as lc

    _fake_audio(monkeypatch, unreadable="broken")
    manifest = pd.DataFrame(
        {
            "filepath": ["a_bona.wav", "broken.wav", "c_spoof.wav"],
            "label": ["bonafide", "spoof", "spoof"],
        }
    )
    x, y, kept = lc.extract(manifest, "root", return_index=True)
    assert kept.tolist() == [0, 2]
    assert x.shape == (2, len(lc.FEATURE_NAMES))
    assert y.tolist() == [1, 0]
    x2, y2 = lc.extract(manifest, "root")  # the two-value form is unchanged
    assert np.array_equal(x, x2) and np.array_equal(y, y2)


def test_run_end_to_end_on_synthetic_audio(monkeypatch, tmp_path):
    _fake_audio(monkeypatch)
    manifests = tmp_path / "manifests"
    manifests.mkdir()

    def manifest(n_speakers, per_class, prefix):
        rows = []
        for s in range(n_speakers):
            for k in range(per_class):
                rows.append(
                    {
                        "filepath": f"{prefix}_{s}_{k}_bona.wav",
                        "label": "bonafide",
                        "speaker": f"{prefix}{s}",
                    }
                )
                rows.append(
                    {
                        "filepath": f"{prefix}_{s}_{k}_spoof.wav",
                        "label": "spoof",
                        "speaker": f"{prefix}{s}",
                    }
                )
        return pd.DataFrame(rows)

    manifest(4, 6, "train").to_csv(manifests / "train.csv", index=False)
    for test_set, (clean, _) in fc.TEST_SETS.items():
        test = manifest(5, 6, test_set)
        test.to_csv(manifests / clean, index=False)
        # a detector that is noisy but informative
        rng = np.random.default_rng(0)
        y = (test["label"] == "bonafide").astype(float)
        test.assign(score=y + 0.8 * rng.standard_normal(len(test)))[["filepath", "score"]].to_csv(
            tmp_path / f"det__{test_set}_scores.csv", index=False
        )

    system = fc.System("toy", "train.csv", "clean", str(tmp_path / "det"))
    results, speakers = fc.run(
        str(manifests),
        "clean-root",
        "channel-root",
        systems=(system,),
        n_boot=50,
        scores_dir=str(tmp_path / "out"),
    )
    assert len(results) == len(fc.TEST_SETS)
    for row in results:
        # loud-vs-quiet separates perfectly, so the floor reaches 0 and the noisy
        # detector cannot beat it
        assert row["eer_b"] == 0.0
        assert row["verdict"] in (fc.WORSE_THAN_FLOOR, fc.INCONCLUSIVE)
        assert row["clips"] == 60
    assert set(speakers["test_set"]) == set(fc.TEST_SETS)
    assert (tmp_path / "out" / "det__cm04_paired.csv").is_file()
