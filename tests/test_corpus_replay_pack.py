"""Corpus replay playback files: held-out speakers only, chosen without scores, one level."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import corpus_replay_pack as pack


def _clean() -> pd.DataFrame:
    rows = []
    for kind, n, sets, label, tool in pack.KINDS:
        for k in range(n + 2):
            for j in range(3):
                rows.append((f"x/{kind}_{k}_{j}.wav", label, tool, sets[0], "test", f"{kind}{k}"))
    rows.append(("x/train.wav", "spoof", "indicf5", "cm07", "test", "t1"))
    rows.append(("x/dev.wav", "spoof", "indicf5", "cm07", "dev", "d1"))
    return pd.DataFrame(rows, columns=["filepath", "label", "tool", "set", "split", "spk"])


def test_only_held_out_test_clips_are_eligible():
    clean = _clean()
    speakers = dict(zip(clean["filepath"].str.split("/").str[-1], clean["spk"], strict=True))
    d = pack.eligible(clean.drop(columns="spk"), speakers, {"t1"}, lambda p: 5.0)
    assert "x/train.wav" not in set(d["filepath"]) and "x/dev.wav" not in set(d["filepath"])


def test_the_choice_is_seeded_one_speaker_per_file_and_shuffled():
    clean = _clean()
    speakers = dict(zip(clean["filepath"].str.split("/").str[-1], clean["spk"], strict=True))
    clips = pack.eligible(clean.drop(columns="spk"), speakers, set(), lambda p: 5.0)
    plan = pack.choose(clips)
    assert plan.equals(pack.choose(clips))
    assert len(plan) == sum(k[1] for k in pack.KINDS) == 30
    assert list(plan["file"]) == [f"R{i:02d}.wav" for i in range(1, 31)]
    assert plan.groupby("kind")["speaker"].nunique().to_dict() == {k[0]: k[1] for k in pack.KINDS}
    assert list(plan["kind"][:6]) != ["genuine"] * 6


def test_too_few_speakers_stops():
    clean = _clean()
    clean = clean[~((clean["tool"] == "tortoise") & (clean["spk"] != "tortoise0"))]
    speakers = dict(zip(clean["filepath"].str.split("/").str[-1], clean["spk"], strict=True))
    with pytest.raises(ValueError):
        pack.choose(pack.eligible(clean.drop(columns="spk"), speakers, set(), lambda p: 5.0))


def test_speakers_with_too_little_audio_are_not_drawn():
    clean = _clean()
    speakers = dict(zip(clean["filepath"].str.split("/").str[-1], clean["spk"], strict=True))
    short = {"x/indicf5_0_0.wav", "x/indicf5_0_1.wav", "x/indicf5_0_2.wav"}
    clips = pack.eligible(
        clean.drop(columns="spk"), speakers, set(), lambda p: 1.0 if p in short else 5.0
    )
    assert "indicf50" not in set(pack.choose(clips)["speaker"])


def test_join_resamples_stops_at_the_target_and_sets_one_level():
    def read(path):
        sr = 24_000 if "loud" in path else 16_000
        amp = 0.5 if "loud" in path else 0.01
        return amp * np.sin(np.arange(sr * 10) / 7.0), sr

    a = pack.join(["loud.wav", "loud2.wav", "loud3.wav", "loud4.wav"], read)
    b = pack.join(["quiet.wav", "quiet2.wav", "quiet3.wav"], read)
    assert len(a) == len(b) == int(3 * 10 * pack.SR + 2 * pack.GAP_S * pack.SR)
    level = [20 * np.log10(np.sqrt(np.mean(x**2))) for x in (a, b)]
    assert level == pytest.approx([pack.TARGET_RMS_DBFS] * 2, abs=0.01)
