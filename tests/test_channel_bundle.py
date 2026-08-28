"""Unit tests for the channel-matched bundle renderer.

No corpus is touched: tones are written to a tmp_path bundle and rendered, which
exercises the same code path the 3,966-clip eval render uses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import channel_bundle as cb

soundfile = pytest.importorskip("soundfile")

SR = 16_000


def _tone(seconds: float = 0.5, freq: float = 440.0, sr: int = SR, hf_amp: float = 0.02):
    """A low tone plus a quiet 6 kHz partial, proportioned like speech.

    The high partial has to be there at all: a pure sine below 4 kHz has nothing
    above the narrowband ceiling to lose, so it cannot show whether the 8 kHz
    round-trip did anything. But it also has to stay *quiet*. ``measured_snr_db``
    compares clean against channel end to end, so the band the chain removes is
    counted as noise alongside the noise it adds; with half the energy above
    4 kHz the removal dominates and a 20 dB render measures 3 dB. At this
    amplitude the HF share is 0.25% -- real speech is 0.6% (W3-T2) -- and a 20 dB
    render measures 19.2 dB, which is the regime the numbers are quoted in.
    """
    t = np.arange(int(seconds * sr), dtype=np.float64) / sr
    wave = 0.4 * np.sin(2 * np.pi * freq * t) + hf_amp * np.sin(2 * np.pi * 6_000.0 * t)
    return wave.astype(np.float32)


def _clean_bundle(tmp_path, names=("a", "b", "c")) -> tuple[pd.DataFrame, str]:
    """A portable manifest plus the clips it points at."""
    clips = tmp_path / "clean" / "clips"
    clips.mkdir(parents=True)
    rows = []
    for i, name in enumerate(names):
        soundfile.write(str(clips / f"{name}.wav"), _tone(freq=300 + 120 * i), SR)
        rows.append(
            {
                "filepath": f"${{DATA_ROOT}}/clips/{name}.wav",
                "label": "bonafide" if i % 2 == 0 else "spoof",
                "speaker": f"spk{i}",
                "tool": "none",
                "condition": "clean",
                "utt_id": name,
            }
        )
    return pd.DataFrame(rows), str(tmp_path / "clean")


def test_condition_name_is_slice_friendly():
    assert cb.condition_name("g711", 20.0) == "channel_g711_20db"
    assert cb.condition_name("amr_nb", 7.5) == "channel_amr_nb_7.5db"


def test_clip_seed_is_stable_across_processes():
    """Seeds must not come from hash(): it is salted per process.

    A drifting seed would make the bundle irreproducible, which is the whole point
    of rendering it deterministically.
    """
    assert cb.clip_seed("x.wav", 1234) == cb.clip_seed("x.wav", 1234)
    assert cb.clip_seed("x.wav", 1234) != cb.clip_seed("y.wav", 1234)
    assert cb.clip_seed("x.wav", 1234) != cb.clip_seed("x.wav", 99)


def test_render_writes_one_clip_per_row_and_relabels_condition(tmp_path):
    manifest, root = _clean_bundle(tmp_path)
    out = cb.render(manifest, tmp_path / "chan", data_root=root, snr_db=20.0)

    assert len(out) == len(manifest)
    assert out["filepath"].nunique() == len(manifest)
    assert (out["condition"] == "channel_g711_20db").all()
    assert out["filepath"].str.startswith("${DATA_ROOT}/clips/").all()
    # Labels and speakers must survive untouched -- the render changes audio only.
    assert out["label"].tolist() == manifest["label"].tolist()
    assert out["speaker"].tolist() == manifest["speaker"].tolist()
    for name in ("a", "b", "c"):
        assert (tmp_path / "chan" / "clips" / f"{name}.wav").is_file()


def test_render_destroys_energy_above_the_narrowband_ceiling(tmp_path):
    """The 8 kHz round-trip must actually remove high frequencies.

    A chain that silently no-ops would leave the audio wideband and the whole
    channel-matched column would be a duplicate of the clean one.
    """
    clips = tmp_path / "clean" / "clips"
    clips.mkdir(parents=True)
    # Equal amplitudes here, unlike the speech-like default: this test is about
    # whether the band survives, so half the energy is put above the ceiling.
    soundfile.write(str(clips / "hi.wav"), _tone(freq=500.0, hf_amp=0.4), SR)
    manifest = pd.DataFrame(
        [{"filepath": "${DATA_ROOT}/clips/hi.wav", "label": "bonafide", "condition": "clean"}]
    )

    cb.render(manifest, tmp_path / "chan", data_root=str(tmp_path / "clean"), snr_db=40.0)
    rendered, sr = soundfile.read(str(tmp_path / "chan" / "clips" / "hi.wav"))

    original, _ = soundfile.read(str(clips / "hi.wav"))
    # Half the energy sits above 4 kHz before the chain and essentially none
    # after: W3-T2 measured the same collapse on real speech (0.6% -> 0.0001%).
    assert cb._hf_energy_ratio(original, SR) > 0.4
    assert cb._hf_energy_ratio(rendered, sr) < 1e-3


def test_render_gives_each_clip_its_own_noise(tmp_path):
    """A single shared noise draw would be a learnable shortcut, not a channel.

    Two clips carrying identical source audio must still differ after rendering,
    or the added noise is the same waveform everywhere.
    """
    clips = tmp_path / "clean" / "clips"
    clips.mkdir(parents=True)
    tone = _tone()
    for name in ("one", "two"):
        soundfile.write(str(clips / f"{name}.wav"), tone, SR)
    manifest = pd.DataFrame(
        [
            {"filepath": f"${{DATA_ROOT}}/clips/{n}.wav", "label": "bonafide", "condition": "clean"}
            for n in ("one", "two")
        ]
    )

    cb.render(manifest, tmp_path / "chan", data_root=str(tmp_path / "clean"), snr_db=10.0)
    a, _ = soundfile.read(str(tmp_path / "chan" / "clips" / "one.wav"))
    b, _ = soundfile.read(str(tmp_path / "chan" / "clips" / "two.wav"))

    assert not np.allclose(a, b)


def test_render_is_deterministic(tmp_path):
    manifest, root = _clean_bundle(tmp_path, names=("a",))
    cb.render(manifest, tmp_path / "one", data_root=root, snr_db=15.0)
    cb.render(manifest, tmp_path / "two", data_root=root, snr_db=15.0)

    a, _ = soundfile.read(str(tmp_path / "one" / "clips" / "a.wav"))
    b, _ = soundfile.read(str(tmp_path / "two" / "clips" / "a.wav"))
    assert np.allclose(a, b)


def test_render_resumes_instead_of_rewriting(tmp_path):
    """An interrupted render must cost only the clips it had not reached."""
    manifest, root = _clean_bundle(tmp_path)
    cb.render(manifest, tmp_path / "chan", data_root=root)
    target = tmp_path / "chan" / "clips" / "a.wav"
    marker = target.stat().st_mtime_ns

    out = cb.render(manifest, tmp_path / "chan", data_root=root)
    assert len(out) == len(manifest)
    assert target.stat().st_mtime_ns == marker  # untouched on the second pass


def test_verify_reports_the_band_limit_and_snr(tmp_path):
    manifest, root = _clean_bundle(tmp_path)
    rendered = cb.render(manifest, tmp_path / "chan", data_root=root, snr_db=20.0)

    stats = cb.verify(manifest, rendered, root, str(tmp_path / "chan"), sample=3)
    assert stats["clips"] == 3
    assert stats["hf_ratio_channel_pct"] < stats["hf_ratio_clean_pct"]
    # Target 20 dB; the chain also drops the band above 4 kHz, which this metric
    # counts as noise, so it lands slightly under. W3-T2 measured 17.6 dB on real
    # speech against the same 20 dB target.
    assert 15.0 < stats["measured_snr_db"] < 25.0


def test_render_skips_a_missing_clip_without_stopping(tmp_path):
    manifest, root = _clean_bundle(tmp_path, names=("a",))
    manifest = pd.concat(
        [
            manifest,
            pd.DataFrame([{"filepath": "${DATA_ROOT}/clips/gone.wav", "label": "spoof"}]),
        ],
        ignore_index=True,
    )
    out = cb.render(manifest, tmp_path / "chan", data_root=root)
    assert len(out) == 1
    assert out.iloc[0]["filepath"].endswith("a.wav")
