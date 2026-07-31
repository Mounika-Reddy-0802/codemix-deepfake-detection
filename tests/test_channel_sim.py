"""CRITICAL-PATH test (Week 2, owner L): telephony channel-simulation sanity.

Stubbed at bootstrap so the CI structure exists. The real assertions land in
Week 2 when ``src/data/channel_sim.py`` is implemented. Keep this file — the
channel-sim chain is core to the channel-matched protocol.
"""

import pytest


@pytest.mark.skip(reason="TODO(week-2, L): implement once channel_sim.py exists")
def test_output_sample_rate_is_16k() -> None:
    """The chain must return audio at 16 kHz regardless of the codec used."""


@pytest.mark.skip(reason="TODO(week-2, L): implement once channel_sim.py exists")
def test_no_clipping_after_codec_and_noise() -> None:
    """Peak amplitude must stay within [-1.0, 1.0] after codec + SNR mixing."""


@pytest.mark.skip(reason="TODO(week-2, L): implement once channel_sim.py exists")
def test_snr_levels_are_monotonic() -> None:
    """Lower target SNR must yield measurably more added noise energy."""


def test_channel_sim_module_is_importable() -> None:
    """Smoke: the module exists and imports (no heavy deps at import time)."""
    import importlib

    mod = importlib.import_module("src.data.channel_sim")
    assert mod.__doc__ is not None
