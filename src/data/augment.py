"""On-the-fly training augmentation that targets the recording shortcut.

Why these operations. After level normalisation the low-level-cue check still finds
real and cloned clips separable by cheap recording statistics: peak amplitude, zero-
crossing rate, RMS variability, high-frequency energy and spectral roll-off
(``docs/results/lowlevel_cue_check_v1.md``, ``bundle_normalisation_v1.md``). A
detector can lean on those instead of on synthesis artefacts, which is the shortcut
the floor measures. Each operation below randomises one of those statistics on every
training clip, real and fake alike, so none of them stays a reliable label.

- ``random_gain``: level and peak amplitude.
- ``coloured_noise``: zero-crossing rate and high-frequency energy (white, pink or
  brown noise at a random SNR).
- ``spectral_tilt``: roll-off and high-frequency energy, standing in for the unknown
  microphone and handset response.
- ``telephone``: a second narrowband G.711 pass at a random SNR, widening the single
  fixed 20 dB line of the channel-matched protocol to a range of line qualities.

Every operation takes a seeded ``numpy.random.Generator``, so a clip's augmentation is
reproducible for a given (seed, epoch, index) -- see ``AudioManifestDataset._rng``.
Pure numpy except ``telephone``, which resamples through ``channel_sim`` (librosa,
imported lazily).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

#: Largest absolute sample allowed after augmentation.
PEAK_LIMIT = 0.99


def _peak_guard(x: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > PEAK_LIMIT:
        x = x * (PEAK_LIMIT / peak)
    return x.astype(np.float32)


def random_gain(x: np.ndarray, rng: np.random.Generator, max_db: float = 6.0) -> np.ndarray:
    """Scale by a gain drawn uniformly from ``[-max_db, +max_db]`` dB."""
    gain_db = rng.uniform(-max_db, max_db)
    return _peak_guard(np.asarray(x, dtype=np.float64) * 10.0 ** (gain_db / 20.0))


def _shaped_noise(n: int, rng: np.random.Generator, exponent: float) -> np.ndarray:
    """Noise with power spectrum ~ 1/f**exponent (0 white, 1 pink, 2 brown)."""
    white = rng.standard_normal(n)
    if exponent == 0 or n < 2:
        return white
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1]  # avoid dividing by zero at DC
    shaped = np.fft.irfft(spectrum / freqs ** (exponent / 2.0), n)
    return shaped / (np.std(shaped) + 1e-12)


def coloured_noise(
    x: np.ndarray,
    rng: np.random.Generator,
    snr_db: tuple[float, float] = (10.0, 40.0),
) -> np.ndarray:
    """Add white, pink or brown noise at an SNR drawn from ``snr_db``."""
    from src.data.channel_sim import add_noise_at_snr

    exponent = float(rng.choice([0.0, 1.0, 2.0]))
    noise = _shaped_noise(len(x), rng, exponent)
    return _peak_guard(add_noise_at_snr(x, noise, rng.uniform(*snr_db)))


def spectral_tilt(
    x: np.ndarray,
    rng: np.random.Generator,
    max_db_per_octave: float = 3.0,
    sr: int = 16_000,
    pivot_hz: float = 1_000.0,
) -> np.ndarray:
    """Tilt the spectrum by a random slope (dB/octave) about ``pivot_hz``.

    Loudness is held fixed (RMS restored), so this changes only the balance of low
    and high frequencies -- the roll-off and high-frequency energy cues.
    """
    signal = np.asarray(x, dtype=np.float64)
    if signal.size < 2:
        return signal.astype(np.float32)
    slope = rng.uniform(-max_db_per_octave, max_db_per_octave)
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sr)
    octaves = np.log2(np.maximum(freqs, 20.0) / pivot_hz)
    tilted = np.fft.irfft(spectrum * 10.0 ** (slope * octaves / 20.0), signal.size)
    rms_in = np.sqrt(np.mean(signal**2))
    rms_out = np.sqrt(np.mean(tilted**2))
    if rms_out > 1e-12:
        tilted *= rms_in / rms_out
    return _peak_guard(tilted)


def telephone(
    x: np.ndarray,
    rng: np.random.Generator,
    snr_db: tuple[float, float] = (5.0, 30.0),
) -> np.ndarray:
    """One more narrowband G.711 round trip with white noise at a random SNR."""
    from src.data.channel_sim import ChannelConfig, simulate_channel

    seed = int(rng.integers(0, 2**31 - 1))
    config = ChannelConfig(codec="g711", snr_db=float(rng.uniform(*snr_db)), seed=seed)
    return _peak_guard(simulate_channel(np.asarray(x, dtype=np.float32), config))


#: A recipe is a list of (operation, probability); each operation is applied
#: independently with its probability, in this order.
RECIPES: dict[str, tuple[tuple[Callable, float], ...]] = {
    # Breaks the residual recording cues without touching the channel.
    "shortcut_v1": (
        (random_gain, 1.0),
        (spectral_tilt, 0.5),
        (coloured_noise, 0.5),
    ),
    # The same, plus a varied telephone line: for the channel-matched (phone) systems.
    "channel_v1": (
        (random_gain, 1.0),
        (spectral_tilt, 0.5),
        (coloured_noise, 0.3),
        (telephone, 0.5),
    ),
}


class Augmenter:
    """A named recipe as a ``(waveform, rng) -> waveform`` callable.

    A class rather than a closure so the training dataset stays picklable: on Windows
    the DataLoader's worker processes receive the dataset by pickling it.
    """

    def __init__(self, name: str) -> None:
        if name not in RECIPES:
            raise ValueError(f"unknown augmentation recipe {name!r}; choose from {sorted(RECIPES)}")
        self.name = name

    def __call__(self, waveform: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        out = np.asarray(waveform, dtype=np.float32)
        for operation, probability in RECIPES[self.name]:
            if rng.random() < probability:
                out = operation(out, rng)
        return out

    def __repr__(self) -> str:
        return f"Augmenter({self.name!r})"


def build_augment(name: str) -> Augmenter:
    """The augmentation callable for a named recipe (see :data:`RECIPES`)."""
    return Augmenter(name)
