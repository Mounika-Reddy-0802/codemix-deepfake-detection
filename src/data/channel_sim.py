"""Telephony channel simulation: 16 kHz -> 8 kHz -> codec -> noise@SNR -> 16 kHz.

This is the core of the channel-matched protocol: every evaluation clip is passed
through the same narrowband telephony chain the live Twilio demo delivers, so the
detector is trained/tested on the condition it is deployed in.

The lossy pieces that matter for the paper are implemented in pure numpy and are
fully unit-testable without the heavy audio stack:

- **G.711 mu-law** companding + 8-bit quantisation (``mu_law_encode`` / ``decode``),
- **additive noise at a controlled SNR** (``add_noise_at_snr``).

Resampling (16k<->8k) uses ``librosa`` lazily via ``audio_utils``. AMR-NB uses
``ffmpeg`` lazily (optional); G.711 mu-law is the dependency-free default.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.utils.audio_utils import TARGET_SR, resample, rms

NARROWBAND_SR = 8_000
MU = 255  # G.711 mu-law companding parameter
DEFAULT_SNR_DB = (5.0, 10.0, 15.0, 20.0)


# --------------------------------------------------------------------------- #
# G.711 mu-law codec (pure numpy, testable)
# --------------------------------------------------------------------------- #
def mu_law_encode(signal: np.ndarray, mu: int = MU) -> np.ndarray:
    """Compand a signal in [-1, 1] to 8-bit mu-law codes (uint8, 0..255)."""
    x = np.clip(np.asarray(signal, dtype=np.float64), -1.0, 1.0)
    compressed = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)  # [-1, 1]
    codes = np.round((compressed + 1.0) / 2.0 * mu).astype(np.uint8)
    return codes


def mu_law_decode(codes: np.ndarray, mu: int = MU) -> np.ndarray:
    """Expand 8-bit mu-law codes back to a float32 signal in [-1, 1]."""
    q = np.asarray(codes, dtype=np.float64)
    compressed = q / mu * 2.0 - 1.0
    x = np.sign(compressed) * (1.0 / mu) * (np.power(1.0 + mu, np.abs(compressed)) - 1.0)
    return x.astype(np.float32)


def apply_g711_ulaw(signal: np.ndarray) -> np.ndarray:
    """Round-trip through G.711 mu-law (adds realistic 8-bit codec distortion)."""
    return mu_law_decode(mu_law_encode(signal))


# --------------------------------------------------------------------------- #
# Additive noise at a target SNR (pure numpy, testable)
# --------------------------------------------------------------------------- #
def add_noise_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Add ``noise`` to ``signal`` scaled to hit exactly ``snr_db``.

    SNR = 10 * log10(P_signal / P_noise). The noise is tiled/truncated to match
    the signal length, then rescaled so the resulting SNR equals ``snr_db``.
    """
    s = np.asarray(signal, dtype=np.float64)
    n = np.asarray(noise, dtype=np.float64)
    if n.size == 0:
        return s.astype(np.float32)
    if n.size < s.size:
        n = np.tile(n, int(np.ceil(s.size / n.size)))
    n = n[: s.size]

    p_sig = float(np.mean(s * s))
    p_noise = float(np.mean(n * n))
    if p_sig < 1e-12 or p_noise < 1e-12:
        return s.astype(np.float32)

    target_p_noise = p_sig / (10.0 ** (snr_db / 10.0))
    n = n * np.sqrt(target_p_noise / p_noise)
    return (s + n).astype(np.float32)


def add_white_noise_at_snr(
    signal: np.ndarray, snr_db: float, seed: int | None = None
) -> np.ndarray:
    """Add white Gaussian noise at a target SNR (deterministic given ``seed``)."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(np.asarray(signal).shape)
    return add_noise_at_snr(signal, noise, snr_db)


def measured_snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    """Estimate SNR (dB) between a clean signal and its noisy version."""
    s = np.asarray(clean, dtype=np.float64)
    noise = np.asarray(noisy, dtype=np.float64)[: s.size] - s[: len(noisy)]
    p_sig, p_noise = float(np.mean(s * s)), float(np.mean(noise * noise))
    if p_noise < 1e-12:
        return float("inf")
    return 10.0 * float(np.log10(p_sig / p_noise))


# --------------------------------------------------------------------------- #
# Codec dispatch + full chain
# --------------------------------------------------------------------------- #
def apply_codec(signal_8k: np.ndarray, codec: str) -> np.ndarray:
    """Apply a narrowband codec round-trip to an 8 kHz signal."""
    codec = codec.lower()
    if codec in {"g711", "g711_ulaw", "ulaw", "mulaw"}:
        return apply_g711_ulaw(signal_8k)
    if codec in {"amr", "amr_nb", "amrnb"}:
        return _apply_amr_nb(signal_8k)
    if codec in {"none", "identity"}:
        return np.asarray(signal_8k, dtype=np.float32)
    raise ValueError(f"unknown codec: {codec!r} (use g711 | amr_nb | none)")


def _apply_amr_nb(signal_8k: np.ndarray, bitrate: str = "7.40k") -> np.ndarray:
    """AMR-NB round-trip via ffmpeg (lazy). Falls back to G.711 if ffmpeg absent."""
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    if shutil.which("ffmpeg") is None:
        # Keep the pipeline running; caller can require AMR explicitly elsewhere.
        return apply_g711_ulaw(signal_8k)

    import soundfile as sf

    with tempfile.TemporaryDirectory() as tmp:
        wav_in = Path(tmp) / "in.wav"
        amr = Path(tmp) / "mid.amr"
        wav_out = Path(tmp) / "out.wav"
        sf.write(wav_in, np.asarray(signal_8k, dtype=np.float32), NARROWBAND_SR)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(wav_in), "-ar", "8000", "-ab", bitrate, str(amr)],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(amr), "-ar", "8000", str(wav_out)],
            check=True,
            capture_output=True,
        )
        out, _ = sf.read(wav_out, dtype="float32")
    return np.asarray(out, dtype=np.float32)


@dataclass
class ChannelConfig:
    """Configuration for one channel-simulation pass."""

    codec: str = "g711"
    snr_db: float = 20.0
    in_sr: int = TARGET_SR
    narrowband_sr: int = NARROWBAND_SR
    noise: np.ndarray | None = field(default=None, repr=False)
    seed: int | None = None


def simulate_channel(audio: np.ndarray, config: ChannelConfig) -> np.ndarray:
    """Run the full telephony chain and return audio back at the input rate.

    16 kHz -> downsample to 8 kHz -> codec round-trip -> add noise@SNR ->
    upsample to 16 kHz.
    """
    x = np.asarray(audio, dtype=np.float32)
    narrow = resample(x, config.in_sr, config.narrowband_sr)
    narrow = apply_codec(narrow, config.codec)
    if config.noise is not None:
        narrow = add_noise_at_snr(narrow, config.noise, config.snr_db)
    else:
        narrow = add_white_noise_at_snr(narrow, config.snr_db, seed=config.seed)
    wide = resample(narrow, config.narrowband_sr, config.in_sr)
    # Guard against any codec/resample overshoot.
    peak = float(np.max(np.abs(wide))) if wide.size else 0.0
    if peak > 1.0:
        wide = (wide / peak).astype(np.float32)
    return wide


def _demo_level(signal: np.ndarray) -> float:
    """Small convenience used by scripts/logs: RMS in dBFS."""
    from src.utils.audio_utils import amp_to_db

    return amp_to_db(rms(signal))
