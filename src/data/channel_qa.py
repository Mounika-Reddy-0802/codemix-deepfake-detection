"""Objective verification of the channel-simulation output (W3-T2, owner L).

The listening test asks a human "does this sound like a phone call?". That
question has two halves, and only one of them needs ears:

- **Is the chain doing what it claims?** Band-limited to a narrowband telephony
  channel, noise at the configured SNR, no clipping, no truncation, still
  correlated with the original. All of that is measurable.
- **Does it sound right?** Perceptual naturalness, intelligibility of the Hindi
  and English halves, whether artefacts are distracting. That needs a person.

This module does the first half, so the listening session starts from "the
numbers check out, now use your ears" rather than from scratch. It is also the
part that catches the failure the protocol most fears: a botched resample or a
mis-tuned SNR that quietly degrades every eval clip while the pipeline reports
success. A file that is merely *quieter* than the original would pass a casual
listen and fail the bandwidth check here.

The decisive measurement is **spectral bandwidth**. A 16 kHz clean recording
carries energy up to ~8 kHz. Passing it through an 8 kHz narrowband codec and
back must leave essentially nothing above 4 kHz -- that is what "telephony" means
physically. If the channel copy still has wideband energy, the downsample never
happened.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

#: Nyquist of the narrowband telephony channel. Nothing meaningful may sit above.
NARROWBAND_NYQUIST_HZ = 4000.0
#: Allowance for filter roll-off when judging whether band-limiting happened.
BANDWIDTH_TOLERANCE_HZ = 300.0
#: Fraction of spectral energy used to define "bandwidth".
ENERGY_QUANTILE = 0.99


#: Above this fraction of energy over 4 kHz, the narrowband stage did not happen.
MAX_HF_ENERGY_RATIO = 0.001


@dataclass(frozen=True)
class PairMeasurement:
    """Objective facts about one clean/channel pair."""

    pair_id: int
    bandwidth_clean_hz: float
    bandwidth_channel_hz: float
    hf_energy_clean: float
    hf_energy_channel: float
    measured_snr_db: float
    correlation: float
    clipping_ratio: float
    duration_ratio: float
    band_limited: bool
    intact: bool

    def as_dict(self) -> dict:
        return asdict(self)


def high_frequency_energy_ratio(
    audio: np.ndarray, sample_rate: int, cutoff_hz: float = NARROWBAND_NYQUIST_HZ
) -> float:
    """Fraction of total energy above ``cutoff_hz``.

    Sharper than a bandwidth quantile for this job. Speech energy concentrates
    below 4 kHz anyway, so the 99% quantile of a genuinely wideband 16 kHz
    recording still lands near 3.2 kHz and looks deceptively "narrowband". The
    energy *above* the cutoff is the honest discriminator: on the real corpora
    clean clips carry 0.02-1.7% up there, and the channel copies carry 0.000%.
    """
    signal = np.asarray(audio, dtype=np.float64)
    if signal.size == 0:
        return 0.0
    spectrum = np.abs(np.fft.rfft(signal)) ** 2
    total = spectrum.sum()
    if total <= 0:
        return 0.0
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate)
    return float(spectrum[freqs > cutoff_hz].sum() / total)


def spectral_bandwidth_hz(
    audio: np.ndarray, sample_rate: int, quantile: float = ENERGY_QUANTILE
) -> float:
    """Frequency below which ``quantile`` of the signal's energy lies.

    A blunt but robust bandwidth estimate: no windowing subtleties, and immune to
    a stray high-frequency tick that a simple "highest non-zero bin" measure would
    be fooled by.
    """
    signal = np.asarray(audio, dtype=np.float64)
    if signal.size == 0:
        return 0.0
    spectrum = np.abs(np.fft.rfft(signal)) ** 2
    total = spectrum.sum()
    if total <= 0:
        return 0.0
    cumulative = np.cumsum(spectrum) / total
    index = int(np.searchsorted(cumulative, quantile))
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate)
    return float(freqs[min(index, freqs.size - 1)])


def measured_snr_db(clean: np.ndarray, processed: np.ndarray) -> float:
    """SNR of ``processed`` treating ``clean`` as signal and the difference as noise.

    The two are level-matched first: the channel chain changes gain, and an
    unmatched comparison would report that gain change as noise.
    """
    a = np.asarray(clean, dtype=np.float64)
    b = np.asarray(processed, dtype=np.float64)
    n = min(a.size, b.size)
    if n == 0:
        return 0.0
    a, b = a[:n], b[:n]

    # Both terms use np.dot so their accumulation order matches: with np.sum for
    # one and np.dot for the other, an identical pair produced a scale of
    # 1.0 +/- 1e-16 and a spuriously finite SNR instead of "perfect".
    energy_a = float(np.dot(a, a))
    if energy_a <= 0:
        return 0.0
    scale = float(np.dot(a, b)) / energy_a  # least-squares gain match
    residual = b - scale * a
    noise = float(np.sum(residual * residual))
    if noise <= 0:
        return float("inf")
    return float(10.0 * np.log10(energy_a * scale * scale / noise))


def correlation(clean: np.ndarray, processed: np.ndarray) -> float:
    """Pearson correlation between the two signals, aligned at the start."""
    a = np.asarray(clean, dtype=np.float64)
    b = np.asarray(processed, dtype=np.float64)
    n = min(a.size, b.size)
    if n < 2:
        return 0.0
    a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denom == 0 else float(np.dot(a, b) / denom)


def clipping_ratio(audio: np.ndarray, threshold: float = 0.999) -> float:
    """Fraction of samples at or beyond full scale."""
    signal = np.abs(np.asarray(audio, dtype=np.float64))
    return 0.0 if signal.size == 0 else float((signal >= threshold).mean())


def measure_pair(
    clean: np.ndarray,
    channel: np.ndarray,
    sample_rate: int,
    pair_id: int = 0,
    min_correlation: float = 0.3,
) -> PairMeasurement:
    """Measure one clean/channel pair against the protocol's claims."""
    bw_clean = spectral_bandwidth_hz(clean, sample_rate)
    bw_channel = spectral_bandwidth_hz(channel, sample_rate)
    hf_clean = high_frequency_energy_ratio(clean, sample_rate)
    hf_channel = high_frequency_energy_ratio(channel, sample_rate)
    snr = measured_snr_db(clean, channel)
    corr = correlation(clean, channel)
    clip = clipping_ratio(channel)
    ratio = (len(channel) / len(clean)) if len(clean) else 0.0

    # Energy above the cutoff is the decisive test; the quantile is context.
    band_limited = hf_channel <= MAX_HF_ENERGY_RATIO and (
        bw_channel <= NARROWBAND_NYQUIST_HZ + BANDWIDTH_TOLERANCE_HZ
    )
    # "intact" = still the same utterance: recognisably correlated, not clipped
    # to death, and not truncated or padded.
    intact = corr >= min_correlation and clip < 0.01 and 0.95 <= ratio <= 1.05

    return PairMeasurement(
        pair_id=pair_id,
        bandwidth_clean_hz=round(bw_clean, 1),
        bandwidth_channel_hz=round(bw_channel, 1),
        hf_energy_clean=round(hf_clean, 6),
        hf_energy_channel=round(hf_channel, 6),
        measured_snr_db=round(snr, 2),
        correlation=round(corr, 4),
        clipping_ratio=round(clip, 6),
        duration_ratio=round(ratio, 4),
        band_limited=bool(band_limited),
        intact=bool(intact),
    )


def summarise(measurements: list[PairMeasurement]) -> dict:
    """Verdict over a whole listening set."""
    if not measurements:
        return {"pairs": 0, "all_band_limited": False, "all_intact": False}
    bw_channel = [m.bandwidth_channel_hz for m in measurements]
    bw_clean = [m.bandwidth_clean_hz for m in measurements]
    snrs = [m.measured_snr_db for m in measurements if np.isfinite(m.measured_snr_db)]
    return {
        "pairs": len(measurements),
        "all_band_limited": all(m.band_limited for m in measurements),
        "all_intact": all(m.intact for m in measurements),
        "median_bandwidth_clean_hz": round(float(np.median(bw_clean)), 1),
        "median_bandwidth_channel_hz": round(float(np.median(bw_channel)), 1),
        "median_hf_energy_clean": round(
            float(np.median([m.hf_energy_clean for m in measurements])), 6
        ),
        "max_hf_energy_channel": round(max(m.hf_energy_channel for m in measurements), 6),
        "median_snr_db": round(float(np.median(snrs)), 2) if snrs else None,
        "median_correlation": round(float(np.median([m.correlation for m in measurements])), 4),
        "failures": [m.pair_id for m in measurements if not (m.band_limited and m.intact)],
    }


def main() -> None:
    """CLI: ``python -m src.data.channel_qa --sheet docs/qa/channel_sim_listening_sheet.csv``."""
    import argparse
    import json
    from pathlib import Path

    import pandas as pd
    import soundfile as sf

    parser = argparse.ArgumentParser(description="Objectively verify the channel simulation")
    parser.add_argument("--sheet", default="docs/qa/channel_sim_listening_sheet.csv")
    parser.add_argument("--out", default="docs/qa/channel_sim_measurements.csv")
    args = parser.parse_args()

    sheet = pd.read_csv(args.sheet).drop_duplicates("pair_id")
    measurements: list[PairMeasurement] = []
    for row in sheet.itertuples(index=False):
        try:
            clean, sr = sf.read(row.clean_path, dtype="float32", always_2d=False)
            channel, _ = sf.read(row.channel_path, dtype="float32", always_2d=False)
        except Exception as exc:  # noqa: BLE001 - a missing pair is reported, not fatal
            print(f"  [skip] pair {row.pair_id}: {type(exc).__name__}: {exc}")
            continue
        measurements.append(measure_pair(clean, channel, sr, pair_id=int(row.pair_id)))

    frame = pd.DataFrame([m.as_dict() for m in measurements])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)

    verdict = summarise(measurements)
    print(json.dumps(verdict, indent=2))
    print(f"\nwrote {out}")
    if verdict["all_band_limited"] and verdict["all_intact"]:
        print("\nthe chain measures as telephony. Perceptual quality still needs ears:")
        print("fill in docs/qa/channel_sim_listening_sheet.csv")
    else:
        print(f"\nFAILURES on pairs {verdict['failures']} -- fix before the listening session")


if __name__ == "__main__":
    main()
