"""Intra-utterance pitch range (f0 IQR) -- the measurement P-019 turns on.

P-019 recorded that XTTS-v2 compresses pitch range: a median intra-utterance f0
IQR of 25-29 Hz against 41.1 Hz for the MUCS lecture speech it clones from, and
that no generation parameter closed the gap. XTTS invents prosody from text and
regresses to a flat contour.

The plan's justification for the *second* attack family (CM02, RVC) is a
prediction that this cannot happen to voice conversion: RVC starts from a real
human recording and swaps timbre only, so the pitch contour that survives is the
source speaker's own. That prediction is the reason CM02 exists, so it has to be
measured rather than assumed -- a converted clip whose pitch range came back at
XTTS levels would be the more important finding, not a rounding error.

P-019's numbers were measured ad hoc with no committed code, so nothing could
reproduce them or be compared against them on equal terms. This module is that
code, and it is paired by design: :func:`compare` measures the converted clips
**and the real clips they were converted from** with one estimator and one set of
settings, so the verdict never rests on matching someone else's tooling.

Estimator: Praat's autocorrelation pitch tracker via ``praat-parselmouth``,
falling back to ``librosa.pyin``. Both are imported lazily (P-004) so CI, which
installs neither, still imports this module and tests the statistics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Pitch search band. The MUCS train pool's median pitch is 110 Hz (P-019) and no
#: gender metadata exists, so the band has to hold both low male and high female
#: voices without letting the tracker chase octave errors.
PITCH_FLOOR_HZ = 60.0
PITCH_CEILING_HZ = 400.0

#: An utterance with fewer voiced frames than this has no contour to take an IQR
#: of; reporting one anyway would be noise dressed as a measurement.
MIN_VOICED_FRAMES = 10

#: P-019's recorded numbers, for reference in reports. Measured ad hoc with
#: unknown tooling -- which is exactly why :func:`compare` re-measures the real
#: speech here instead of comparing against a remembered constant.
P019_XTTS_F0_IQR_HZ = (25.0, 29.0)
P019_MUCS_REAL_F0_IQR_HZ = 41.1
P019_HIACC_REAL_F0_IQR_HZ = 42.2


@dataclass(frozen=True)
class F0Config:
    """Pitch-tracking settings. One object for both sides of a comparison."""

    floor_hz: float = PITCH_FLOOR_HZ
    ceiling_hz: float = PITCH_CEILING_HZ
    time_step: float = 0.01
    min_voiced_frames: int = MIN_VOICED_FRAMES


def iqr(values) -> float:
    """Interquartile range of ``values``; 0.0 when there is nothing to spread."""
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 2:
        return 0.0
    q75, q25 = np.percentile(array, [75, 25])
    return float(q75 - q25)


def contour_stats(contour, config: F0Config | None = None) -> dict:
    """Median, IQR and voiced-frame count for one utterance's f0 contour.

    Separate from the estimator so the statistics are testable without Praat --
    the part that can silently change under a library upgrade is the part CI
    should be pinning.
    """
    cfg = config or F0Config()
    voiced = np.asarray(list(contour), dtype=float)
    voiced = voiced[np.isfinite(voiced) & (voiced > 0)]
    enough = voiced.size >= cfg.min_voiced_frames
    return {
        "voiced_frames": int(voiced.size),
        "f0_median_hz": round(float(np.median(voiced)), 2) if voiced.size else 0.0,
        "f0_iqr_hz": round(iqr(voiced), 2) if enough else 0.0,
        "usable": bool(enough),
    }


def f0_contour(audio, sample_rate: int, config: F0Config | None = None):
    """Voiced f0 values for one clip, in Hz. Praat first, ``librosa.pyin`` second.

    Returns only the voiced frames: an unvoiced frame is the absence of pitch, and
    folding it in as a zero would drag every IQR toward the floor.
    """
    cfg = config or F0Config()
    signal = np.asarray(audio, dtype=float)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    if signal.size == 0:
        return np.zeros(0, dtype=float)

    try:
        import parselmouth
    except ImportError:
        parselmouth = None

    if parselmouth is not None:
        sound = parselmouth.Sound(signal, sampling_frequency=float(sample_rate))
        pitch = sound.to_pitch(
            time_step=cfg.time_step,
            pitch_floor=cfg.floor_hz,
            pitch_ceiling=cfg.ceiling_hz,
        )
        values = np.asarray(pitch.selected_array["frequency"], dtype=float)
        return values[values > 0]

    import librosa

    f0, voiced_flag, _ = librosa.pyin(
        signal.astype(np.float32),
        fmin=cfg.floor_hz,
        fmax=cfg.ceiling_hz,
        sr=sample_rate,
    )
    f0 = np.asarray(f0, dtype=float)
    return f0[np.isfinite(f0) & np.asarray(voiced_flag, dtype=bool)]


def clip_f0_stats(audio, sample_rate: int, config: F0Config | None = None) -> dict:
    """:func:`f0_contour` then :func:`contour_stats`, for one clip."""
    return contour_stats(f0_contour(audio, sample_rate, config), config)


def summarise(rows) -> dict:
    """Median f0 IQR across clips, plus the spread of that median.

    The headline is a **median of per-utterance IQRs**, which is what P-019
    reported: it asks how much pitch movement a typical utterance carries, and is
    unmoved by one clip whose tracker went octave-hunting.
    """
    usable = [r for r in rows if r.get("usable")]
    if not usable:
        return {"clips": len(list(rows)), "usable": 0}
    iqrs = np.asarray([float(r["f0_iqr_hz"]) for r in usable])
    medians = np.asarray([float(r["f0_median_hz"]) for r in usable])
    return {
        "clips": int(len(rows)),
        "usable": int(len(usable)),
        "median_f0_iqr_hz": round(float(np.median(iqrs)), 2),
        "mean_f0_iqr_hz": round(float(iqrs.mean()), 2),
        "p25_f0_iqr_hz": round(float(np.percentile(iqrs, 25)), 2),
        "p75_f0_iqr_hz": round(float(np.percentile(iqrs, 75)), 2),
        "median_f0_hz": round(float(np.median(medians)), 2),
    }


def retention(converted: dict, real: dict) -> float:
    """Converted median f0 IQR as a percentage of the real one.

    100% means the conversion kept the source speaker's pitch movement intact.
    P-019 measured XTTS at roughly 60-70% of real; the CM02 prediction is that
    RVC lands near 100 because it never invents a contour.
    """
    reference = float(real.get("median_f0_iqr_hz") or 0.0)
    if reference <= 0:
        return 0.0
    return round(100.0 * float(converted.get("median_f0_iqr_hz") or 0.0) / reference, 1)


def verdict(converted: dict, real: dict, tolerance: float = 15.0) -> str:
    """One sentence stating whether P-019's prediction for CM02 held.

    ``tolerance`` is how many percentage points of pitch-range loss still counts
    as "the contour is the source speaker's own". Below that the finding is that
    RVC compresses pitch too, which would make CM02 a second easy attack rather
    than the harder family the plan was written around.
    """
    kept = retention(converted, real)
    if kept <= 0:
        return "no usable pitch measurement -- cannot rule on P-019's prediction"
    xtts_low, xtts_high = P019_XTTS_F0_IQR_HZ
    if kept >= 100.0 - tolerance:
        return (
            f"P-019's prediction HOLDS: converted clips keep {kept:.1f}% of the real "
            f"pitch range ({converted['median_f0_iqr_hz']:.1f} Hz vs "
            f"{real['median_f0_iqr_hz']:.1f} Hz), where XTTS-v2 kept only "
            f"{xtts_low:.0f}-{xtts_high:.0f} Hz. RVC starts from real speech, so the "
            f"contour is human and the CM01 compression does not occur."
        )
    return (
        f"P-019's prediction FAILS: converted clips keep only {kept:.1f}% of the real "
        f"pitch range ({converted['median_f0_iqr_hz']:.1f} Hz vs "
        f"{real['median_f0_iqr_hz']:.1f} Hz). CM02 compresses pitch as well, so it is "
        f"not the harder attack family the plan assumed -- report this, do not smooth it."
    )


def compare(converted_rows, real_rows, tolerance: float = 15.0) -> dict:
    """Both sides plus the verdict, ready to drop into a results document."""
    converted = summarise(converted_rows)
    real = summarise(real_rows)
    return {
        "converted": converted,
        "real_source": real,
        "retention_pct": retention(converted, real),
        "p019_reference": {
            "xtts_f0_iqr_hz": list(P019_XTTS_F0_IQR_HZ),
            "mucs_real_f0_iqr_hz": P019_MUCS_REAL_F0_IQR_HZ,
            "hiacc_real_f0_iqr_hz": P019_HIACC_REAL_F0_IQR_HZ,
            "note": "measured ad hoc, tooling unknown -- the real column above is re-measured here",
        },
        "verdict": verdict(converted, real, tolerance),
    }
