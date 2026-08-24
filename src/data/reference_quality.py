"""Pick cloning references by how much SPEECH they contain, not how long they are.

The failure this exists for. `choose_references` takes the longest clip a speaker
has over the minimum duration, which is a proxy for "most material for XTTS to
condition on". It is a bad proxy: in the 4,000-clip Week-4 run, speaker 987461's
reference was 25 seconds of which **89% was silence** — under 3 seconds of actual
voice. XTTS built a near-empty speaker embedding from it and then stalled or
truncated on **95 of that speaker's 160 clips**, which is 82% of every bad clip in
the entire run. The other 24 speakers, whose references sit at 0.32-0.61 silence,
produced 0-6 failures each.

Nothing raised. The clips were written, logged as successes, and would have gone
into the training corpus as ~95 files of near-silence labelled "spoof".

So the rule is: rank candidates by **effective speech seconds** (duration minus
silence), not by duration. A 10-second clip that is 90% voice beats a 25-second
clip that is 11% voice, which is the correct preference and the opposite of what
duration-ranking gives.

Audio imports are lazy so the module still imports under CI (P-004), which
installs no audio stack.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

#: Fraction of the 95th-percentile frame energy below which a frame is silence.
SILENCE_REL_THRESHOLD = 0.15

#: A reference must carry at least this much actual speech. 987461's failing
#: reference had ~2.75 s; every healthy speaker in the run cleared 5 s comfortably.
MIN_SPEECH_SECONDS = 5.0

#: How many duration-ranked candidates to probe per speaker. Probing costs an
#: audio decode each, so this is bounded rather than exhaustive.
MAX_CANDIDATES = 10


@dataclass(frozen=True)
class ReferenceQuality:
    """What a candidate reference actually contains."""

    utt_id: str
    duration_seconds: float
    silence_fraction: float
    speech_seconds: float

    @property
    def usable(self) -> bool:
        return self.speech_seconds >= MIN_SPEECH_SECONDS


def silence_fraction(audio, sample_rate: int) -> float:
    """Fraction of frames whose energy is negligible against the clip's own peak.

    Relative rather than absolute, so a quietly-recorded speaker is not judged
    silent: the threshold is a fraction of that clip's own 95th-percentile energy.
    """
    import numpy as np

    if audio is None or len(audio) == 0:
        return 1.0
    frame, hop = 1024, 256
    if len(audio) < frame:
        return 0.0
    n = 1 + (len(audio) - frame) // hop
    frames = np.lib.stride_tricks.as_strided(
        audio, shape=(n, frame), strides=(audio.strides[0] * hop, audio.strides[0])
    )
    energy = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
    if energy.size == 0:
        return 1.0
    threshold = SILENCE_REL_THRESHOLD * float(np.percentile(energy, 95))
    if threshold <= 0:
        return 1.0
    return float((energy < threshold).mean())


def probe(row: dict, target_sr: int = 16000) -> ReferenceQuality:
    """Measure one candidate clip. Decodes audio, so callers should bound the count."""
    from src.data.corpora import load_clip

    audio, sample_rate = load_clip(row, target_sr=target_sr)
    duration = len(audio) / sample_rate if sample_rate else 0.0
    quiet = silence_fraction(audio, sample_rate)
    return ReferenceQuality(
        utt_id=str(row.get("utt_id", "")),
        duration_seconds=round(duration, 2),
        silence_fraction=round(quiet, 3),
        speech_seconds=round(duration * (1.0 - quiet), 2),
    )


def rank_candidates(candidates: pd.DataFrame, max_candidates: int = MAX_CANDIDATES) -> list:
    """Probe the longest ``max_candidates`` clips and order them by speech content.

    Ties break on ``utt_id`` so two machines choose the same reference (P-016).
    """
    shortlist = candidates.sort_values(
        ["duration_seconds", "utt_id"], ascending=[False, True], kind="stable"
    ).head(max_candidates)
    probed = [probe(row._asdict()) for row in shortlist.itertuples(index=False)]
    return sorted(probed, key=lambda q: (-q.speech_seconds, q.utt_id))


def best_reference(candidates: pd.DataFrame, max_candidates: int = MAX_CANDIDATES):
    """The candidate carrying the most speech, or ``None`` when the frame is empty."""
    if candidates.empty:
        return None
    ranked = rank_candidates(candidates, max_candidates)
    return ranked[0] if ranked else None


def audit(references: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """Report speech content of the CURRENT reference per speaker, worst first.

    Run this before generating at scale: a speaker whose ``usable`` is False will
    produce a large share of that run's dead clips, and it costs one decode per
    speaker to find out instead of one wasted GPU-hour per speaker to discover.
    """
    rows: list[dict] = []
    for row in references.itertuples(index=False):
        quality = probe(row._asdict())
        rows.append(
            {
                "speaker": str(row.speaker),
                "utt_id": quality.utt_id,
                "duration_seconds": quality.duration_seconds,
                "silence_fraction": quality.silence_fraction,
                "speech_seconds": quality.speech_seconds,
                "usable": quality.usable,
            }
        )
    return pd.DataFrame(rows).sort_values("speech_seconds").reset_index(drop=True)
