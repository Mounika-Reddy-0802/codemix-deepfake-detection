"""Dataset + collate for variable-length audio, driven by a manifest CSV.

Kept import-light: only numpy + pandas at module top (both available in CI), with
torch imported lazily inside :func:`collate_fn`. The dataset is a map-style object
(``__len__`` + ``__getitem__``) so it works with ``torch.utils.data.DataLoader``
without importing torch here. ``pad_batch`` is pure numpy and unit-tested.

Clip paths go through :mod:`src.utils.paths`, so one frozen manifest loads on the
CPU laptop and on a Colab GPU runtime without being edited — only ``$DATA_ROOT``
differs between the two machines.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.utils.paths import resolve as resolve_path

LABEL_TO_INT = {"bonafide": 1, "spoof": 0}


def pad_batch(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Right-pad 1-D arrays to the longest length.

    Returns ``(padded [B, T_max], lengths [B])``. Empty input -> empty arrays.
    """
    if not arrays:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    lengths = np.array([a.shape[0] for a in arrays], dtype=np.int64)
    t_max = int(lengths.max())
    padded = np.zeros((len(arrays), t_max), dtype=np.float32)
    for i, a in enumerate(arrays):
        padded[i, : a.shape[0]] = np.asarray(a, dtype=np.float32)
    return padded, lengths


@dataclass
class AudioSample:
    """One item returned by the dataset."""

    waveform: np.ndarray
    label: int
    language: str
    speaker: str
    condition: str


class AudioManifestDataset:
    """Map-style dataset over a manifest (filepath, label, language, speaker, ...)."""

    def __init__(
        self,
        manifest: str | pd.DataFrame,
        target_sr: int = 16_000,
        split: str | None = None,
        data_root: str | os.PathLike[str] | None = None,
    ) -> None:
        df = pd.read_csv(manifest) if isinstance(manifest, str) else manifest.copy()
        if split is not None:
            df = df[df["split"] == split].reset_index(drop=True)
        self.df = df.reset_index(drop=True)
        self.target_sr = target_sr
        #: Resolved lazily per item so ``$DATA_ROOT`` set after construction (or in
        #: a DataLoader worker process) is still honoured.
        self.data_root = data_root

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> AudioSample:
        from src.utils.audio_utils import load_wav

        row = self.df.iloc[index]
        filepath = resolve_path(row["filepath"], self.data_root)
        waveform, _ = load_wav(filepath, target_sr=self.target_sr)
        return AudioSample(
            waveform=waveform,
            label=LABEL_TO_INT[str(row["label"]).lower()],
            language=str(row.get("language", "")),
            speaker=str(row.get("speaker", "")),
            condition=str(row.get("condition", "")),
        )


def collate_fn(batch: list[AudioSample]) -> dict[str, Any]:
    """Collate a list of :class:`AudioSample` into padded torch tensors.

    Returns ``{"waveforms": [B, T], "lengths": [B], "labels": [B]}``. Torch is
    imported here so the module stays importable without it.
    """
    import torch

    padded, lengths = pad_batch([b.waveform for b in batch])
    labels = np.array([b.label for b in batch], dtype=np.int64)
    return {
        "waveforms": torch.from_numpy(padded),
        "lengths": torch.from_numpy(lengths),
        "labels": torch.from_numpy(labels),
    }
