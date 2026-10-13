"""An epoch-aware shuffling sampler, so per-epoch randomness reaches the workers.

The training DataLoader keeps its worker processes alive between epochs
(``persistent_workers``), and each worker holds its own copy of the dataset. Setting
an epoch number on the main process's dataset therefore never reaches the copies that
actually load audio. The sampler, though, runs in the main process: by yielding
``(index, epoch)`` keys instead of bare indices, it carries the epoch to whichever
worker loads the clip, and :class:`src.data.dataset.AudioManifestDataset` seeds that
clip's crop and augmentation from it. Same seed and epoch, same order and same crops,
on any number of workers.

Only used when a config opts in (``crop_per_epoch`` or ``augment``); the published
recipe keeps PyTorch's default shuffling.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch.utils.data import Sampler


class EpochSampler(Sampler):
    """Shuffle ``range(n)`` afresh each epoch and yield ``(index, epoch)`` pairs."""

    def __init__(self, n: int, seed: int = 1234) -> None:
        self.n = int(n)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Call before iterating each epoch; the order and the crops follow it."""
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[tuple[int, int]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + 7919 * self.epoch)
        for index in torch.randperm(self.n, generator=generator).tolist():
            yield (index, self.epoch)

    def __len__(self) -> int:
        return self.n
