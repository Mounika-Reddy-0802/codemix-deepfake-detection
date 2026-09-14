"""The one detector a live-call server shares across every call (owner SK).

Loading wav2vec2 takes seconds and about a gigabyte, so it happens once, on first
use, behind a lock. Inference runs on a **single worker thread**: torch already uses
every core for one forward pass, so two passes in parallel only fight for them, and
one thread keeps windows from different calls in arrival order.

Configuration comes from the environment, so the same code runs on the laptop and
on a server:

- ``DFD_CHECKPOINT``: detector checkpoint (S1, S2 LoRA or S3), required;
- ``DFD_THRESHOLD_FILE``: frozen operating point, default ``configs/threshold.yaml``;
- ``DFD_DEVICE``: ``cpu`` (default) or ``cuda``.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from live_call.verdict_engine import EngineConfig, load_config


@dataclass
class DetectorSettings:
    checkpoint: str
    threshold_file: str
    device: str

    @classmethod
    def from_env(cls) -> DetectorSettings:
        return cls(
            checkpoint=os.environ.get("DFD_CHECKPOINT", ""),
            threshold_file=os.environ.get("DFD_THRESHOLD_FILE", "configs/threshold.yaml"),
            device=os.environ.get("DFD_DEVICE", "cpu"),
        )


class DetectorService:
    """Lazy, thread-safe model plus the executor every session scores on."""

    def __init__(
        self, settings: DetectorSettings, score_fn=None, config: EngineConfig | None = None
    ):
        self.settings = settings
        self._score_fn = score_fn  # injected in tests; loaded from the checkpoint otherwise
        self._config = config
        self._lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="detector")

    @property
    def loaded(self) -> bool:
        return self._score_fn is not None

    def config(self) -> EngineConfig:
        if self._config is None:
            self._config = load_config(self.settings.threshold_file)
        return self._config

    def score_fn(self):
        if self._score_fn is None:
            with self._lock:
                if self._score_fn is None:
                    if not self.settings.checkpoint or not Path(self.settings.checkpoint).is_file():
                        raise FileNotFoundError(
                            f"DFD_CHECKPOINT not found: {self.settings.checkpoint!r}"
                        )
                    from src.inference.predict import load_detector, make_score_fn

                    model, device = load_detector(
                        self.settings.checkpoint, device=self.settings.device
                    )
                    self._score_fn = make_score_fn(model, device)
        return self._score_fn

    def status(self) -> dict:
        try:
            threshold = self.config().threshold
            threshold_error = None
        except FileNotFoundError as exc:
            threshold, threshold_error = None, str(exc)
        return {
            "checkpoint": Path(self.settings.checkpoint).name if self.settings.checkpoint else None,
            "checkpoint_found": bool(self.settings.checkpoint)
            and Path(self.settings.checkpoint).is_file(),
            "model_loaded": self.loaded,
            "device": self.settings.device,
            "threshold": threshold,
            "threshold_error": threshold_error,
        }
