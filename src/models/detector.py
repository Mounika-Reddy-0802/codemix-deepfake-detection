"""Full detector: SSL encoder + attentive statistics pooling + MLP head."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.classifier import MLPHead
from src.models.encoder import SSLEncoder
from src.models.pooling import AttentiveStatsPooling


class Detector(nn.Module):
    """End-to-end spoof detector producing 2-class logits from a raw waveform."""

    def __init__(
        self,
        encoder_name: str = "wav2vec2-base",
        num_classes: int = 2,
        head_hidden: int = 256,
        dropout: float = 0.3,
        freeze_feature_extractor: bool = True,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = SSLEncoder(encoder_name, freeze_feature_extractor, freeze_encoder)
        hidden = self.encoder.hidden_size
        self.pooling = AttentiveStatsPooling(hidden)
        self.head = MLPHead(2 * hidden, head_hidden, num_classes, dropout)

    def forward(
        self,
        waveforms: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``waveforms`` ``(B, T)`` -> logits ``(B, num_classes)``."""
        features = self.encoder(waveforms, attention_mask)
        pooled = self.pooling(features, lengths)
        return self.head(pooled)
