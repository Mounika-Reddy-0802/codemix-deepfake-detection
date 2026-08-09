"""Attentive statistics pooling.

Pools a variable-length frame sequence ``(B, T, H)`` into a fixed vector
``(B, 2H)`` = attention-weighted mean concatenated with weighted std. Optional
length masking excludes padded frames from the attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AttentiveStatsPooling(nn.Module):
    """Attentive statistics pooling (mean + std), output dim ``2 * input_dim``."""

    def __init__(self, input_dim: int, attention_dim: int = 128, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """``x`` ``(B, T, H)`` -> pooled ``(B, 2H)``."""
        scores = self.attention(x)  # (B, T, 1)
        if lengths is not None:
            t = x.shape[1]
            idx = torch.arange(t, device=x.device).unsqueeze(0)  # (1, T)
            mask = idx < lengths.unsqueeze(1)  # (B, T)
            scores = scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        weights = torch.softmax(scores, dim=1)  # (B, T, 1)
        mean = torch.sum(weights * x, dim=1)  # (B, H)
        var = torch.sum(weights * x * x, dim=1) - mean * mean
        std = torch.sqrt(var.clamp(min=self.eps))
        return torch.cat([mean, std], dim=1)  # (B, 2H)
