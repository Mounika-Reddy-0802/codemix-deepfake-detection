"""SSL encoder wrapper (wav2vec2 / WavLM / XLSR) with freeze control.

Wraps a Hugging Face SSL speech model and exposes its frame-level hidden states.
``transformers`` is imported lazily in ``__init__`` so importing this module needs
only torch. The CNN feature extractor is frozen by default (standard for
anti-spoofing fine-tuning); the transformer layers are trainable unless
``freeze_encoder`` is set.
"""

from __future__ import annotations

import torch
import torch.nn as nn

ENCODER_IDS = {
    "wav2vec2-base": "facebook/wav2vec2-base",
    "wavlm-base": "microsoft/wavlm-base",
    "xlsr-53": "facebook/wav2vec2-large-xlsr-53",
}


class SSLEncoder(nn.Module):
    """Frame-level SSL speech encoder returning ``(B, T, H)`` hidden states."""

    def __init__(
        self,
        name: str = "wav2vec2-base",
        freeze_feature_extractor: bool = True,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        repo = ENCODER_IDS.get(name, name)
        self.model = AutoModel.from_pretrained(repo)
        self.hidden_size: int = int(self.model.config.hidden_size)

        if freeze_feature_extractor and hasattr(self.model, "feature_extractor"):
            for param in self.model.feature_extractor.parameters():
                param.requires_grad = False
        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(
        self, waveforms: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``waveforms`` ``(B, T)`` at 16 kHz -> hidden states ``(B, T', H)``."""
        out = self.model(waveforms, attention_mask=attention_mask)
        return out.last_hidden_state
