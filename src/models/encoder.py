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


class LayerWeightedSum(nn.Module):
    """Softmax-weighted sum over a stack of hidden-state layers.

    The weights start equal and are learned. Used instead of the last layer alone:
    the interpretability probes (P-034) show the bonafide/spoof decision forming by
    transformer layer 3, and the last layer of a large SSL model is tuned to its
    pretraining objective, so reading only that layer discards the layers where the
    cue lives.
    """

    def __init__(self, n_layers: int) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(int(n_layers)))

    def weights(self) -> torch.Tensor:
        """The current per-layer weights (sum to 1), e.g. for reporting."""
        return torch.softmax(self.logits, dim=0)

    def forward(self, layers: tuple[torch.Tensor, ...] | list[torch.Tensor]) -> torch.Tensor:
        """``n_layers`` tensors of ``(B, T, H)`` -> one ``(B, T, H)``."""
        if len(layers) != self.logits.numel():
            raise ValueError(f"expected {self.logits.numel()} layers, got {len(layers)}")
        stacked = torch.stack(tuple(layers), dim=0)
        w = self.weights().to(stacked.dtype).view(-1, *([1] * (stacked.dim() - 1)))
        return (w * stacked).sum(dim=0)


class SSLEncoder(nn.Module):
    """Frame-level SSL speech encoder returning ``(B, T, H)`` hidden states."""

    def __init__(
        self,
        name: str = "wav2vec2-base",
        freeze_feature_extractor: bool = True,
        freeze_encoder: bool = False,
        layer_weighting: bool = False,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        repo = ENCODER_IDS.get(name, name)
        self.model = AutoModel.from_pretrained(repo)
        self.hidden_size: int = int(self.model.config.hidden_size)
        #: ``None`` = the published behaviour, last layer only. Otherwise a learned
        #: mix of every hidden state (the input projection plus each transformer
        #: layer). It sits outside ``self.model``, so freezing the encoder never
        #: freezes it.
        self.layer_mix: LayerWeightedSum | None = None
        if layer_weighting:
            self.layer_mix = LayerWeightedSum(int(self.model.config.num_hidden_layers) + 1)

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
        if self.layer_mix is None:
            out = self.model(waveforms, attention_mask=attention_mask)
            return out.last_hidden_state
        out = self.model(waveforms, attention_mask=attention_mask, output_hidden_states=True)
        return self.layer_mix(out.hidden_states)
