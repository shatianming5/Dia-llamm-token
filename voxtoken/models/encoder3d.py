from __future__ import annotations

from typing import Any, Dict

from ..torch_compat import Module, Tensor


class Encoder3D(Module):
    """3D encoder interface used by the tokenizer."""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg

    def forward(self, volume: Tensor) -> Tensor:
        """Encode a (C, D, H, W) volume into latent features."""
        raise NotImplementedError

