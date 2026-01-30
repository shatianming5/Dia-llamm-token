from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from ..torch_compat import Module, Tensor


@dataclass
class QuantizeResult:
    """Quantization outputs for VQ/RVQ-style tokenizers."""

    codes: Tensor
    embeddings: Tensor
    aux: Dict[str, Any]


class ResidualVectorQuantizer(Module):
    """Residual VQ (RVQ) interface placeholder."""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg

    def quantize(self, feats: Tensor) -> QuantizeResult:
        raise NotImplementedError

