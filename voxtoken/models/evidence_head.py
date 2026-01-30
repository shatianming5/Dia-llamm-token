from __future__ import annotations

from typing import Any, Dict, List

from ..schemas import EvidenceNode, Token
from ..torch_compat import Module, Tensor


class EvidenceHead(Module):
    """Token -> structured evidence nodes."""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg

    def forward(self, tokens: List[Token], embed_bank: Tensor) -> List[EvidenceNode]:
        """
        Args:
            tokens: selected tokens
            embed_bank: (N, d) embeddings aligned with tokens

        Returns:
            Evidence nodes with structured attrs & supported_token_ids.
        """
        raise NotImplementedError


__all__ = [
    "EvidenceHead",
]

