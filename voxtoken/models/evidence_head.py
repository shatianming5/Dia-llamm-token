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
        nodes: List[EvidenceNode] = []
        for t in tokens:
            nodes.append(
                EvidenceNode(
                    eid=f"tok-{t.token_id}",
                    finding_type=str(self.cfg.get("default_finding_type", "token")),
                    attrs={
                        "side": "U",
                        "location": "U",
                        "size_bin": "U",
                        "certainty": "U",
                    },
                    supported_token_ids=[int(t.token_id)],
                )
            )
        return nodes


__all__ = [
    "EvidenceHead",
]
