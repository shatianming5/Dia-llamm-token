from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from ..schemas import EvidenceNode, Token
from ..torch_compat import Module, Tensor


class EvidenceHead(Module):
    """Token -> structured evidence nodes."""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.checkpoint_path: str | None = None
        self.code_to_finding: Dict[int, str] = {}

        ckpt = cfg.get("checkpoint_path") or cfg.get("checkpoint")
        if ckpt:
            p = Path(str(ckpt))
            if p.exists():
                payload = json.loads(p.read_text(encoding="utf-8"))
                mapping = payload.get("code_to_finding", {}) if isinstance(payload, dict) else {}
                if isinstance(mapping, dict):
                    self.checkpoint_path = str(p)
                    for k, v in mapping.items():
                        try:
                            code = int(k)
                        except Exception:
                            continue
                        self.code_to_finding[int(code)] = str(v)

    def forward(self, tokens: List[Token], embed_bank: Tensor) -> List[EvidenceNode]:
        """
        Args:
            tokens: selected tokens
            embed_bank: (N, d) embeddings aligned with tokens

        Returns:
            Evidence nodes with structured attrs & supported_token_ids.
        """
        default_finding = str(self.cfg.get("default_finding_type", "token"))
        nodes: List[EvidenceNode] = []
        for t in tokens:
            finding = default_finding
            if t.code is not None and int(t.code) in self.code_to_finding:
                finding = str(self.code_to_finding[int(t.code)])
            root_id = int(t.token_id)
            if t.parent_id is not None:
                # For max_levels=2 (repo-skeleton default for refinement), level-1 tokens point to a level-0 root.
                # This enables stable per-root sentence grouping in the planner for grounding experiments.
                root_id = int(t.parent_id)
            nodes.append(
                EvidenceNode(
                    eid=f"tok-{t.token_id}",
                    finding_type=str(finding),
                    attrs={
                        "side": "U",
                        "location": "U",
                        "size_bin": "U",
                        "certainty": "U",
                        "root_token_id": int(root_id),
                    },
                    supported_token_ids=[int(t.token_id)],
                )
            )
        return nodes

    def export(self) -> Dict[str, Any]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "mapping_size": int(len(self.code_to_finding)),
            "enabled": bool(self.code_to_finding),
        }


__all__ = [
    "EvidenceHead",
]
