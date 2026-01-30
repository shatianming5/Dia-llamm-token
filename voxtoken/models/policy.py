from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..schemas import TokenFeatures
from ..torch_compat import Module


class SplitPolicy(Module):
    """Learned split policy interface (contextual bandit / offline RL)."""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg

    def score(self, feats: List[TokenFeatures]) -> List[Tuple[int, float]]:
        """Returns (token_id, score)."""
        scored: List[Tuple[int, float]] = []
        for f in feats:
            s = float(f.recon_error) + float(f.evidence_entropy) + float(f.citation_pressure) - float(f.history_splits)
            scored.append((int(f.token_id), s))
        scored.sort(key=lambda x: (-float(x[1]), int(x[0])))
        return scored


__all__ = [
    "SplitPolicy",
]
