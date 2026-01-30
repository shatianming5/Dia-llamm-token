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
        raise NotImplementedError


__all__ = [
    "SplitPolicy",
]

