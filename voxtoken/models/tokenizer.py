from __future__ import annotations

from typing import Any, Dict, List, NamedTuple

from ..schemas import Token
from ..torch_compat import Module, Tensor, no_grad


class TokenPyramid(NamedTuple):
    """Multi-level tokens + on-disk embedding bank refs."""

    tokens_by_level: Dict[int, List[Token]]
    embed_bank_path: Dict[int, str]


class Tokenizer3D(Module):
    """Hierarchical 3D tokenizer interface (discrete-mainline + continuous MVP)."""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg

    @no_grad()
    def build_pyramid(self, volume: Tensor) -> TokenPyramid:
        """
        Args:
            volume: (C, D, H, W) float32

        Returns:
            Multi-level tokens, each token has omega_box_mm.
        """
        raise NotImplementedError

    @no_grad()
    def select_tokens(self, pyramid: TokenPyramid, active_nodes: List[int], budget_B: int) -> List[Token]:
        """Select tokens from pyramid (coarse + split-derived fine tokens)."""
        raise NotImplementedError


__all__ = [
    "TokenPyramid",
    "Tokenizer3D",
]

