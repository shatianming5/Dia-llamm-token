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

    def _shape_cdhw(self, volume: Tensor) -> List[int]:
        if hasattr(volume, "shape"):
            shape = list(getattr(volume, "shape"))
            if len(shape) == 4:
                return [int(x) for x in shape]

        c = len(volume)  # type: ignore[arg-type]
        d = len(volume[0])  # type: ignore[index]
        h = len(volume[0][0])  # type: ignore[index]
        w = len(volume[0][0][0])  # type: ignore[index]
        return [int(c), int(d), int(h), int(w)]

    def _grid_patch(self) -> int:
        grid = self.cfg.get("grid", {})
        patch = int(grid.get("patch", 8))
        return max(1, patch)

    def _voxel_spacing_xyz_mm(self) -> List[float]:
        spacing = self.cfg.get("voxel_spacing_mm", [1.0, 1.0, 1.0])
        if isinstance(spacing, (list, tuple)) and len(spacing) == 3:
            return [float(spacing[0]), float(spacing[1]), float(spacing[2])]
        return [1.0, 1.0, 1.0]

    @no_grad()
    def build_pyramid(self, volume: Tensor) -> TokenPyramid:
        """
        Args:
            volume: (C, D, H, W) float32

        Returns:
            Multi-level tokens, each token has omega_box_mm.
        """
        _, d, h, w = self._shape_cdhw(volume)
        patch = self._grid_patch()
        sx, sy, sz = self._voxel_spacing_xyz_mm()  # x,y,z per-voxel spacing (W,H,D)

        tokens: List[Token] = []
        token_id = 0
        for z0 in range(0, d, patch):
            z1 = min(d, z0 + patch)
            for y0 in range(0, h, patch):
                y1 = min(h, y0 + patch)
                for x0 in range(0, w, patch):
                    x1 = min(w, x0 + patch)
                    tokens.append(
                        Token(
                            token_id=token_id,
                            level=0,
                            omega_box_mm=(
                                float(x0) * sx,
                                float(x1) * sx,
                                float(y0) * sy,
                                float(y1) * sy,
                                float(z0) * sz,
                                float(z1) * sz,
                            ),
                        )
                    )
                    token_id += 1

        return TokenPyramid(tokens_by_level={0: tokens}, embed_bank_path={})

    @no_grad()
    def select_tokens(self, pyramid: TokenPyramid, active_nodes: List[int], budget_B: int) -> List[Token]:
        """Select tokens from pyramid (coarse + split-derived fine tokens)."""
        tokens = list(pyramid.tokens_by_level.get(0, []))
        tokens.sort(key=lambda t: int(t.token_id))
        if budget_B <= 0:
            return []
        return tokens[: int(budget_B)]


__all__ = [
    "TokenPyramid",
    "Tokenizer3D",
]
