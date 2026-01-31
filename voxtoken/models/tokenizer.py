from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from ..schemas import Token
from ..torch_compat import Module, Tensor, no_grad


@dataclass
class TokenPyramid:
    """Multi-level tokens + lightweight topology for refinement."""

    tokens_by_level: Dict[int, List[Token]]
    token_by_id: Dict[int, Token]
    children_map: Dict[int, List[int]]  # parent token_id -> child token_ids
    embed_bank_path: Dict[int, str] = field(default_factory=dict)


class Tokenizer3D(Module):
    """Hierarchical 3D tokenizer interface (discrete-mainline + continuous MVP)."""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.checkpoint_path: str | None = None
        self.codebook_centers: List[float] | None = None

        ckpt = cfg.get("checkpoint_path") or cfg.get("checkpoint")
        if ckpt:
            p = Path(str(ckpt))
            if p.exists():
                payload = json.loads(p.read_text(encoding="utf-8"))
                cb = payload.get("codebook", {}) if isinstance(payload, dict) else {}
                centers = cb.get("centers", []) if isinstance(cb, dict) else []
                if isinstance(centers, list) and centers:
                    self.checkpoint_path = str(p)
                    self.codebook_centers = [float(x) for x in centers]

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

    def _grid_max_levels(self) -> int:
        grid = self.cfg.get("grid", {})
        max_levels = int(grid.get("max_levels", 1))
        return max(1, max_levels)

    def _voxel_spacing_xyz_mm(self) -> List[float]:
        spacing = self.cfg.get("voxel_spacing_mm", [1.0, 1.0, 1.0])
        if isinstance(spacing, (list, tuple)) and len(spacing) == 3:
            return [float(spacing[0]), float(spacing[1]), float(spacing[2])]
        return [1.0, 1.0, 1.0]

    def _token_code(self, volume: Tensor, *, x0: int, x1: int, y0: int, y1: int, z0: int, z1: int) -> int | None:
        centers = self.codebook_centers
        if not centers:
            return None

        # Compute patch mean (stdlib-only).
        try:
            c = len(volume)  # type: ignore[arg-type]
        except Exception:
            return None

        n = 0
        mean = 0.0
        for cc in range(int(c)):
            for zz in range(int(z0), int(z1)):
                for yy in range(int(y0), int(y1)):
                    row = volume[cc][zz][yy]  # type: ignore[index]
                    for xx in range(int(x0), int(x1)):
                        v = float(row[xx])
                        n += 1
                        mean += (v - mean) / float(n)
        if n <= 0:
            return None

        # Nearest-center quantization.
        best = 0
        best_d = abs(float(mean) - float(centers[0]))
        for i in range(1, len(centers)):
            d = abs(float(mean) - float(centers[i]))
            if d < best_d:
                best = int(i)
                best_d = float(d)
        return int(best)

    def export(self) -> Dict[str, Any]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "codes_enabled": bool(self.codebook_centers),
            "codebook_size": int(len(self.codebook_centers or [])),
        }

    @no_grad()
    def build_pyramid(self, volume: Tensor) -> TokenPyramid:
        """
        Args:
            volume: (C, D, H, W) float32

        Returns:
            Multi-level tokens, each token has omega_box_mm.
        """
        _, d, h, w = self._shape_cdhw(volume)
        patch0 = self._grid_patch()
        max_levels = self._grid_max_levels()
        sx, sy, sz = self._voxel_spacing_xyz_mm()  # x,y,z per-voxel spacing (W,H,D)

        patches: List[int] = [int(patch0)]
        for _ in range(1, int(max_levels)):
            p_next = max(1, int(patches[-1] // 2))
            if p_next == patches[-1]:
                break
            patches.append(p_next)

        tokens_by_level: Dict[int, List[Token]] = {}
        token_by_id: Dict[int, Token] = {}
        children_map: Dict[int, List[int]] = {}
        by_key: Dict[tuple[int, int, int, int], int] = {}

        token_id = 0
        for level, patch in enumerate(patches):
            level_tokens: List[Token] = []
            for z0 in range(0, d, patch):
                z1 = min(d, z0 + patch)
                for y0 in range(0, h, patch):
                    y1 = min(h, y0 + patch)
                    for x0 in range(0, w, patch):
                        x1 = min(w, x0 + patch)
                        parent_id = None
                        if level > 0:
                            parent_patch = patches[level - 1]
                            px0 = (int(x0) // int(parent_patch)) * int(parent_patch)
                            py0 = (int(y0) // int(parent_patch)) * int(parent_patch)
                            pz0 = (int(z0) // int(parent_patch)) * int(parent_patch)
                            parent_id = by_key.get((level - 1, px0, py0, pz0))

                        code = self._token_code(volume, x0=int(x0), x1=int(x1), y0=int(y0), y1=int(y1), z0=int(z0), z1=int(z1))
                        t = Token(
                            token_id=int(token_id),
                            level=int(level),
                            omega_box_mm=(
                                float(x0) * sx,
                                float(x1) * sx,
                                float(y0) * sy,
                                float(y1) * sy,
                                float(z0) * sz,
                                float(z1) * sz,
                            ),
                            parent_id=(int(parent_id) if parent_id is not None else None),
                            code=(int(code) if code is not None else None),
                        )
                        level_tokens.append(t)
                        token_by_id[int(token_id)] = t
                        by_key[(int(level), int(x0), int(y0), int(z0))] = int(token_id)

                        if parent_id is not None:
                            children_map.setdefault(int(parent_id), []).append(int(token_id))

                        token_id += 1

            tokens_by_level[int(level)] = level_tokens

        # Stable ordering for downstream determinism.
        for lvl in list(tokens_by_level.keys()):
            tokens_by_level[lvl].sort(key=lambda t: int(t.token_id))
        for pid in list(children_map.keys()):
            children_map[pid] = sorted(children_map[pid])

        # Populate Token.children_ids for convenience (still keep children_map as source of truth).
        for pid, cids in children_map.items():
            if pid in token_by_id:
                token_by_id[pid].children_ids = list(cids)

        return TokenPyramid(tokens_by_level=tokens_by_level, token_by_id=token_by_id, children_map=children_map)

    @no_grad()
    def select_tokens(self, pyramid: TokenPyramid, active_nodes: List[int], budget_B: int) -> List[Token]:
        """Select tokens from pyramid (coarse + split-derived fine tokens)."""
        if budget_B <= 0:
            return []
        if active_nodes:
            out = [pyramid.token_by_id[int(tid)] for tid in active_nodes if int(tid) in pyramid.token_by_id]
            out.sort(key=lambda t: int(t.token_id))
            return out[: int(budget_B)]

        # Default: start from the coarsest level only, leaving room for refinement.
        tokens = list(pyramid.tokens_by_level.get(0, []))
        tokens.sort(key=lambda t: int(t.token_id))
        return tokens[: min(int(budget_B), len(tokens))]


__all__ = [
    "TokenPyramid",
    "Tokenizer3D",
]
