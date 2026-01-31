from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

BoxMM = Tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class GridSpec:
    patch: int
    voxel_spacing_mm: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # x,y,z spacing


def grid_cell_to_box_mm(x0: int, x1: int, y0: int, y1: int, z0: int, z1: int, spec: GridSpec) -> BoxMM:
    sx, sy, sz = (float(spec.voxel_spacing_mm[0]), float(spec.voxel_spacing_mm[1]), float(spec.voxel_spacing_mm[2]))
    return (
        float(x0) * sx,
        float(x1) * sx,
        float(y0) * sy,
        float(y1) * sy,
        float(z0) * sz,
        float(z1) * sz,
    )

