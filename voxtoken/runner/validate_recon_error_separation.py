from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml  # pyyaml

    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(cfg or {})


def _target_shape_cdhw(cfg: Dict[str, Any]) -> List[int]:
    vol_cfg = dict(cfg.get("volume", {}))
    target = vol_cfg.get("target_shape_cdhw") or vol_cfg.get("shape_cdhw") or [1, 8, 8, 8]
    try:
        c, d, h, w = [int(x) for x in target]
    except Exception:
        return [1, 8, 8, 8]
    return [max(1, c), max(1, d), max(1, h), max(1, w)]


def _token_variances(tokens: Sequence[Dict[str, Any]], volume: Any, *, spacing_xyz_mm: Sequence[float]) -> List[float]:
    sx, sy, sz = [float(x) for x in (list(spacing_xyz_mm) + [1.0, 1.0, 1.0])[:3]]

    out: List[float] = []
    for t in tokens:
        box = t.get("omega_box_mm", None)
        if not (isinstance(box, (list, tuple)) and len(box) == 6):
            continue
        x0, x1, y0, y1, z0, z1 = [float(x) for x in box]
        ix0 = int(round(x0 / sx))
        ix1 = int(round(x1 / sx))
        iy0 = int(round(y0 / sy))
        iy1 = int(round(y1 / sy))
        iz0 = int(round(z0 / sz))
        iz1 = int(round(z1 / sz))

        try:
            c = len(volume)
            d = len(volume[0])
            h = len(volume[0][0])
            w = len(volume[0][0][0])
        except Exception:
            continue

        iz0 = max(0, min(int(iz0), int(d)))
        iz1 = max(0, min(int(iz1), int(d)))
        iy0 = max(0, min(int(iy0), int(h)))
        iy1 = max(0, min(int(iy1), int(h)))
        ix0 = max(0, min(int(ix0), int(w)))
        ix1 = max(0, min(int(ix1), int(w)))
        if iz1 <= iz0 or iy1 <= iy0 or ix1 <= ix0:
            out.append(0.0)
            continue

        # Welford variance over (C,D,H,W) patch.
        n = 0
        mean = 0.0
        m2 = 0.0
        for cc in range(int(c)):
            for zz in range(int(iz0), int(iz1)):
                for yy in range(int(iy0), int(iy1)):
                    row = volume[cc][zz][yy]
                    for xx in range(int(ix0), int(ix1)):
                        v = float(row[xx])
                        n += 1
                        delta = v - mean
                        mean += delta / float(n)
                        delta2 = v - mean
                        m2 += delta * delta2
        out.append(0.0 if n <= 0 else float(m2 / float(n)))

    return out


def validate_recon_error_separation(
    *,
    cfg: Dict[str, Any],
    require_delta_ge: float = 0.05,
    require_max_ge: float = 0.05,
    require_min_le: float = 0.001,
    seed: int = 0,
) -> Tuple[List[str], Dict[str, Any]]:
    errors: List[str] = []

    from voxtoken.models.tokenizer import Tokenizer3D
    from voxtoken.runner.infer_refine import _make_dummy_volume_with_cfg

    shape = _target_shape_cdhw(cfg)
    tok_cfg = dict(cfg.get("tokenizer", {}))
    vol_cfg = dict(cfg.get("volume", {}))

    grid = dict(tok_cfg.get("grid", {}))
    patch = int(grid.get("patch", 4))
    patch = max(1, int(patch))

    pattern_cfg = dict(vol_cfg)
    pattern_cfg.setdefault("region_patch", int(patch))
    pattern_cfg.setdefault("high_scale", 1.0)
    pattern_cfg.setdefault("low_scale", 0.01)

    volume = _make_dummy_volume_with_cfg(shape, seed=int(seed), pattern="region_noise", pattern_cfg=pattern_cfg)
    tokenizer = Tokenizer3D(tok_cfg)
    pyramid = tokenizer.build_pyramid(volume)
    tokens0 = pyramid.tokens_by_level.get(0, [])
    token_dicts = [t.__dict__ for t in tokens0]

    spacing = tok_cfg.get("voxel_spacing_mm", [1.0, 1.0, 1.0])
    if not (isinstance(spacing, (list, tuple)) and len(spacing) == 3):
        spacing = [1.0, 1.0, 1.0]

    vars_ = _token_variances(token_dicts, volume, spacing_xyz_mm=spacing)
    if not vars_:
        errors.append("no token variances computed (unexpected)")
        stats = {"n": 0, "min": 0.0, "max": 0.0, "delta": 0.0}
        return errors, stats

    vmin = float(min(vars_))
    vmax = float(max(vars_))
    delta = float(vmax - vmin)

    if delta < float(require_delta_ge):
        errors.append(f"recon_error dynamic range too small: delta({delta}) < required({float(require_delta_ge)})")
    if vmax < float(require_max_ge):
        errors.append(f"recon_error max too small: max({vmax}) < required({float(require_max_ge)})")
    if vmin > float(require_min_le):
        errors.append(f"recon_error min too large: min({vmin}) > required({float(require_min_le)})")

    stats = {"n": int(len(vars_)), "min": vmin, "max": vmax, "delta": delta}
    return errors, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate that recon_error has strong dynamic range on a structured dummy volume.")
    parser.add_argument("--config", required=True, help="YAML config (uses tokenizer/volume shape settings)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--require-delta-ge", type=float, default=0.05)
    parser.add_argument("--require-max-ge", type=float, default=0.05)
    parser.add_argument("--require-min-le", type=float, default=0.001)
    args = parser.parse_args()

    cfg = _load_yaml(Path(args.config))
    errors, stats = validate_recon_error_separation(
        cfg=cfg,
        require_delta_ge=float(args.require_delta_ge),
        require_max_ge=float(args.require_max_ge),
        require_min_le=float(args.require_min_le),
        seed=int(args.seed),
    )
    if errors:
        for e in errors:
            print(f"[ERR] {e}")
        print(json.dumps(stats, ensure_ascii=False))
        sys.exit(1)
    print("[OK] recon_error separation validated")
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()

