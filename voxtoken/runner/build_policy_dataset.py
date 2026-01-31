from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..models.tokenizer import TokenPyramid, Tokenizer3D
from ..schemas import Token, TokenFeatures
from .infer_refine import _make_dummy_volume_with_cfg  # repo-skeleton reuse


Box = Tuple[float, float, float, float, float, float]  # x0,x1,y0,y1,z0,z1 (mm)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_int_hash(s: str) -> int:
    # Keep consistent with voxtoken.runner.infer_refine (avoid Python's randomized hash()).
    h = 0
    for ch in str(s):
        h = (h * 131 + ord(ch)) % 2147483647
    return int(h)


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml  # pyyaml

    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(cfg or {})


def _load_manifest_row(manifest_jsonl: Path, case_id: str | None) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for line in manifest_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            rows.append(obj)
    if not rows:
        raise ValueError("manifest has no rows")
    if case_id:
        want = str(case_id).strip()
        for row in rows:
            if str(row.get("case_id", "")).strip() == want:
                return row
        raise KeyError(f"case_id not found in manifest: {want}")
    return rows[0]


def _parse_gt_boxes_by_sent(row: Dict[str, Any]) -> Dict[int, List[Box]]:
    raw = row.get("grounding_boxes_by_sent_mm", {}) or row.get("gt_boxes_by_sent_mm", {})
    if not isinstance(raw, dict):
        return {}
    out: Dict[int, List[Box]] = {}
    for k, v in raw.items():
        try:
            sid = int(k)
        except Exception:
            continue
        if not isinstance(v, list):
            continue
        boxes: List[Box] = []
        for b in v:
            if not isinstance(b, (list, tuple)) or len(b) != 6:
                continue
            try:
                boxes.append(tuple(float(x) for x in b))  # type: ignore[assignment]
            except Exception:
                continue
        if boxes:
            out[int(sid)] = boxes
    return out


def _box_iou_3d(a: Box, b: Box) -> float:
    ax0, ax1, ay0, ay1, az0, az1 = [float(x) for x in a]
    bx0, bx1, by0, by1, bz0, bz1 = [float(x) for x in b]

    ix0 = max(ax0, bx0)
    ix1 = min(ax1, bx1)
    iy0 = max(ay0, by0)
    iy1 = min(ay1, by1)
    iz0 = max(az0, bz0)
    iz1 = min(az1, bz1)

    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0) * max(0.0, iz1 - iz0)
    if inter <= 0.0:
        return 0.0

    va = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0) * max(0.0, az1 - az0)
    vb = max(0.0, bx1 - bx0) * max(0.0, by1 - by0) * max(0.0, bz1 - bz0)
    union = va + vb - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def _best_iou(pred: List[Box], gt: List[Box]) -> float:
    best = 0.0
    for pb in pred:
        for gb in gt:
            best = max(best, float(_box_iou_3d(pb, gb)))
    return float(best)


def _token_feature_vector(
    token: Token,
    *,
    volume: Any,
    voxel_spacing_mm: List[float],
) -> TokenFeatures:
    sx, sy, sz = [float(x) for x in (voxel_spacing_mm or [1.0, 1.0, 1.0])]

    def _shape_cdhw(vol: Any) -> Tuple[int, int, int, int]:
        try:
            c = len(vol)
            d = len(vol[0])
            h = len(vol[0][0])
            w = len(vol[0][0][0])
            return int(c), int(d), int(h), int(w)
        except Exception:
            return 1, 1, 1, 1

    def _box_mm_to_zyx_slices(box_mm: Box) -> Tuple[int, int, int, int, int, int]:
        x0, x1, y0, y1, z0, z1 = box_mm
        ix0 = int(round(float(x0) / sx))
        ix1 = int(round(float(x1) / sx))
        iy0 = int(round(float(y0) / sy))
        iy1 = int(round(float(y1) / sy))
        iz0 = int(round(float(z0) / sz))
        iz1 = int(round(float(z1) / sz))
        return iz0, iz1, iy0, iy1, ix0, ix1

    def _patch_variance() -> float:
        c, d, h, w = _shape_cdhw(volume)
        z0, z1, y0, y1, x0, x1 = _box_mm_to_zyx_slices(token.omega_box_mm)
        z0 = max(0, min(int(z0), int(d)))
        z1 = max(0, min(int(z1), int(d)))
        y0 = max(0, min(int(y0), int(h)))
        y1 = max(0, min(int(y1), int(h)))
        x0 = max(0, min(int(x0), int(w)))
        x1 = max(0, min(int(x1), int(w)))
        if z1 <= z0 or y1 <= y0 or x1 <= x0:
            return 0.0

        n = 0
        mean = 0.0
        m2 = 0.0
        for cc in range(int(c)):
            for zz in range(int(z0), int(z1)):
                for yy in range(int(y0), int(y1)):
                    row = volume[cc][zz][yy]
                    for xx in range(int(x0), int(x1)):
                        v = float(row[xx])
                        n += 1
                        delta = v - mean
                        mean += delta / float(n)
                        delta2 = v - mean
                        m2 += delta * delta2
        if n <= 0:
            return 0.0
        return float(m2 / float(n))

    var = float(_patch_variance())
    return TokenFeatures(
        token_id=int(token.token_id),
        level=int(token.level),
        recon_error=float(var),
        evidence_entropy=float(math.log1p(max(0.0, float(var)))),
        citation_pressure=1.0,
        history_splits=int(len(token.children_ids)),
    )


def build_policy_dataset(
    *,
    manifest_jsonl: Path,
    case_id: str | None,
    cfg: Dict[str, Any],
    out_jsonl: Path,
) -> Dict[str, Any]:
    row = _load_manifest_row(manifest_jsonl, case_id)
    cid = str(row.get("case_id", "")).strip() or "case-0000"
    gt_by_sent = _parse_gt_boxes_by_sent(row)
    if not gt_by_sent:
        raise ValueError("manifest row has no grounding_boxes_by_sent_mm / gt_boxes_by_sent_mm")

    tokenizer = Tokenizer3D(dict(cfg.get("tokenizer", {})))
    vol_cfg = cfg.get("volume", {})
    if not isinstance(vol_cfg, dict):
        vol_cfg = {}
    vol_cfg = dict(vol_cfg)

    target = vol_cfg.get("target_shape_cdhw") or vol_cfg.get("shape_cdhw") or [1, 8, 8, 8]
    try:
        c, d, h, w = [int(x) for x in target]
    except Exception:
        c, d, h, w = 1, 8, 8, 8
    volume = _make_dummy_volume_with_cfg([int(c), int(d), int(h), int(w)], seed=_stable_int_hash(cid), pattern=str(vol_cfg.get("pattern", "gradient")), pattern_cfg=vol_cfg)

    pyramid: TokenPyramid = tokenizer.build_pyramid(volume)
    roots = list(pyramid.tokens_by_level.get(0, []))
    roots.sort(key=lambda t: int(t.token_id))

    spacing = tokenizer.cfg.get("voxel_spacing_mm", [1.0, 1.0, 1.0])
    if not isinstance(spacing, list) or len(spacing) != 3:
        spacing = [1.0, 1.0, 1.0]
    spacing = [float(x) for x in spacing]

    out_rows: List[str] = []
    for t in roots:
        feats = _token_feature_vector(t, volume=volume, voxel_spacing_mm=spacing)
        sid = int(t.token_id)
        gt_boxes = gt_by_sent.get(int(sid), [])
        if not gt_boxes:
            # Still emit the row (reward=0) to keep dataset shape predictable.
            reward = 0.0
        else:
            before = _best_iou([t.omega_box_mm], gt_boxes)
            child_ids = list(pyramid.children_map.get(int(t.token_id), []))
            child_boxes = [pyramid.token_by_id[cid].omega_box_mm for cid in child_ids if cid in pyramid.token_by_id]
            after = _best_iou(child_boxes, gt_boxes) if child_boxes else float(before)
            reward = float(after) - float(before)

        out_rows.append(
            json.dumps(
                {
                    "case_id": str(cid),
                    "token_id": int(t.token_id),
                    "recon_error": float(feats.recon_error),
                    "evidence_entropy": float(feats.evidence_entropy),
                    "citation_pressure": float(feats.citation_pressure),
                    "history_splits": int(feats.history_splits),
                    "reward": float(reward),
                },
                ensure_ascii=False,
            )
        )

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.write_text("\n".join(out_rows) + ("\n" if out_rows else ""), encoding="utf-8")

    summary = {
        "timestamp_utc": _utc_now_iso(),
        "case_id": str(cid),
        "manifest": str(manifest_jsonl),
        "out_jsonl": str(out_jsonl),
        "n_rows": int(len(out_rows)),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a small policy training dataset from GT grounding targets.")
    parser.add_argument("--manifest", required=True, help="Processed manifest.jsonl containing grounding_boxes_by_sent_mm")
    parser.add_argument("--case-id", default="", help="Case ID to select (defaults to first row)")
    parser.add_argument("--config", required=True, help="YAML config (reuses tokenizer/volume settings)")
    parser.add_argument("--out", required=True, help="Output dataset JSONL path")
    args = parser.parse_args()

    cfg = _load_yaml(Path(args.config))
    summary = build_policy_dataset(
        manifest_jsonl=Path(args.manifest),
        case_id=(str(args.case_id).strip() if args.case_id else None),
        cfg=cfg,
        out_jsonl=Path(args.out),
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

