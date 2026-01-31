from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple


Box = Tuple[float, float, float, float, float, float]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sentences(report: str) -> List[str]:
    return [s.strip() for s in str(report or "").splitlines() if s.strip()]


def _token_boxes_from_run(run: Dict[str, Any]) -> Dict[int, Box]:
    out: Dict[int, Box] = {}
    tokens = run.get("tokens", [])
    if not isinstance(tokens, list):
        return out
    for t in tokens:
        if not isinstance(t, dict):
            continue
        tid = t.get("token_id", None)
        box = t.get("omega_box_mm", None)
        if not isinstance(box, (list, tuple)) or len(box) != 6:
            continue
        try:
            tid_i = int(tid)
            out[int(tid_i)] = tuple(float(x) for x in box)  # type: ignore[assignment]
        except Exception:
            continue
    return out


def _cited_token_ids_by_sent(citations: Any) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    if not isinstance(citations, list):
        return out
    for c in citations:
        if not isinstance(c, dict):
            continue
        try:
            sid = int(c.get("sent_id", -1))
        except Exception:
            continue
        raw = c.get("cited_token_ids", []) or []
        ids: List[int] = []
        if isinstance(raw, list):
            for x in raw:
                try:
                    ids.append(int(x))
                except Exception:
                    continue
        out[int(sid)] = ids
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


def _load_gt_boxes_by_sent_from_manifest(manifest_jsonl: Path, case_id: str) -> Dict[int, List[Box]]:
    want = str(case_id).strip()
    for line in manifest_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            continue
        if str(obj.get("case_id", "")).strip() != want:
            continue
        raw = obj.get("grounding_boxes_by_sent_mm", {}) or obj.get("gt_boxes_by_sent_mm", {})
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
    raise KeyError(f"case_id not found in manifest: {want}")


def _load_gt_boxes_by_sent_from_json(path: Path) -> Dict[int, List[Box]]:
    payload = _load_json(path)
    raw = (
        payload.get("grounding_boxes_by_sent_mm", None)
        or payload.get("gt_boxes_by_sent_mm", None)
        or payload.get("boxes_by_sent_mm", None)
        or payload.get("boxes_by_sent", None)
        or payload
    )
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


def _write_svg(path: Path, *, title: str, cited: List[Box], gt: List[Box]) -> None:
    """
    Minimal 2D overlay: project boxes onto (x,y) plane and draw GT (green) and cited (red).
    """
    boxes = [*cited, *gt]
    if not boxes:
        return

    min_x = min(float(b[0]) for b in boxes)
    max_x = max(float(b[1]) for b in boxes)
    min_y = min(float(b[2]) for b in boxes)
    max_y = max(float(b[3]) for b in boxes)
    dx = max(1e-6, max_x - min_x)
    dy = max(1e-6, max_y - min_y)

    w = 360
    h = 360
    pad = 20

    def sx(x: float) -> float:
        return float(pad) + (float(x) - float(min_x)) / float(dx) * float(w)

    def sy(y: float) -> float:
        # SVG y increases downward; keep it consistent by flipping.
        return float(pad) + (float(max_y) - float(y)) / float(dy) * float(h)

    def rect(box: Box, stroke: str) -> str:
        x0, x1, y0, y1, _z0, _z1 = box
        x = sx(float(x0))
        y = sy(float(y1))
        rw = max(1.0, sx(float(x1)) - sx(float(x0)))
        rh = max(1.0, sy(float(y0)) - sy(float(y1)))
        return f'<rect x="{x:.2f}" y="{y:.2f}" width="{rw:.2f}" height="{rh:.2f}" fill="none" stroke="{stroke}" stroke-width="2" />'

    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w + 2*pad}" height="{h + 2*pad + 24}">')
    parts.append(f'<text x="{pad}" y="{pad + h + 18}" font-family="monospace" font-size="12">{title}</text>')
    for b in gt:
        parts.append(rect(b, "#2ca02c"))
    for b in cited:
        parts.append(rect(b, "#d62728"))
    parts.append("</svg>\n")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Grounding visualization (boxes overlay; repo-skeleton friendly).")
    parser.add_argument("--in", dest="run_json", required=True, help="Path to run.json")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--gt", default="", help="Optional GT JSON file (maps sent_id -> [box...])")
    parser.add_argument("--manifest", default="", help="Optional manifest.jsonl (contains grounding_boxes_by_sent_mm)")
    parser.add_argument("--case-id", default="", help="Case ID to select from manifest (required if --manifest is set)")
    parser.add_argument("--max-sentences", type=int, default=50)
    args = parser.parse_args()

    run_path = Path(args.run_json)
    if not run_path.exists():
        print(f"[ERR] missing run.json: {run_path}", file=sys.stderr)
        sys.exit(2)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    run = _load_json(run_path)
    report = str(run.get("report", ""))
    sentences = _sentences(report)[: max(0, int(args.max_sentences))]
    token_boxes = _token_boxes_from_run(run)
    cited_by_sent = _cited_token_ids_by_sent(run.get("citations", []))

    if args.gt:
        gt_by_sent = _load_gt_boxes_by_sent_from_json(Path(args.gt))
    elif args.manifest:
        if not args.case_id:
            print("[ERR] --case-id is required when using --manifest", file=sys.stderr)
            sys.exit(2)
        gt_by_sent = _load_gt_boxes_by_sent_from_manifest(Path(args.manifest), str(args.case_id))
    else:
        print("[ERR] must provide --gt or --manifest/--case-id", file=sys.stderr)
        sys.exit(2)

    rows: List[Dict[str, Any]] = []
    for sid, sent in enumerate(sentences):
        gt_boxes = gt_by_sent.get(int(sid), [])
        cited_boxes: List[Box] = []
        for tid in cited_by_sent.get(int(sid), []):
            if int(tid) in token_boxes:
                cited_boxes.append(token_boxes[int(tid)])

        max_iou = 0.0
        for cb in cited_boxes:
            for gb in gt_boxes:
                max_iou = max(max_iou, float(_box_iou_3d(cb, gb)))

        rows.append(
            {
                "sent_id": int(sid),
                "sentence": str(sent),
                "gt_boxes_mm": [list(b) for b in gt_boxes],
                "cited_boxes_mm": [list(b) for b in cited_boxes],
                "max_iou": float(max_iou),
            }
        )

        if gt_boxes:
            _write_svg(
                out_dir / f"sent_{sid:03d}.svg",
                title=f"sent={sid} iou={max_iou:.3f}",
                cited=cited_boxes,
                gt=gt_boxes,
            )

    (out_dir / "grounding_overlay.json").write_text(
        json.dumps({"rows": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(
        json.dumps({"timestamp_utc": _utc_now_iso(), "in_run_json": str(run_path), "out_dir": str(out_dir)}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps({"out_dir": str(out_dir), "overlay_path": str(out_dir / "grounding_overlay.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
