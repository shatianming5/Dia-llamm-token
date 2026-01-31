from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


Box = Tuple[float, float, float, float, float, float]  # x0,x1,y0,y1,z0,z1 (mm)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def grounding_eval(
    run_json: str,
    out_dir: str,
    *,
    gt_boxes_by_sent: Dict[int, List[Box]],
) -> Dict[str, Any]:
    run = _load_json(Path(run_json))
    report = str(run.get("report", ""))
    sentences = _sentences(report)

    token_boxes = _token_boxes_from_run(run)
    cited_by_sent = _cited_token_ids_by_sent(run.get("citations", []))

    rows: List[Dict[str, Any]] = []
    eligible = 0
    hits_0 = 0
    hits_01 = 0
    sum_iou = 0.0

    for sid in range(len(sentences)):
        gt_boxes = gt_boxes_by_sent.get(int(sid), [])
        if not gt_boxes:
            continue
        eligible += 1

        cited_ids = cited_by_sent.get(int(sid), [])
        cited_boxes: List[Box] = []
        for tid in cited_ids:
            if int(tid) in token_boxes:
                cited_boxes.append(token_boxes[int(tid)])

        max_iou = 0.0
        for cb in cited_boxes:
            for gb in gt_boxes:
                max_iou = max(max_iou, float(_box_iou_3d(cb, gb)))

        hit0 = bool(max_iou > 0.0)
        hit01 = bool(max_iou >= 0.1)
        if hit0:
            hits_0 += 1
        if hit01:
            hits_01 += 1
        sum_iou += float(max_iou)

        rows.append(
            {
                "sent_id": int(sid),
                "sentence": sentences[sid],
                "n_cited": int(len(cited_boxes)),
                "n_gt": int(len(gt_boxes)),
                "max_iou": float(max_iou),
                "hit@0.0": int(hit0),
                "hit@0.1": int(hit01),
            }
        )

    metrics = {
        "n_sentences": int(len(sentences)),
        "n_eligible": int(eligible),
        "ground_hit@0.0": float(hits_0) / float(eligible) if eligible > 0 else 0.0,
        "ground_hit@0.1": float(hits_01) / float(eligible) if eligible > 0 else 0.0,
        "ground_mean_iou": float(sum_iou) / float(eligible) if eligible > 0 else 0.0,
    }

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    (out_path / "grounding_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_path / "grounding.json").write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with (out_path / "grounding.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sent_id", "n_cited", "n_gt", "max_iou", "hit@0.0", "hit@0.1"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})

    summary = {
        "timestamp_utc": _utc_now_iso(),
        "in_run_json": str(run_json),
        "out_dir": str(out_path),
    }
    (out_path / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return {"out_dir": str(out_path), "metrics_path": str(out_path / "grounding_metrics.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Grounding evaluation (GT boxes per sentence).")
    parser.add_argument("--run", required=True, help="Path to run.json")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--gt", default="", help="Optional GT JSON file (maps sent_id -> [box...])")
    parser.add_argument("--manifest", default="", help="Optional manifest.jsonl (must contain grounding_boxes_by_sent_mm)")
    parser.add_argument("--case-id", default="", help="Case ID to select from manifest (required if --manifest is set)")
    args = parser.parse_args()

    gt_boxes: Dict[int, List[Box]] = {}
    if args.gt:
        gt_boxes = _load_gt_boxes_by_sent_from_json(Path(args.gt))
    elif args.manifest:
        if not args.case_id:
            raise SystemExit("[ERR] --case-id is required when using --manifest")
        gt_boxes = _load_gt_boxes_by_sent_from_manifest(Path(args.manifest), str(args.case_id))
    else:
        raise SystemExit("[ERR] must provide --gt or --manifest/--case-id")

    result = grounding_eval(args.run, args.out, gt_boxes_by_sent=gt_boxes)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

