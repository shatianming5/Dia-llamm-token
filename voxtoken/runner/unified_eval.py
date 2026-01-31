from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..verify.extract_slots import extract_slots_from_report


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

Box = Tuple[float, float, float, float, float, float]  # x0,x1,y0,y1,z0,z1 (mm)


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


def _box_iou_3d(a: Tuple[float, float, float, float, float, float], b: Tuple[float, float, float, float, float, float]) -> float:
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


def _compute_unsupported_rate(report: str, citations: List[Dict[str, Any]], plan: Dict[str, Any]) -> float:
    sentences = [s.strip() for s in report.splitlines() if s.strip()]
    if not sentences:
        return 0.0

    cited_by_sent = {int(c.get("sent_id", -1)): c.get("cited_token_ids", []) for c in citations}
    supported_by_finding: Dict[str, Set[int]] = {}
    facts = plan.get("facts", []) if isinstance(plan, dict) else []
    if isinstance(facts, list):
        for f in facts:
            if not isinstance(f, dict):
                continue
            ft = str(f.get("finding_type", "")).strip().lower()
            if not ft:
                continue
            tids_raw = f.get("supported_token_ids", []) or []
            if not isinstance(tids_raw, list):
                continue
            tids = set()
            for x in tids_raw:
                try:
                    tids.add(int(x))
                except Exception:
                    continue
            if tids:
                supported_by_finding.setdefault(ft, set()).update(tids)

    unsupported = 0
    for sent_id in range(len(sentences)):
        token_ids = cited_by_sent.get(sent_id, [])
        if not token_ids:
            unsupported += 1
            continue

        if supported_by_finding:
            sent = sentences[sent_id]
            finding = sent.split("(", 1)[0].strip().lower()
            if finding and finding in supported_by_finding:
                cited = set()
                if isinstance(token_ids, list):
                    for x in token_ids:
                        try:
                            cited.add(int(x))
                        except Exception:
                            continue
                if not (cited & supported_by_finding[finding]):
                    unsupported += 1
    return unsupported / float(len(sentences))


def _slot_key(d: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
    return (
        str(d.get("finding_type", "")).strip(),
        str(d.get("side", "U")).strip(),
        str(d.get("location", "U")).strip(),
        str(d.get("size_bin", "U")).strip(),
        str(d.get("certainty", "U")).strip(),
    )


def _multiset_f1(gold: List[Dict[str, Any]], pred: List[Dict[str, Any]]) -> float:
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0

    g = Counter(_slot_key(x) for x in gold)
    p = Counter(_slot_key(x) for x in pred)
    match = 0
    for k, gv in g.items():
        match += min(int(gv), int(p.get(k, 0)))

    n_pred = int(sum(p.values()))
    n_gold = int(sum(g.values()))
    if n_pred <= 0 or n_gold <= 0:
        return 0.0

    prec = float(match) / float(n_pred)
    rec = float(match) / float(n_gold)
    if prec + rec <= 0.0:
        return 0.0
    return 2.0 * prec * rec / (prec + rec)


def unified_eval(run_json: str, out_dir: str, *, case_id: str = "case-0000", budget_B: int = 0) -> Dict[str, Any]:
    run_path = Path(run_json)
    run = _load_json(run_path)

    report = str(run.get("report", ""))
    sentences = [s.strip() for s in report.splitlines() if s.strip()]
    n_sentences = int(len(sentences))
    citations = list(run.get("citations", []))
    plan = run.get("plan", {})
    if not isinstance(plan, dict):
        plan = {}
    unsupported_rate = _compute_unsupported_rate(report, citations, plan)
    gold_facts = []
    if isinstance(plan, dict):
        gold_facts = list(plan.get("facts", []) or [])
    pred_facts = [x.__dict__ for x in extract_slots_from_report(report)]
    slot_f1 = float(_multiset_f1(gold_facts, pred_facts))

    trace = list(run.get("trace", []))
    tokens_used = int(run.get("tokens_used", 0))
    if trace:
        last = trace[-1]
        if isinstance(last, dict) and "budget_used" in last:
            tokens_used = int(last.get("budget_used", tokens_used))
        if budget_B == 0 and isinstance(last, dict) and "budget_total" in last:
            budget_B = int(last.get("budget_total", budget_B))

    # Prefer the run's own budget if not provided via CLI / trace.
    if budget_B == 0 and "budget_B" in run:
        try:
            budget_B = int(run.get("budget_B", 0))
        except Exception:
            budget_B = int(budget_B)

    verifier_score = float(run.get("verifier_score", 0.0))

    latency_total = 0.0
    latency = run.get("latency_ms", {})
    if isinstance(latency, dict) and "total" in latency:
        try:
            latency_total = float(latency.get("total", 0.0))
        except Exception:
            latency_total = 0.0

    issues = run.get("issues", [])
    if not isinstance(issues, list):
        issues = []

    # Grounding: if GT sentence boxes are provided, compute hit-rate vs GT; otherwise fall back
    # to a repo-skeleton proxy (use the plan's supported_token boxes).
    token_box_by_id: Dict[int, Tuple[float, float, float, float, float, float]] = {}
    tokens_raw = run.get("tokens", [])
    if isinstance(tokens_raw, list):
        for t in tokens_raw:
            if not isinstance(t, dict):
                continue
            tid = t.get("token_id", None)
            box = t.get("omega_box_mm", None)
            if not isinstance(box, (list, tuple)) or len(box) != 6:
                continue
            try:
                tid_i = int(tid)
                box_t = tuple(float(x) for x in box)  # type: ignore[misc]
            except Exception:
                continue
            token_box_by_id[int(tid_i)] = box_t  # type: ignore[assignment]

    cited_by_sent: Dict[int, List[int]] = {}
    for c in citations:
        if not isinstance(c, dict):
            continue
        try:
            sid = int(c.get("sent_id", -1))
        except Exception:
            continue
        tids = []
        raw = c.get("cited_token_ids", []) or []
        if isinstance(raw, list):
            for x in raw:
                try:
                    tids.append(int(x))
                except Exception:
                    continue
        cited_by_sent[int(sid)] = tids

    # Optional GT boxes (paper-facing): load from run.meta or via CLI-injected globals (set in main()).
    gt_boxes_by_sent: Optional[Dict[int, List[Box]]] = None
    if isinstance(run.get("_gt_boxes_by_sent_mm", None), dict):
        try:
            gt_boxes_by_sent = {int(k): [tuple(float(x) for x in b) for b in v] for k, v in run["_gt_boxes_by_sent_mm"].items() if isinstance(v, list)}  # type: ignore[assignment]
        except Exception:
            gt_boxes_by_sent = None

    if gt_boxes_by_sent is None:
        facts_raw = plan.get("facts", [])
        if not isinstance(facts_raw, list):
            facts_raw = []
        gt_boxes_by_sent = {}
        for sent_id in range(n_sentences):
            if sent_id >= len(facts_raw):
                continue
            fact = facts_raw[sent_id]
            if not isinstance(fact, dict):
                continue
            supported_ids = fact.get("supported_token_ids", []) or []
            if not isinstance(supported_ids, list):
                continue
            boxes: List[Box] = []
            for x in supported_ids:
                try:
                    tid = int(x)
                except Exception:
                    continue
                if tid in token_box_by_id:
                    boxes.append(token_box_by_id[tid])  # type: ignore[arg-type]
            if boxes:
                gt_boxes_by_sent[int(sent_id)] = boxes

    eligible = 0
    hits_0 = 0
    hits_01 = 0
    sum_iou = 0.0
    for sent_id in range(n_sentences):
        gt_boxes = gt_boxes_by_sent.get(int(sent_id), []) if gt_boxes_by_sent else []
        if not gt_boxes:
            continue
        eligible += 1

        cited_ids = cited_by_sent.get(int(sent_id), [])
        cited_boxes: List[Box] = []
        for x in cited_ids:
            if int(x) in token_box_by_id:
                cited_boxes.append(token_box_by_id[int(x)])  # type: ignore[arg-type]

        max_iou = 0.0
        for cb in cited_boxes:
            for gb in gt_boxes:
                max_iou = max(max_iou, float(_box_iou_3d(cb, gb)))

        sum_iou += float(max_iou)
        if max_iou > 0.0:
            hits_0 += 1
        if max_iou >= 0.1:
            hits_01 += 1

    ground_hit_0 = float(hits_0) / float(eligible) if eligible > 0 else 0.0
    ground_hit_01 = float(hits_01) / float(eligible) if eligible > 0 else 0.0
    ground_mean_iou = float(sum_iou) / float(eligible) if eligible > 0 else 0.0

    missing_slot_per_report = 0
    inconsistency_per_report = 0
    overclaim_sent_ids: Set[int] = set()
    for it in issues:
        if not isinstance(it, dict):
            continue
        t = str(it.get("type", "")).strip().lower()
        if t == "missing_slot":
            missing_slot_per_report += 1
        elif t == "inconsistency":
            inconsistency_per_report += 1
        elif t == "overclaim":
            span = it.get("span", None)
            if isinstance(span, (list, tuple)) and span:
                try:
                    overclaim_sent_ids.add(int(span[0]))
                except Exception:
                    continue

    overclaim_sent_pct = 0.0
    if n_sentences > 0:
        overclaim_sent_pct = 100.0 * float(len(overclaim_sent_ids)) / float(n_sentences)

    metrics: Dict[str, Any] = {
        "case_id": case_id,
        "budget_B": budget_B,
        "tokens_used": int(tokens_used),
        "latency_ms": {"total": float(latency_total)},
        "slot_f1": float(slot_f1),
        "unsupported_rate": float(unsupported_rate),
        "verifier_score": float(verifier_score),

        # Paper-facing aliases (docs/plan.md section 7.*).
        "tokens_final": int(tokens_used),
        "lat_total_ms": float(latency_total),
        "slot_f1_micro": float(slot_f1),
        "ground_hit@0.0": float(ground_hit_0),
        "ground_hit@0.1": float(ground_hit_01),
        "ground_mean_iou": float(ground_mean_iou),
        "unsupported_sent_pct": 100.0 * float(unsupported_rate),
        "overclaim_sent_pct": float(overclaim_sent_pct),
        "missing_slot_per_report": int(missing_slot_per_report),
        "inconsistency_per_report": int(inconsistency_per_report),
    }

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_path / "metrics.jsonl").write_text(json.dumps(metrics, ensure_ascii=False) + "\n", encoding="utf-8")

    return {"out_dir": str(out_path), "metrics_path": str(out_path / "metrics.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified evaluation (schema placeholder).")
    parser.add_argument("--in", dest="run_json", required=True, help="Path to run.json")
    parser.add_argument(
        "--out",
        default="",
        help="Output directory (default: write next to run.json, per docs/plan.md artifacts/runs convention).",
    )
    parser.add_argument("--gt", default="", help="Optional GT JSON file (maps sent_id -> [box...])")
    parser.add_argument("--manifest", default="", help="Optional manifest.jsonl (contains grounding_boxes_by_sent_mm)")
    parser.add_argument("--case-id-manifest", default="", help="Case ID to select from manifest for GT loading")
    parser.add_argument("--case-id", default="case-0000")
    parser.add_argument("--budget", type=int, default=0)
    args = parser.parse_args()

    out_dir = args.out
    if not out_dir:
        out_dir = str(Path(args.run_json).resolve().parent)

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # If GT is provided, inject it into the run payload via a reserved key to keep the
    # unified_eval() signature stable (repo skeleton style).
    if args.gt or args.manifest:
        run_path = Path(args.run_json)
        run = _load_json(run_path)

        gt_boxes: Dict[int, List[Box]] = {}
        if args.gt:
            gt_boxes = _load_gt_boxes_by_sent_from_json(Path(args.gt))
        elif args.manifest:
            cid = str(args.case_id_manifest).strip() or str(run.get("case_id", "")).strip()
            if not cid:
                meta = run.get("meta", {})
                inp = meta.get("input", {}) if isinstance(meta, dict) else {}
                cid = str(inp.get("case_id", "")).strip()
            if not cid:
                raise SystemExit("[ERR] --case-id-manifest is required when case_id cannot be inferred from run.json")
            gt_boxes = _load_gt_boxes_by_sent_from_manifest(Path(args.manifest), cid)

        run["_gt_boxes_by_sent_mm"] = {str(k): [list(b) for b in v] for k, v in gt_boxes.items()}
        # Write a temporary in-memory file by reusing the same out_dir and filename.
        tmp_path = Path(out_dir) / "_tmp_run_with_gt.json"
        tmp_path.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
        result = unified_eval(str(tmp_path), out_dir, case_id=args.case_id, budget_B=args.budget)
    else:
        result = unified_eval(args.run_json, out_dir, case_id=args.case_id, budget_B=args.budget)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
