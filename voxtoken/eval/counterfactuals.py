from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..verify.extract_slots import extract_slots_from_report


def run_counterfactuals(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Causal tests: shuffle citations, swap tokens, etc."""
    run_json = cfg.get("run_json") or cfg.get("path")
    if not run_json:
        return {"error": "missing run_json"}

    run = json.loads(Path(str(run_json)).read_text(encoding="utf-8"))
    report = str(run.get("report", ""))
    citations = list(run.get("citations", []))
    plan = run.get("plan", {})
    if not isinstance(plan, dict):
        plan = {}

    sentences = [s.strip() for s in report.splitlines() if s.strip()]

    Box = Tuple[float, float, float, float, float, float]

    def _parse_boxes_by_sent(raw: Any) -> Dict[int, List[Box]]:
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
            return _parse_boxes_by_sent(raw)
        raise KeyError(f"case_id not found in manifest: {want}")

    def _load_gt_boxes_by_sent_from_json(path: Path) -> Dict[int, List[Box]]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw = (
            payload.get("grounding_boxes_by_sent_mm", None)
            or payload.get("gt_boxes_by_sent_mm", None)
            or payload.get("boxes_by_sent_mm", None)
            or payload.get("boxes_by_sent", None)
            or payload
        )
        return _parse_boxes_by_sent(raw)

    def box_iou_3d(a: Box, b: Box) -> float:
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

    def token_box_by_id(payload: Dict[str, Any]) -> Dict[int, Box]:
        out: Dict[int, Box] = {}
        tokens = payload.get("tokens", [])
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

    def cited_ids_by_sent(cites: List[Dict[str, Any]]) -> Dict[int, List[int]]:
        out: Dict[int, List[int]] = {}
        for c in cites:
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

    def supported_ids_by_sent(facts: Any, n_sentences: int) -> Dict[int, List[int]]:
        out: Dict[int, List[int]] = {}
        if not isinstance(facts, list):
            return out
        for sid in range(min(int(n_sentences), len(facts))):
            f = facts[sid]
            if not isinstance(f, dict):
                continue
            raw = f.get("supported_token_ids", []) or []
            if not isinstance(raw, list):
                continue
            ids: List[int] = []
            for x in raw:
                try:
                    ids.append(int(x))
                except Exception:
                    continue
            out[int(sid)] = ids
        return out

    def unsupported_rate(
        n_sentences: int, cited_by_sent: Dict[int, List[int]], supported_by_sent: Dict[int, List[int]]
    ) -> float:
        if n_sentences <= 0:
            return 0.0
        unsupported = 0
        for sid in range(int(n_sentences)):
            cited = set(int(x) for x in (cited_by_sent.get(int(sid), []) or []))
            if not cited:
                unsupported += 1
                continue
            supported = set(int(x) for x in (supported_by_sent.get(int(sid), []) or []))
            if supported and not (cited & supported):
                unsupported += 1
        return float(unsupported) / float(n_sentences)

    def grounding_metrics(
        n_sentences: int,
        cited_by_sent: Dict[int, List[int]],
        cited_boxes_by_id: Dict[int, Box],
        gt_boxes_by_sent: Dict[int, List[Box]],
    ) -> Dict[str, float]:
        eligible = 0
        hits0 = 0
        hits1 = 0
        ious: List[float] = []
        for sid in range(int(n_sentences)):
            gt_boxes = list(gt_boxes_by_sent.get(int(sid), []) or [])
            if not gt_boxes:
                continue
            eligible += 1

            cited_boxes: List[Box] = []
            for tid in cited_by_sent.get(int(sid), []) or []:
                if int(tid) in cited_boxes_by_id:
                    cited_boxes.append(cited_boxes_by_id[int(tid)])

            max_iou = 0.0
            for cb in cited_boxes:
                for gb in gt_boxes:
                    max_iou = max(max_iou, float(box_iou_3d(cb, gb)))

            ious.append(float(max_iou))
            if max_iou > 0.0:
                hits0 += 1
            if max_iou >= 0.1:
                hits1 += 1
        if eligible <= 0:
            return {"ground_hit@0.0": 0.0, "ground_hit@0.1": 0.0, "ground_mean_iou": 0.0, "eligible": 0.0}
        mean_iou = float(sum(ious) / float(len(ious))) if ious else 0.0
        return {
            "ground_hit@0.0": float(hits0) / float(eligible),
            "ground_hit@0.1": float(hits1) / float(eligible),
            "ground_mean_iou": float(mean_iou),
            "eligible": float(eligible),
        }

    def mask_sanity(supported_boxes_all: List[Box], new_token_ids: Set[int], boxes_by_id: Dict[int, Box]) -> Dict[str, Any]:
        ious: List[float] = []
        for tid in sorted(int(x) for x in new_token_ids):
            if int(tid) not in boxes_by_id:
                continue
            tb = boxes_by_id[int(tid)]
            best = 0.0
            for sb in supported_boxes_all:
                best = max(best, float(box_iou_3d(tb, sb)))
            ious.append(float(best))

        def q(vs: List[float], qv: float) -> float:
            if not vs:
                return 0.0
            srt = sorted(float(x) for x in vs)
            idx = int(round(float(qv) * float(len(srt) - 1)))
            idx = max(0, min(len(srt) - 1, idx))
            return float(srt[idx])

        return {
            "n_new_tokens": int(len(ious)),
            "p50_iou": q(ious, 0.50),
            "p75_iou": q(ious, 0.75),
            "p90_iou": q(ious, 0.90),
        }

    def multiset_f1(gold: List[Dict[str, Any]], pred: List[Dict[str, Any]]) -> float:
        from collections import Counter

        def slot_key(d: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
            return (
                str(d.get("finding_type", "")).strip(),
                str(d.get("side", "U")).strip(),
                str(d.get("location", "U")).strip(),
                str(d.get("size_bin", "U")).strip(),
                str(d.get("certainty", "U")).strip(),
            )

        if not gold and not pred:
            return 1.0
        if not gold or not pred:
            return 0.0

        g = Counter(slot_key(x) for x in gold)
        p = Counter(slot_key(x) for x in pred)
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

    n_sent = int(len(sentences))
    facts = plan.get("facts", []) if isinstance(plan, dict) else []
    supported_by_sent = supported_ids_by_sent(facts, n_sentences=n_sent)

    boxes_base = token_box_by_id(run)
    cited_base = cited_ids_by_sent([c for c in citations if isinstance(c, dict)])

    # Optional GT sentence boxes (RadGenome-style). Prefer explicit cfg, then manifest/gt.json,
    # and finally fall back to a repo-skeleton proxy (plan-supported token boxes).
    gt_boxes_by_sent: Optional[Dict[int, List[Box]]] = None
    if isinstance(cfg.get("gt_boxes_by_sent_mm"), dict):
        gt_boxes_by_sent = _parse_boxes_by_sent(cfg.get("gt_boxes_by_sent_mm"))
    elif cfg.get("gt_json") or cfg.get("gt"):
        gt_path = Path(str(cfg.get("gt_json") or cfg.get("gt"))).expanduser()
        if gt_path.exists():
            gt_boxes_by_sent = _load_gt_boxes_by_sent_from_json(gt_path)
    elif cfg.get("manifest") or cfg.get("manifest_jsonl"):
        mp = Path(str(cfg.get("manifest") or cfg.get("manifest_jsonl"))).expanduser()
        if mp.exists():
            case_id = (
                str(cfg.get("case_id") or cfg.get("case") or cfg.get("case_id_manifest") or "").strip()
                or str(run.get("case_id", "")).strip()
                or str(((run.get("meta", {}) or {}).get("input", {}) or {}).get("case_id", "")).strip()
            )
            if case_id:
                gt_boxes_by_sent = _load_gt_boxes_by_sent_from_manifest(mp, case_id)

    if not gt_boxes_by_sent:
        proxy: Dict[int, List[Box]] = {}
        for sid, ids in supported_by_sent.items():
            boxes = [boxes_base[int(tid)] for tid in ids if int(tid) in boxes_base]
            if boxes:
                proxy[int(sid)] = boxes
        gt_boxes_by_sent = proxy

    gt_all_boxes: List[Box] = []
    for v in (gt_boxes_by_sent or {}).values():
        if isinstance(v, list):
            gt_all_boxes.extend([b for b in v if isinstance(b, tuple) and len(b) == 6])

    # Slot-F1 stays constant across citation/omega perturbations (report text unchanged).
    gold_facts = [f for f in facts if isinstance(f, dict)]
    pred_facts = [x.__dict__ for x in extract_slots_from_report(report)]
    slot_f1_micro = float(multiset_f1(gold_facts, pred_facts))

    def permute_omega(boxes_by_id: Dict[int, Box], *, seed: int) -> Dict[int, Box]:
        ids = sorted(int(k) for k in boxes_by_id.keys())
        boxes = [boxes_by_id[i] for i in ids]
        rng = random.Random(int(seed))
        rng.shuffle(boxes)
        return {int(i): boxes[j] for j, i in enumerate(ids)}

    def swap_citations(cited_by_sent: Dict[int, List[int]], *, seed: int) -> Dict[int, List[int]]:
        rng = random.Random(int(seed))
        sids = sorted(int(s) for s in cited_by_sent.keys() if int(s) >= 0)
        pools = [list(cited_by_sent.get(s, []) or []) for s in sids]
        rng.shuffle(pools)
        return {int(sid): list(pools[i]) for i, sid in enumerate(sids)}

    def remove_citations(cited_by_sent: Dict[int, List[int]]) -> Dict[int, List[int]]:
        return {int(k): [] for k in cited_by_sent.keys()}

    def collect_added_token_ids(payload: Dict[str, Any]) -> Set[int]:
        out: Set[int] = set()
        trace = payload.get("trace", [])
        if not isinstance(trace, list):
            return out
        for step in trace:
            if not isinstance(step, dict):
                continue
            raw = step.get("added_token_ids", []) or []
            if not isinstance(raw, list):
                continue
            for x in raw:
                try:
                    out.add(int(x))
                except Exception:
                    continue
        return out

    def supported_boxes_all(supported_map: Dict[int, List[int]], boxes_by_id: Dict[int, Box]) -> List[Box]:
        out: List[Box] = []
        seen: Set[int] = set()
        for ids in supported_map.values():
            for tid in ids:
                tid = int(tid)
                if tid in seen:
                    continue
                seen.add(tid)
                if tid in boxes_by_id:
                    out.append(boxes_by_id[tid])
        return out

    # Scenarios.
    cited_swap = swap_citations(cited_base, seed=0)
    cited_removed = remove_citations(cited_base)
    boxes_perm = permute_omega(boxes_base, seed=0)

    base = float(unsupported_rate(n_sent, cited_base, supported_by_sent))
    swap_rate = float(unsupported_rate(n_sent, cited_swap, supported_by_sent))
    removed_rate = float(unsupported_rate(n_sent, cited_removed, supported_by_sent))

    def scenario_row(
        cf_type: str,
        cited: Dict[int, List[int]],
        cite_boxes: Dict[int, Box],
    ) -> Dict[str, Any]:
        u = float(unsupported_rate(n_sent, cited, supported_by_sent))
        gm = grounding_metrics(n_sent, cited, cite_boxes, gt_boxes_by_sent or {})
        gh0 = float(gm.get("ground_hit@0.0", 0.0))
        gh1 = float(gm.get("ground_hit@0.1", 0.0))
        gmiou = float(gm.get("ground_mean_iou", 0.0))

        added_ids = collect_added_token_ids(run)
        ms = mask_sanity(gt_all_boxes, added_ids, cite_boxes)

        return {
            "cf_type": str(cf_type),
            "slot_f1_micro": float(slot_f1_micro),
            "ground_hit@0.0": float(gh0),
            "ground_hit@0.1": float(gh1),
            "ground_mean_iou": float(gmiou),
            "unsupported_sent_pct": 100.0 * float(u),
            "mask_sanity": ms,
        }

    rows = [
        scenario_row("base", cited_base, boxes_base),
        scenario_row("swap_citations", cited_swap, boxes_base),
        scenario_row("remove_citations", cited_removed, boxes_base),
        scenario_row("permute_omega", cited_base, boxes_perm),
    ]

    return {
        "n_sentences": int(n_sent),
        "rows": rows,
        "unsupported_rate": {
            "base": base,
            "citation_swap": swap_rate,
            "remove_citations": removed_rate,
        },
        "ground_hit@0.0": {
            "base": float(rows[0]["ground_hit@0.0"]),
            "swap_citations": float(rows[1]["ground_hit@0.0"]),
            "remove_citations": float(rows[2]["ground_hit@0.0"]),
            "permute_omega": float(rows[3]["ground_hit@0.0"]),
        },
        "ground_hit@0.1": {
            "base": float(rows[0]["ground_hit@0.1"]),
            "swap_citations": float(rows[1]["ground_hit@0.1"]),
            "remove_citations": float(rows[2]["ground_hit@0.1"]),
            "permute_omega": float(rows[3]["ground_hit@0.1"]),
        },
        "ground_mean_iou": {
            "base": float(rows[0].get("ground_mean_iou", 0.0)),
            "swap_citations": float(rows[1].get("ground_mean_iou", 0.0)),
            "remove_citations": float(rows[2].get("ground_mean_iou", 0.0)),
            "permute_omega": float(rows[3].get("ground_mean_iou", 0.0)),
        },
        "mask_sanity": {
            "base": rows[0]["mask_sanity"],
            "swap_citations": rows[1]["mask_sanity"],
            "remove_citations": rows[2]["mask_sanity"],
            "permute_omega": rows[3]["mask_sanity"],
        },
    }
