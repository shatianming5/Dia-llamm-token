from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from ..verify.extract_slots import extract_slots_from_report


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_manifest_row(manifest_jsonl: Path, case_id: str) -> Dict[str, Any]:
    for line in manifest_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            continue
        if str(obj.get("case_id", "")).strip() == str(case_id).strip():
            return obj
    raise KeyError(f"case_id not found in manifest: {case_id}")


def _norm_label(s: str) -> str:
    # Normalize both report finding types and label names into a comparable form.
    x = str(s or "").strip().lower()
    x = x.replace("_", " ")
    x = re.sub(r"[^a-z0-9 ]+", " ", x)
    x = re.sub(r"\\s+", " ", x).strip()
    return x


def compute_ct_rate_label_metrics(
    run: Dict[str, Any],
    manifest_row: Dict[str, Any],
) -> Dict[str, Any]:
    case_id = str(run.get("case_id", "")).strip() or str(manifest_row.get("case_id", "")).strip() or "case-0000"

    labels_pos_raw = manifest_row.get("labels_pos", [])
    if not isinstance(labels_pos_raw, list):
        labels_pos_raw = []
    gold = {_norm_label(str(x)) for x in labels_pos_raw if str(x).strip()}

    report = str(run.get("report", ""))
    slots = extract_slots_from_report(report)
    pred_raw = {_norm_label(str(s.finding_type)) for s in slots if str(getattr(s, "finding_type", "")).strip()}

    # Only count predictions that match some known label string in the gold universe.
    # (This keeps the metric stable even if reports contain auxiliary tokens.)
    pred = {p for p in pred_raw if p in gold} if gold else set()

    match = set(gold) & set(pred)
    n_gold = len(gold)
    n_pred = len(pred)
    n_match = len(match)

    if n_pred <= 0:
        precision = 0.0 if n_gold > 0 else 1.0
    else:
        precision = float(n_match) / float(n_pred)

    if n_gold <= 0:
        recall = 0.0 if n_pred > 0 else 1.0
    else:
        recall = float(n_match) / float(n_gold)

    if precision + recall <= 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    return {
        "case_id": case_id,
        "n_gold_pos": int(n_gold),
        "n_pred_pos": int(n_pred),
        "n_match": int(n_match),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "labels_gold_pos": sorted(gold),
        "labels_pred_pos": sorted(pred),
        "labels_match": sorted(match),
    }


def ct_rate_label_eval(
    *,
    run_json: str,
    manifest_jsonl: str,
    out_dir: str,
    case_id: str | None = None,
) -> Dict[str, Any]:
    run_path = Path(run_json)
    manifest_path = Path(manifest_jsonl)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    run = _load_json(run_path)
    cid = (
        str(case_id).strip()
        if case_id
        else str(run.get("case_id", "")).strip()
        or str(((run.get("meta", {}) or {}).get("input", {}) or {}).get("case_id", "")).strip()
    )
    if not cid:
        raise ValueError("case_id is required (not present in run.json and not provided via --case-id)")

    row = _load_manifest_row(manifest_path, cid)
    metrics = compute_ct_rate_label_metrics(run, row)

    (out_path / "label_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_path / "label_metrics.jsonl").write_text(
        json.dumps(metrics, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        "timestamp_utc": _utc_now_iso(),
        "run_json": str(run_path),
        "manifest_jsonl": str(manifest_path),
        "case_id": str(cid),
        "out_dir": str(out_path),
    }
    (out_path / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return {"out_dir": str(out_path), "label_metrics_path": str(out_path / "label_metrics.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description="CT-RATE multi-label evaluation from predicted labels.")
    parser.add_argument("--run", dest="run_json", required=True, help="Path to run.json")
    parser.add_argument("--manifest", dest="manifest_jsonl", required=True, help="Path to manifest.jsonl (must contain labels_pos)")
    parser.add_argument("--out", required=True, help="Output directory (writes label_metrics.json(l))")
    parser.add_argument("--case-id", default=None, help="Case ID (defaults to run.json case_id if present)")
    args = parser.parse_args()

    result = ct_rate_label_eval(
        run_json=str(args.run_json),
        manifest_jsonl=str(args.manifest_jsonl),
        out_dir=str(args.out),
        case_id=(str(args.case_id) if args.case_id else None),
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

