from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_metrics(
    metrics: Dict[str, Any],
    *,
    require_unsupported_gt: float | None = None,
    require_slot_f1_ge: float | None = None,
    require_slot_f1_lt: float | None = None,
    require_latency_total_gt: float | None = None,
    require_ground_hit_0_ge: float | None = None,
    require_ground_hit_01_ge: float | None = None,
    require_ground_mean_iou_ge: float | None = None,
) -> List[str]:
    errors: List[str] = []
    if "unsupported_rate" not in metrics:
        errors.append("missing required key: unsupported_rate")
        return errors

    try:
        unsupported_rate = float(metrics.get("unsupported_rate", 0.0))
    except Exception:
        errors.append("unsupported_rate is not parseable as float")
        return errors

    if require_unsupported_gt is not None:
        thr = float(require_unsupported_gt)
        if not (unsupported_rate > thr):
            errors.append(f"unsupported_rate({unsupported_rate}) is not > {thr}")

    if require_slot_f1_ge is not None or require_slot_f1_lt is not None:
        if "slot_f1" not in metrics:
            errors.append("missing required key: slot_f1")
        else:
            try:
                slot_f1 = float(metrics.get("slot_f1", 0.0))
            except Exception:
                errors.append("slot_f1 is not parseable as float")
            else:
                if require_slot_f1_ge is not None:
                    thr = float(require_slot_f1_ge)
                    if not (slot_f1 >= thr):
                        errors.append(f"slot_f1({slot_f1}) is not >= {thr}")
                if require_slot_f1_lt is not None:
                    thr = float(require_slot_f1_lt)
                    if not (slot_f1 < thr):
                        errors.append(f"slot_f1({slot_f1}) is not < {thr}")

    if require_latency_total_gt is not None:
        latency = metrics.get("latency_ms", {})
        total = None
        if isinstance(latency, dict) and "total" in latency:
            try:
                total = float(latency.get("total", 0.0))
            except Exception:
                total = None
        if total is None:
            errors.append("latency_ms.total is missing or not parseable as float")
        else:
            thr = float(require_latency_total_gt)
            if not (total > thr):
                errors.append(f"latency_ms.total({total}) is not > {thr}")

    def _check_float(key: str, *, ge: float | None = None) -> None:
        if ge is None:
            return
        if key not in metrics:
            errors.append(f"missing required key: {key}")
            return
        try:
            v = float(metrics.get(key, 0.0))
        except Exception:
            errors.append(f"{key} is not parseable as float")
            return
        thr = float(ge)
        if not (v >= thr):
            errors.append(f"{key}({v}) is not >= {thr}")

    _check_float("ground_hit@0.0", ge=require_ground_hit_0_ge)
    _check_float("ground_hit@0.1", ge=require_ground_hit_01_ge)
    _check_float("ground_mean_iou", ge=require_ground_mean_iou_ge)

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate metrics.json invariants for the repo skeleton.")
    parser.add_argument("--in", dest="metrics_json", required=True, help="Path to metrics.json")
    parser.add_argument("--require-unsupported-gt", type=float, default=None)
    parser.add_argument("--require-slot-f1-ge", type=float, default=None)
    parser.add_argument("--require-slot-f1-lt", type=float, default=None)
    parser.add_argument("--require-latency-total-gt", type=float, default=None)
    parser.add_argument("--require-ground-hit-0-ge", type=float, default=None)
    parser.add_argument("--require-ground-hit-01-ge", type=float, default=None)
    parser.add_argument("--require-ground-mean-iou-ge", type=float, default=None)
    args = parser.parse_args()

    metrics = _load_json(Path(args.metrics_json))
    errors = validate_metrics(
        metrics,
        require_unsupported_gt=args.require_unsupported_gt,
        require_slot_f1_ge=args.require_slot_f1_ge,
        require_slot_f1_lt=args.require_slot_f1_lt,
        require_latency_total_gt=args.require_latency_total_gt,
        require_ground_hit_0_ge=args.require_ground_hit_0_ge,
        require_ground_hit_01_ge=args.require_ground_hit_01_ge,
        require_ground_mean_iou_ge=args.require_ground_mean_iou_ge,
    )
    if errors:
        for e in errors:
            print(f"[ERR] {e}")
        sys.exit(1)
    print("[OK] metrics.json validated")


if __name__ == "__main__":
    main()
