from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_label_metrics(
    metrics: Dict[str, Any],
    *,
    require_n_gold_pos_gt: int | None = None,
    require_f1_ge: float | None = None,
    require_f1_lt: float | None = None,
) -> List[str]:
    errors: List[str] = []

    for k in ["case_id", "n_gold_pos", "n_pred_pos", "precision", "recall", "f1"]:
        if k not in metrics:
            errors.append(f"missing required key: {k}")
            return errors

    try:
        f1 = float(metrics.get("f1", 0.0))
    except Exception:
        errors.append("f1 is not parseable as float")
        return errors
    if not (0.0 <= f1 <= 1.0):
        errors.append(f"f1({f1}) not in [0,1]")
    if require_f1_ge is not None:
        thr = float(require_f1_ge)
        if not (f1 >= thr):
            errors.append(f"f1({f1}) is not >= {thr}")
    if require_f1_lt is not None:
        thr = float(require_f1_lt)
        if not (f1 < thr):
            errors.append(f"f1({f1}) is not < {thr}")

    if require_n_gold_pos_gt is not None:
        thr = int(require_n_gold_pos_gt)
        try:
            n = int(metrics.get("n_gold_pos", 0))
        except Exception:
            errors.append("n_gold_pos is not parseable as int")
        else:
            if not (n > thr):
                errors.append(f"n_gold_pos({n}) is not > {thr}")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate label_metrics.json invariants.")
    parser.add_argument("--in", dest="label_metrics_json", required=True, help="Path to label_metrics.json")
    parser.add_argument("--require-n-gold-pos-gt", type=int, default=None)
    parser.add_argument("--require-f1-ge", type=float, default=None)
    parser.add_argument("--require-f1-lt", type=float, default=None)
    args = parser.parse_args()

    metrics = _load_json(Path(args.label_metrics_json))
    errors = validate_label_metrics(
        metrics,
        require_n_gold_pos_gt=args.require_n_gold_pos_gt,
        require_f1_ge=args.require_f1_ge,
        require_f1_lt=args.require_f1_lt,
    )
    if errors:
        for e in errors:
            print(f"[ERR] {e}")
        sys.exit(1)
    print("[OK] label_metrics.json validated")


if __name__ == "__main__":
    main()
