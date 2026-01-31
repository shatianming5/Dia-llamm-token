from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


REQUIRED_KEYS = [
    "case_id",
    "n_gold_pos",
    "n_pred_pos",
    "precision",
    "recall",
    "f1",
]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise TypeError("label_metrics.jsonl rows must be JSON objects")
        rows.append(obj)
    return rows


def validate_label_metrics_jsonl(
    rows: List[Dict[str, Any]],
    *,
    require_n_ge: int | None = None,
    require_f1_ge: float | None = None,
    require_f1_lt: float | None = None,
) -> List[str]:
    errors: List[str] = []

    if require_n_ge is not None:
        thr = int(require_n_ge)
        if len(rows) < thr:
            errors.append(f"n({len(rows)}) is not >= {thr}")

    seen: set[str] = set()
    for i, row in enumerate(rows):
        for k in REQUIRED_KEYS:
            if k not in row:
                errors.append(f"row[{i}] missing required key: {k}")
                break

        cid = str(row.get("case_id", "")).strip()
        if not cid:
            errors.append(f"row[{i}] empty case_id")
        elif cid in seen:
            errors.append(f"row[{i}] duplicate case_id: {cid}")
        else:
            seen.add(cid)

        try:
            f1 = float(row.get("f1", 0.0))
        except Exception:
            errors.append(f"row[{i}] f1 not parseable as float")
        else:
            if not (0.0 <= f1 <= 1.0):
                errors.append(f"row[{i}] f1({f1}) not in [0,1]")
            if require_f1_ge is not None:
                thr = float(require_f1_ge)
                if not (f1 >= thr):
                    errors.append(f"row[{i}] f1({f1}) is not >= {thr}")
            if require_f1_lt is not None:
                thr = float(require_f1_lt)
                if not (f1 < thr):
                    errors.append(f"row[{i}] f1({f1}) is not < {thr}")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate label_metrics.jsonl invariants.")
    parser.add_argument("--in", dest="label_metrics_jsonl", required=True, help="Path to label_metrics.jsonl")
    parser.add_argument("--require-n-ge", type=int, default=None)
    parser.add_argument("--require-f1-ge", type=float, default=None)
    parser.add_argument("--require-f1-lt", type=float, default=None)
    args = parser.parse_args()

    rows = _load_jsonl(Path(args.label_metrics_jsonl))
    errors = validate_label_metrics_jsonl(
        rows,
        require_n_ge=args.require_n_ge,
        require_f1_ge=args.require_f1_ge,
        require_f1_lt=args.require_f1_lt,
    )
    if errors:
        for e in errors:
            print(f"[ERR] {e}")
        sys.exit(1)
    print("[OK] label_metrics.jsonl validated")


if __name__ == "__main__":
    main()
