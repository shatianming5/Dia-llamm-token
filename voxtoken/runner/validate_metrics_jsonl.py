from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


REQUIRED_KEYS = [
    "case_id",
    "budget_B",
    "tokens_used",
    "latency_ms",
    "slot_f1",
    "unsupported_rate",
    "verifier_score",
]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise TypeError("metrics.jsonl rows must be JSON objects")
        rows.append(obj)
    return rows


def validate_metrics_jsonl(
    rows: List[Dict[str, Any]],
    *,
    require_n_ge: int | None = None,
    require_unique_case_ids: bool = True,
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
        elif require_unique_case_ids:
            if cid in seen:
                errors.append(f"row[{i}] duplicate case_id: {cid}")
            else:
                seen.add(cid)

        # latency_ms.total should be present and parseable.
        lat = row.get("latency_ms", {})
        if not isinstance(lat, dict) or "total" not in lat:
            errors.append(f"row[{i}] missing latency_ms.total")
        else:
            try:
                float(lat.get("total", 0.0))
            except Exception:
                errors.append(f"row[{i}] latency_ms.total not parseable as float")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate aggregated metrics.jsonl invariants.")
    parser.add_argument("--in", dest="metrics_jsonl", required=True, help="Path to metrics.jsonl")
    parser.add_argument("--require-n-ge", type=int, default=None)
    parser.add_argument("--allow-duplicate-case-ids", action="store_true")
    args = parser.parse_args()

    rows = _load_jsonl(Path(args.metrics_jsonl))
    errors = validate_metrics_jsonl(
        rows,
        require_n_ge=args.require_n_ge,
        require_unique_case_ids=not bool(args.allow_duplicate_case_ids),
    )
    if errors:
        for e in errors:
            print(f"[ERR] {e}")
        sys.exit(1)
    print("[OK] metrics.jsonl validated")


if __name__ == "__main__":
    main()

