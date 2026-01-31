from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def validate_policy_dataset(rows: List[Dict[str, Any]], *, require_rows_ge: int | None, require_cases_ge: int | None) -> List[str]:
    errors: List[str] = []
    if require_rows_ge is not None and len(rows) < int(require_rows_ge):
        errors.append(f"n_rows({len(rows)}) is not >= {int(require_rows_ge)}")

    cases = {str(r.get('case_id', '')).strip() for r in rows if str(r.get('case_id', '')).strip()}
    if require_cases_ge is not None and len(cases) < int(require_cases_ge):
        errors.append(f"n_cases({len(cases)}) is not >= {int(require_cases_ge)}")

    required_keys = ["case_id", "token_id", "recon_error", "evidence_entropy", "citation_pressure", "history_splits", "reward"]
    for i, r in enumerate(rows[: min(50, len(rows))]):
        for k in required_keys:
            if k not in r:
                errors.append(f"row[{i}] missing key: {k}")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate policy dataset.jsonl invariants.")
    parser.add_argument("--in", dest="dataset_jsonl", required=True, help="Path to dataset.jsonl")
    parser.add_argument("--require-rows-ge", type=int, default=None)
    parser.add_argument("--require-cases-ge", type=int, default=None)
    args = parser.parse_args()

    p = Path(args.dataset_jsonl)
    if not p.exists():
        print(f"[ERR] missing dataset: {p}", file=sys.stderr)
        sys.exit(2)

    rows = _load_jsonl(p)
    errors = validate_policy_dataset(rows, require_rows_ge=args.require_rows_ge, require_cases_ge=args.require_cases_ge)
    if errors:
        for e in errors:
            print(f"[ERR] {e}")
        sys.exit(1)
    print("[OK] policy dataset validated")


if __name__ == "__main__":
    main()

