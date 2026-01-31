from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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


def validate_split_counts(
    rows: List[Dict[str, Any]],
    *,
    split_key: str,
    require_train_ge: int | None,
    require_val_ge: int | None,
    require_test_ge: int | None,
) -> List[str]:
    errors: List[str] = []

    key = str(split_key or "split").strip() or "split"
    counts: Counter[str] = Counter()
    for r in rows:
        counts[str(r.get(key, "")).strip()] += 1

    def _check(name: str, thr: int | None) -> None:
        if thr is None:
            return
        if int(counts.get(name, 0)) < int(thr):
            errors.append(f"split '{name}' count({int(counts.get(name, 0))}) is not >= {int(thr)}")

    _check("train", require_train_ge)
    _check("val", require_val_ge)
    _check("test", require_test_ge)
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate that a JSONL manifest has non-empty train/val/test splits.")
    parser.add_argument("--in", dest="manifest_jsonl", required=True, help="Path to manifest.jsonl")
    parser.add_argument("--split-key", default="split", help="JSON field holding the split label (default: split)")
    parser.add_argument("--require-train-ge", type=int, default=None)
    parser.add_argument("--require-val-ge", type=int, default=None)
    parser.add_argument("--require-test-ge", type=int, default=None)
    args = parser.parse_args()

    p = Path(args.manifest_jsonl)
    if not p.exists():
        print(f"[ERR] missing manifest: {p}", file=sys.stderr)
        sys.exit(2)

    rows = _load_jsonl(p)
    errors = validate_split_counts(
        rows,
        split_key=str(args.split_key),
        require_train_ge=args.require_train_ge,
        require_val_ge=args.require_val_ge,
        require_test_ge=args.require_test_ge,
    )
    if errors:
        for e in errors:
            print(f"[ERR] {e}")
        sys.exit(1)
    print("[OK] split counts validated")


if __name__ == "__main__":
    main()

