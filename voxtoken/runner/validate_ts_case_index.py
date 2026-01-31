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
        if not isinstance(obj, dict):
            raise TypeError("rows must be JSON objects")
        rows.append(obj)
    return rows


def validate_ts_case_index(
    rows: List[Dict[str, Any]],
    *,
    require_n_ge: int | None,
    valid_splits: List[str] | None,
) -> List[str]:
    errors: List[str] = []

    if require_n_ge is not None and len(rows) < int(require_n_ge):
        errors.append(f"n({len(rows)}) is not >= {int(require_n_ge)}")

    allowed = {str(x) for x in (valid_splits or []) if str(x).strip()}
    if not allowed:
        allowed = {"train", "val", "test"}

    for i, r in enumerate(rows):
        cid = str(r.get("case_id", "")).strip()
        if not cid:
            errors.append(f"row[{i}] missing case_id")

        split = str(r.get("split", "")).strip()
        if split not in allowed:
            errors.append(f"row[{i}] split '{split}' not in {sorted(allowed)}")

        vp = str(r.get("volume_path", "")).strip()
        if not vp:
            errors.append(f"row[{i}] missing volume_path")
        elif not Path(vp).exists():
            errors.append(f"row[{i}] volume_path not found: {vp}")

        mp = str(r.get("gt_mask_path", "")).strip()
        if not mp:
            errors.append(f"row[{i}] missing gt_mask_path")
        elif not Path(mp).exists():
            errors.append(f"row[{i}] gt_mask_path not found: {mp}")

        if r.get("gt_is_pseudo", None) is not True:
            errors.append(f"row[{i}] gt_is_pseudo must be true")

        if not str(r.get("gt_source", "")).strip():
            errors.append(f"row[{i}] gt_source missing")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate TS case index JSONL invariants.")
    parser.add_argument("--in", dest="cases_jsonl", required=True, help="Path to cases.jsonl")
    parser.add_argument("--require-n-ge", type=int, default=None)
    parser.add_argument("--valid-splits", nargs="+", default=None)
    args = parser.parse_args()

    rows = _load_jsonl(Path(args.cases_jsonl))
    errors = validate_ts_case_index(rows, require_n_ge=args.require_n_ge, valid_splits=args.valid_splits)
    if errors:
        for e in errors:
            print(f"[ERR] {e}")
        sys.exit(1)
    print("[OK] ts case index validated")


if __name__ == "__main__":
    main()

