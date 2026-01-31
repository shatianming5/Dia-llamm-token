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
            raise TypeError("manifest rows must be JSON objects")
        rows.append(obj)
    return rows


def validate_ct_rate_ts_manifest(rows: List[Dict[str, Any]], *, require_n_ge: int | None = None) -> List[str]:
    errors: List[str] = []

    if require_n_ge is not None:
        thr = int(require_n_ge)
        if len(rows) < thr:
            errors.append(f"n({len(rows)}) is not >= {thr}")

    for i, row in enumerate(rows):
        vp = str(row.get("volume_path", "")).strip()
        if not vp:
            errors.append(f"row[{i}] missing volume_path")
        elif not Path(vp).exists():
            errors.append(f"row[{i}] volume_path not found: {vp}")

        mp = str(row.get("gt_mask_path", "")).strip()
        if not mp:
            errors.append(f"row[{i}] missing gt_mask_path")
        elif not Path(mp).exists():
            errors.append(f"row[{i}] gt_mask_path not found: {mp}")

        cp = str(row.get("totalseg_candidates_path", "")).strip()
        if cp and not Path(cp).exists():
            errors.append(f"row[{i}] totalseg_candidates_path not found: {cp}")

        # Optional paper-track provenance fields (validator can gate them via CLI flags).

        gb = row.get("grounding_boxes_by_sent_mm", {})
        if not isinstance(gb, dict):
            errors.append(f"row[{i}] grounding_boxes_by_sent_mm is not a dict")
            continue
        b0 = gb.get("0", None)
        if not isinstance(b0, list) or not b0:
            errors.append(f"row[{i}] grounding_boxes_by_sent_mm['0'] is empty")
            continue
        for j, b in enumerate(b0):
            if not isinstance(b, (list, tuple)) or len(b) != 6:
                errors.append(f"row[{i}] box[{j}] is not length-6")
                continue
            try:
                x0, x1, y0, y1, z0, z1 = [float(x) for x in b]
            except Exception:
                errors.append(f"row[{i}] box[{j}] has non-numeric entries")
                continue
            if not (x1 > x0 and y1 > y0 and z1 > z0):
                errors.append(f"row[{i}] box[{j}] has non-positive size")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate CT-RATE TS GT-box manifest invariants.")
    parser.add_argument("--in", dest="manifest_jsonl", required=True, help="Path to manifest.jsonl")
    parser.add_argument("--require-n-ge", type=int, default=None)
    parser.add_argument("--require-gt-is-pseudo", action="store_true", help="Require gt_is_pseudo=true on every row.")
    parser.add_argument("--require-coord-system", default=None, help="Require row.coord_system to equal this value.")
    args = parser.parse_args()

    rows = _load_jsonl(Path(args.manifest_jsonl))
    errors = validate_ct_rate_ts_manifest(rows, require_n_ge=args.require_n_ge)
    if args.require_gt_is_pseudo:
        for i, row in enumerate(rows):
            if row.get("gt_is_pseudo", None) is not True:
                errors.append(f"row[{i}] gt_is_pseudo must be true")
    if args.require_coord_system is not None:
        want = str(args.require_coord_system).strip()
        for i, row in enumerate(rows):
            if str(row.get("coord_system", "")).strip() != want:
                errors.append(f"row[{i}] coord_system must be '{want}'")
    if errors:
        for e in errors:
            print(f"[ERR] {e}")
        sys.exit(1)
    print("[OK] ct_rate_ts_manifest validated")


if __name__ == "__main__":
    main()
