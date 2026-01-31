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


def validate_manifest(
    rows: List[Dict[str, Any]],
    *,
    require_n_ge: int | None = None,
    valid_splits: List[str] | None = None,
    require_report_path_exists: bool = False,
    require_volume_path_exists: bool = False,
    require_nonempty_volume_paths_ge: int | None = None,
    require_case_id_prefix: str | None = None,
    require_labels_pos: bool = False,
    require_nonempty_labels_pos_ge: int | None = None,
) -> List[str]:
    errors: List[str] = []

    if require_n_ge is not None:
        thr = int(require_n_ge)
        if len(rows) < thr:
            errors.append(f"n({len(rows)}) is not >= {thr}")

    allowed = None
    if valid_splits:
        allowed = {str(x) for x in valid_splits if str(x)}
        if not allowed:
            allowed = None

    seen_case_ids: set[str] = set()
    nonempty_volume_paths = 0
    nonempty_labels_pos = 0
    for i, row in enumerate(rows):
        case_id = str(row.get("case_id", "")).strip()
        if not case_id:
            errors.append(f"row[{i}] missing case_id")
        elif case_id in seen_case_ids:
            errors.append(f"row[{i}] duplicate case_id: {case_id}")
        else:
            seen_case_ids.add(case_id)
            if require_case_id_prefix is not None:
                prefix = str(require_case_id_prefix)
                if prefix and not case_id.startswith(prefix):
                    errors.append(f"row[{i}] case_id '{case_id}' does not start with prefix '{prefix}'")

        if allowed is not None:
            if "split" not in row:
                errors.append(f"row[{i}] missing split")
            else:
                split = str(row.get("split", "")).strip()
                if split not in allowed:
                    errors.append(f"row[{i}] split '{split}' not in {sorted(allowed)}")

        if require_report_path_exists:
            rp = str(row.get("report_path", "")).strip()
            if not rp:
                errors.append(f"row[{i}] missing report_path")
            else:
                if not Path(rp).exists():
                    errors.append(f"row[{i}] report_path not found: {rp}")

        vp = str(row.get("volume_path", "")).strip()
        if vp:
            nonempty_volume_paths += 1
            if require_volume_path_exists and not Path(vp).exists():
                errors.append(f"row[{i}] volume_path not found: {vp}")

        if require_labels_pos:
            if "labels_pos" not in row:
                errors.append(f"row[{i}] missing labels_pos")
            else:
                lp = row.get("labels_pos", [])
                if not isinstance(lp, list):
                    errors.append(f"row[{i}] labels_pos is not a list")
                else:
                    if len([x for x in lp if str(x).strip()]) > 0:
                        nonempty_labels_pos += 1

    if require_nonempty_volume_paths_ge is not None:
        thr = int(require_nonempty_volume_paths_ge)
        if nonempty_volume_paths < thr:
            errors.append(f"nonempty_volume_paths({nonempty_volume_paths}) is not >= {thr}")

    if require_nonempty_labels_pos_ge is not None:
        thr = int(require_nonempty_labels_pos_ge)
        if nonempty_labels_pos < thr:
            errors.append(f"nonempty_labels_pos({nonempty_labels_pos}) is not >= {thr}")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate JSONL manifest invariants for the repo skeleton.")
    parser.add_argument("--in", dest="manifest_jsonl", required=True, help="Path to manifest.jsonl")
    parser.add_argument("--require-n-ge", type=int, default=None)
    parser.add_argument("--valid-splits", nargs="+", default=None, help="Allowed split labels, e.g. train val test")
    parser.add_argument("--require-report-path-exists", action="store_true")
    parser.add_argument("--require-volume-path-exists", action="store_true")
    parser.add_argument("--require-nonempty-volume-paths-ge", type=int, default=None)
    parser.add_argument("--require-case-id-prefix", default=None)
    parser.add_argument("--require-labels-pos", action="store_true")
    parser.add_argument("--require-nonempty-labels-pos-ge", type=int, default=None)
    args = parser.parse_args()

    rows = _load_jsonl(Path(args.manifest_jsonl))
    errors = validate_manifest(
        rows,
        require_n_ge=args.require_n_ge,
        valid_splits=args.valid_splits,
        require_report_path_exists=bool(args.require_report_path_exists),
        require_volume_path_exists=bool(args.require_volume_path_exists),
        require_nonempty_volume_paths_ge=args.require_nonempty_volume_paths_ge,
        require_case_id_prefix=args.require_case_id_prefix,
        require_labels_pos=bool(args.require_labels_pos),
        require_nonempty_labels_pos_ge=args.require_nonempty_labels_pos_ge,
    )
    if errors:
        for e in errors:
            print(f"[ERR] {e}")
        sys.exit(1)
    print("[OK] manifest validated")


if __name__ == "__main__":
    main()
