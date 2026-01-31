from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from .ct_rate_paths import resolve_ct_rate_volume_path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strip_nii_suffix(name: str) -> str:
    s = str(name or "").strip()
    if s.endswith(".nii.gz"):
        return s[: -len(".nii.gz")]
    if s.endswith(".nii"):
        return s[: -len(".nii")]
    return s


def _stable_int_hash(s: str) -> int:
    # Deterministic across processes (avoid Python's randomized hash()).
    h = 0
    for ch in str(s):
        h = (h * 131 + ord(ch)) % 2147483647
    return int(h)


def _assign_split(case_id: str) -> str:
    # Deterministic 60/20/20 split (dataset on this machine is small).
    r = _stable_int_hash(case_id) % 100
    if r < 60:
        return "train"
    if r < 80:
        return "val"
    return "test"


def _index_masks(root: Path) -> Dict[str, str]:
    idx: Dict[str, str] = {}
    for p in sorted(root.rglob("*.nii.gz")):
        name = p.name
        if name not in idx:
            idx[name] = str(p)
    return idx


def build_ts_nodule_case_index(
    *,
    ts_csv: Path,
    masks_root: Path,
    out_dir: Path,
    ct_rate_root: str,
) -> Dict[str, object]:
    if not ts_csv.exists():
        raise FileNotFoundError(f"ts_csv not found: {ts_csv}")
    if not masks_root.exists():
        raise FileNotFoundError(f"masks_root not found: {masks_root}")

    # Collect unique mask_file entries.
    mask_files: set[str] = set()
    with ts_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not isinstance(row, dict):
                continue
            mf = str(row.get("mask_file", "")).strip()
            if mf:
                mask_files.add(mf)

    mask_index = _index_masks(masks_root)

    out_rows: List[Dict[str, object]] = []
    missing_mask = 0
    missing_volume = 0
    split_counts: Counter[str] = Counter()

    for mf in sorted(mask_files):
        gt_mask_path = mask_index.get(mf, "")
        if not gt_mask_path or not Path(gt_mask_path).exists():
            missing_mask += 1
            continue

        volume_path = resolve_ct_rate_volume_path(mf, root=ct_rate_root) or ""
        if not volume_path or not Path(volume_path).exists():
            missing_volume += 1
            continue

        case_id = _strip_nii_suffix(mf)
        split = _assign_split(case_id)
        split_counts[split] += 1

        out_rows.append(
            {
                "case_id": str(case_id),
                "volume_name": str(mf),
                "volume_path": str(volume_path),
                "gt_mask_path": str(gt_mask_path),
                "split": str(split),
                "gt_source": "ts_seg.lung_nodules",
                "gt_is_pseudo": True,
            }
        )

    # Stable ordering.
    out_rows.sort(key=lambda r: str(r.get("case_id", "")))

    out_dir.mkdir(parents=True, exist_ok=True)
    cases_path = out_dir / "cases.jsonl"
    cases_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows), encoding="utf-8")

    summary = {
        "timestamp_utc": _utc_now_iso(),
        "ts_csv": str(ts_csv),
        "masks_root": str(masks_root),
        "ct_rate_root": str(ct_rate_root),
        "out_dir": str(out_dir),
        "cases_jsonl": str(cases_path),
        "n_unique_mask_files": int(len(mask_files)),
        "n_cases_written": int(len(out_rows)),
        "n_missing_mask": int(missing_mask),
        "n_missing_volume": int(missing_volume),
        "split_counts": dict(split_counts),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a per-volume case index from TS lung_nodules metadata (pseudo-GT).")
    parser.add_argument("--ts-csv", required=True, help="Path to TS lung_nodules valid_metadata.csv")
    parser.add_argument("--masks-root", required=True, help="Root dir containing per-volume masks (valid_fixed)")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--ct-rate-root", default="/data/CT-RATE/dataset", help="CT-RATE dataset root directory")
    args = parser.parse_args()

    try:
        summary = build_ts_nodule_case_index(
            ts_csv=Path(str(args.ts_csv)).expanduser(),
            masks_root=Path(str(args.masks_root)).expanduser(),
            out_dir=Path(str(args.out)).expanduser(),
            ct_rate_root=str(args.ct_rate_root),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[ERR] {exc}", file=sys.stderr)
        raise

    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
