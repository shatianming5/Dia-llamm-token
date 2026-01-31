from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .ct_rate_report_from_labels import ct_rate_report_from_labels


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_manifest_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            continue
        rows.append(obj)
    return rows


def batch_ct_rate_report_from_labels(
    *,
    manifest_jsonl: str,
    out_dir: str,
    max_cases: int = 0,
    require_nonempty_labels: bool = True,
) -> Dict[str, Any]:
    manifest_path = Path(manifest_jsonl)
    out_path = Path(out_dir)
    runs_dir = out_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_manifest_rows(manifest_path)

    selected_case_ids: List[str] = []
    for row in rows:
        cid = str(row.get("case_id", "")).strip()
        if not cid:
            continue
        labels_pos = row.get("labels_pos", [])
        has_labels = isinstance(labels_pos, list) and any(str(x).strip() for x in labels_pos)
        if require_nonempty_labels and not has_labels:
            continue
        selected_case_ids.append(cid)

    if max_cases and int(max_cases) > 0:
        selected_case_ids = selected_case_ids[: int(max_cases)]

    failures: List[Dict[str, Any]] = []
    n_ok = 0
    for cid in selected_case_ids:
        try:
            ct_rate_report_from_labels(
                manifest_jsonl=str(manifest_path),
                out_dir=str(runs_dir / cid),
                case_id=str(cid),
                max_labels=None,
            )
            n_ok += 1
        except Exception as exc:  # noqa: BLE001
            failures.append({"case_id": str(cid), "error": str(exc)})

    summary = {
        "timestamp_utc": _utc_now_iso(),
        "manifest_jsonl": str(manifest_path),
        "out_dir": str(out_path),
        "runs_dir": str(runs_dir),
        "require_nonempty_labels": bool(require_nonempty_labels),
        "max_cases": int(max_cases),
        "n_selected": int(len(selected_case_ids)),
        "n_ok": int(n_ok),
        "n_failed": int(len(failures)),
        "failures": failures,
    }
    (out_path / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if failures:
        raise RuntimeError(f"{len(failures)} case(s) failed: {failures[:3]}")

    return {"out_dir": str(out_path), "runs_dir": str(runs_dir), "n": int(n_ok)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-generate CT-RATE-style reports from manifest labels_pos.")
    parser.add_argument("--manifest", dest="manifest_jsonl", required=True, help="Path to manifest.jsonl (must contain labels_pos)")
    parser.add_argument("--out", required=True, help="Output directory (writes runs/<case_id>/run.json)")
    parser.add_argument("--max-cases", type=int, default=0, help="Max number of cases to run (0 means all selected rows)")
    parser.add_argument(
        "--allow-empty-labels",
        action="store_true",
        help="Include rows with empty labels_pos (will likely fail report generation for those rows).",
    )
    args = parser.parse_args()

    result = batch_ct_rate_report_from_labels(
        manifest_jsonl=str(args.manifest_jsonl),
        out_dir=str(args.out),
        max_cases=int(args.max_cases),
        require_nonempty_labels=(not bool(args.allow_empty_labels)),
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
