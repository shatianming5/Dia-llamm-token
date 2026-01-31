from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .ct_rate_label_eval import compute_ct_rate_label_metrics


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_manifest_index(path: Path) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            continue
        cid = str(obj.get("case_id", "")).strip()
        if cid:
            idx[cid] = obj
    return idx


def batch_label_eval(
    *,
    manifest_jsonl: str,
    runs_dir: str,
    out_dir: str,
) -> Dict[str, Any]:
    manifest_path = Path(manifest_jsonl)
    runs_path = Path(runs_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    idx = _load_manifest_index(manifest_path)

    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    # Support both `runs/<case_id>/run.json` and `runs/*.json` layouts (best-effort).
    run_files = list(runs_path.rglob("run.json"))
    if not run_files:
        run_files = list(runs_path.glob("*.json"))
    for run_file in sorted(run_files, key=lambda p: str(p)):
        try:
            run = _load_json(run_file)
            cid = str(run.get("case_id", "")).strip()
            if not cid:
                cid = run_file.parent.name
            if not cid:
                raise ValueError("missing case_id")
            if cid not in idx:
                raise KeyError(f"case_id not found in manifest: {cid}")

            metrics = compute_ct_rate_label_metrics(run, idx[cid])
            rows.append(metrics)
        except Exception as exc:  # noqa: BLE001
            failures.append({"run_path": str(run_file), "error": str(exc)})

    out_jsonl = out_path / "label_metrics.jsonl"
    out_jsonl.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    summary = {
        "timestamp_utc": _utc_now_iso(),
        "manifest_jsonl": str(manifest_path),
        "runs_dir": str(runs_path),
        "n_ok": int(len(rows)),
        "n_failed": int(len(failures)),
        "failures": failures,
        "label_metrics_jsonl": str(out_jsonl),
    }
    (out_path / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if failures:
        raise RuntimeError(f"{len(failures)} run(s) failed: {failures[:3]}")

    return {"out_dir": str(out_path), "label_metrics_jsonl": str(out_jsonl), "n": int(len(rows))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch CT-RATE label eval for a directory of run.json files.")
    parser.add_argument("--manifest", dest="manifest_jsonl", required=True, help="Manifest JSONL (must contain labels_pos)")
    parser.add_argument("--runs", dest="runs_dir", required=True, help="Directory containing per-case run.json")
    parser.add_argument("--out", required=True, help="Output directory (writes label_metrics.jsonl)")
    args = parser.parse_args()

    result = batch_label_eval(
        manifest_jsonl=str(args.manifest_jsonl),
        runs_dir=str(args.runs_dir),
        out_dir=str(args.out),
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

