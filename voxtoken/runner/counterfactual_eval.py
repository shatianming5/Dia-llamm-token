from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from ..eval.counterfactuals import run_counterfactuals


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _infer_case_id_from_run_json(path: Path) -> str:
    try:
        run = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    if isinstance(run, dict):
        cid = str(run.get("case_id", "")).strip()
        if cid:
            return cid
        meta = run.get("meta", {})
        if isinstance(meta, dict):
            inp = meta.get("input", {})
            if isinstance(inp, dict):
                cid = str(inp.get("case_id", "")).strip()
                if cid:
                    return cid
    return ""

def _write_counterfactual_csv(path: Path, rows: Any) -> None:
    if not isinstance(rows, list):
        return
    cols = ["cf_type", "slot_f1_micro", "ground_hit@0.1", "unsupported_sent_pct"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            if not isinstance(r, dict):
                continue
            w.writerow({c: r.get(c, "") for c in cols})


def counterfactual_eval(
    run_json: str,
    out_dir: str,
    *,
    gt: str | None = None,
    manifest: str | None = None,
    case_id: str | None = None,
) -> Dict[str, Any]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cfg: Dict[str, Any] = {"run_json": run_json}
    if gt:
        cfg["gt"] = str(gt)
    if manifest:
        cfg["manifest"] = str(manifest)
        if case_id:
            cfg["case_id"] = str(case_id)
        else:
            inferred = _infer_case_id_from_run_json(Path(run_json))
            if inferred:
                cfg["case_id"] = str(inferred)

    payload = run_counterfactuals(cfg)
    _write_counterfactual_csv(out_path / "counterfactual.csv", payload.get("rows", []))

    (out_path / "counterfactuals.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_path / "counterfactuals.jsonl").write_text(
        json.dumps(payload, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary = {
        "timestamp_utc": _utc_now_iso(),
        "in_run_json": str(run_json),
        "out_dir": str(out_path),
    }
    (out_path / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return {"out_dir": str(out_path), "counterfactuals_path": str(out_path / "counterfactuals.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Counterfactual evaluation runner (citation perturbations).")
    parser.add_argument("--in", dest="run_json", required=True, help="Path to run.json")
    parser.add_argument(
        "--out",
        default="",
        help="Output directory (default: <run.json dir>/counterfactual, per docs/plan.md).",
    )
    parser.add_argument("--gt", default=None, help="Optional GT boxes JSON (RadGenome-style boxes_by_sent_mm)")
    parser.add_argument("--manifest", default=None, help="Optional JSONL manifest to load GT boxes for this case")
    parser.add_argument("--case-id", default=None, help="Case ID for --manifest (defaults to run.json case_id)")
    args = parser.parse_args()

    out_dir = args.out
    if not out_dir:
        out_dir = str(Path(args.run_json).resolve().parent / "counterfactual")

    result = counterfactual_eval(
        args.run_json,
        out_dir,
        gt=(str(args.gt).strip() if args.gt else None),
        manifest=(str(args.manifest).strip() if args.manifest else None),
        case_id=(str(args.case_id).strip() if args.case_id else None),
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
